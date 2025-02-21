import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("wound_classification_model.h5")


# Define a function to predict the wound type for a given image file
def predict_wound_type(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    prediction = model.predict(img_array)
    # Get the class index with the highest probability
    predicted_class_index = np.argmax(prediction)
    # Mapping of class index to wound type
    class_mapping = {
        0: "Abrasions",
        1: "Bruises",
        2: "Cut",
        3: "Laceration",
        4: "Stab Wound",
    }
    # Get the predicted wound type
    predicted_wound_type = class_mapping[predicted_class_index]
    return predicted_wound_type


# Example usage
image_path = "dataset/Abrasions/abrasions (1).jpg"
predicted_type = predict_wound_type(image_path)
print("Predicted Wound Type:", predicted_type)
