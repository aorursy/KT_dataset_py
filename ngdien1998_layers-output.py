def visualize(img_type, layer_name, img):

    layer_output = model.get_layer(layer_name).output

    intermediate_model = Model(inputs=model.input, outputs=layer_output)

    

    intermediate_prediction = intermediate_model.predict(img)

    size = np.shape(intermediate_prediction)[3]

    

    col_size = size // 8

    row_size = 8

    

    fig, ax = plt.subplots(row_size, col_size, figsize=(15, 10))

    for i in range(size):

        ax[i // col_size][i % col_size].imshow(intermediate_prediction[0, :, :, i], cmap='gray')

        ax[i // col_size][i % col_size].axis("off")

        ax[i // col_size][i % col_size].set_xticklabels([])

        ax[i // col_size][i % col_size].set_yticklabels([])

        

    plt.savefig("{}_{}.png".format(img_type, layer_name))

    

def preprocess_image(file_name, target_size=(224, 224)):

    img = cv2.imread(file_name)

    img = cv2.resize(img, target_size)

    if img.shape[2] == 1:

        img = np.dstack([img, img, img])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0

    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])



    return img
import cv2

import sys

import numpy as np

import matplotlib.pyplot as plt

from keras.models import load_model, Model, Sequential 

from keras.applications.vgg16 import preprocess_input

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from numpy import expand_dims

from keras.layers import Dense, Dropout, Flatten, SeparableConv2D, Conv2D, MaxPooling2D, BatchNormalization



model = load_model("../input/best-model-751792h5/best_model_751792.h5")

model.summary()

idxs = [1, 2, 4, 5, 7, 9, 11, 13, 15, 17]



# Pneumonia image

print("Pneumonia image")

img_path = "../input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/person1_virus_13.jpeg"

img = preprocess_image(img_path)

for idx in idxs:

    visualize("pneumonia", model.layers[idx].name, img.copy())



# Normal image

print("Normal image")

img_path = "../input/chest-xray-pneumonia/chest_xray/test/NORMAL/IM-0006-0001.jpeg"

img = preprocess_image(img_path)

for idx in idxs:

    visualize("normal", model.layers[idx].name, img.copy())