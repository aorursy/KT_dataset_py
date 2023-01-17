import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing import image

np.seterr(divide='ignore', invalid='ignore')

import os

print(os.listdir("../input"))

from keras.models import load_model

model = load_model('../input/cats-dogs-wo-pretrain-nw/catndogs_wo_pretrain_nw.h5')
model.summary()
img = image.load_img('../input/dogpic/dog.jpg', target_size=(128,128))

img_tensor = image.img_to_array(img)

img_tensor = np.expand_dims(img_tensor, axis=0)

img_tensor /= 255.
import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])

plt.xlabel('The test dog picture')

plt.show()
from keras import models

layer_outputs = [layer.output for layer in model.layers[:8]] #Extracts the ouput of the top eight layers

activation_model = models.Model(inputs=model.input, outputs=layer_outputs) #creates a model that will return these outputs
activations = activation_model.predict(img_tensor) 

first_layer_activation = activations[0]#activation of the first convolution layer for the cat image input

print(first_layer_activation.shape) #It’s a 126 x 126 feature map with 32 channels
import matplotlib.pyplot as plt

plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
import math

# defining number of layers

layer_names = []

for layer in model.layers[:8]:

    layer_names.append(layer.name)

    

images_per_row = 16    



# defining number of features in the feature map

for layer_name, layer_activation in zip(layer_names, activations):

    n_features = layer_activation.shape[-1]

    size = layer_activation.shape[1]

    n_cols = n_features // images_per_row

    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols):

        for row in range(images_per_row):

            channel_image = layer_activation[0,:, :,col * images_per_row + row]

            channel_image = (channel_image-channel_image.mean())//channel_image.std()  

            channel_image *= 64

            channel_image += 128

            channel_image = np.clip(channel_image, 0, 255).astype('uint8')

            display_grid[col * size : (col + 1) * size,

            row * size : (row + 1) * size] = channel_image

    scale = 1. / size

    plt.figure(figsize=(scale * display_grid.shape[1],

    scale * display_grid.shape[0]))

    plt.title(layer_name)

    plt.grid(False)

    plt.imshow(display_grid, aspect='auto', cmap='viridis')



    