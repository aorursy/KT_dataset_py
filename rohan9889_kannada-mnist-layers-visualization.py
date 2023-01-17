import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from keras.models import Sequential, Model

from keras.layers import Conv2D, Dense, Dropout, BatchNormalization, Flatten, MaxPool2D
model = Sequential()



model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))

model.add(Conv2D(32,kernel_size=3,activation='relu'))

model.add(MaxPool2D())

model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))

model.add(Dropout(0.4))



model.add(Conv2D(64,kernel_size=3,activation='relu'))

model.add(Conv2D(64,kernel_size=3,activation='relu'))

model.add(MaxPool2D())

model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))

model.add(Dropout(0.4))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(10, activation='softmax'))



model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.load_weights('/kaggle/input/kannada-mnist-model-weights/BWeight.md5')
data = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
X = data.drop(['id'],axis=1)
X = X.values.reshape(X.shape[0],28,28,1)
layer_outputs = [layer.output for layer in model.layers]

# Storing layers of model in a list

activation_model = Model(inputs=model.input, outputs=layer_outputs)

# A simple model that takes its input as input of previously trained model and produces output based on provided 

# list of layers
activations = activation_model.predict(X[0:10])
plt.matshow(activations[0][9, :, :, 10], cmap='viridis')
layer_names = []

for layer in model.layers[:6]:

    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot

    

images_per_row = 16



for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps

    n_features = layer_activation.shape[-1] # Number of features in the feature map

    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).

    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix

    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols): # Tiles each filter into a big horizontal grid

        for row in range(images_per_row):

            channel_image = layer_activation[0,

                                             :, :,

                                             col * images_per_row + row]

            channel_image = np.clip(channel_image, 0, 255).astype('uint8')

            display_grid[col * size : (col + 1) * size, # Displays the grid

                         row * size : (row + 1) * size] = channel_image

    scale = 1 / size

    plt.figure(figsize=(scale * display_grid.shape[1],

                        scale * display_grid.shape[0]))

    plt.title(layer_name)

    plt.grid(False)

    plt.imshow(display_grid, aspect='auto')