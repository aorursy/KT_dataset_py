!pip install -q wandb

!pip install -q fastparquet
!wandb --version
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import keras

from keras.utils import np_utils

from keras.utils.np_utils import to_categorical

from keras.datasets import cifar10

from keras.datasets import mnist

import glob

import matplotlib

from matplotlib import pyplot as plt

import matplotlib.image as mpimg

import numpy as np

import imageio as im



# Load MNIST

(X_train_val, Y_train_val), (X_test, Y_test) = mnist.load_data()



# Data Prep

X_train_val = X_train_val / 255.0

X_test = X_test / 255.0

X_train_val = X_train_val.reshape(-1,28,28,1)

X_test = X_test.reshape(-1,28,28,1)

Y_train_val = to_categorical(Y_train_val, num_classes = 10)

annealer = keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x, verbose=0)

styles=[':','-.','--','-',':','-.','--','-',':','-.','--','-']



# WandB

import wandb

from wandb.keras import WandbCallback

# You can change your project name here. For more config options, see https://docs.wandb.com/docs/init.html

wandb.init(project="building-neural-nets", name="mnist")

labels=[str(i) for i in range(10)]



# Go to https://app.wandb.ai/authorize to get your WandB key
# Log Images

wandb.log({"summary_table": wandb.Html('<img src="https://lumos642.files.wordpress.com/2019/08/summary_table-8.png" width="100%">')})

wandb.log({"activation_map": wandb.Html('<img src="https://lumos642.files.wordpress.com/2019/08/activation_map-4.png" width="100%">')})

wandb.log({"misclassified_activation_map": wandb.Html('<img src="https://lumos642.files.wordpress.com/2019/08/misclassified_activation_map-1.png" width="100%">')})

wandb.log({"feature_maps_4": wandb.Html('<img src="https://lumos642.files.wordpress.com/2019/08/4.png" width="100%">')})

wandb.log({"feature_maps_5": wandb.Html('<img src="https://lumos642.files.wordpress.com/2019/08/5.png" width="100%" width="100%">')})

wandb.log({"feature_maps_8": wandb.Html('<img src="https://lumos642.files.wordpress.com/2019/08/8.png" width="100%">')})
nets = 1

model = [0] *nets



# Define model architecture

j=0

model[j] = keras.models.Sequential()

model[j].add(keras.layers.Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))

model[j].add(keras.layers.BatchNormalization())

model[j].add(keras.layers.Conv2D(32,kernel_size=3,activation='relu'))

model[j].add(keras.layers.BatchNormalization())

model[j].add(keras.layers.Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))

model[j].add(keras.layers.BatchNormalization())

model[j].add(keras.layers.Dropout(0.4))



model[j].add(keras.layers.Conv2D(64,kernel_size=3,activation='relu'))

model[j].add(keras.layers.BatchNormalization())

model[j].add(keras.layers.Conv2D(64,kernel_size=3,activation='relu'))

model[j].add(keras.layers.BatchNormalization())

model[j].add(keras.layers.Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))

model[j].add(keras.layers.BatchNormalization())

model[j].add(keras.layers.Dropout(0.4))



model[j].add(keras.layers.Flatten())

model[j].add(keras.layers.Dense(128, activation='relu'))

model[j].add(keras.layers.BatchNormalization())

model[j].add(keras.layers.Dropout(0.4))

model[j].add(keras.layers.Dense(10, activation='softmax'))



model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
%%wandb

# Train model

X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size = 0.2)

history = [0] * nets

epochs = 30



j=0

history[j] = model[j].fit(X_train,Y_train, batch_size=64, epochs = epochs,  

    validation_data = (X_val,Y_val), verbose=0,

    callbacks=[WandbCallback(validation_data=(X_val,Y_val), input_type="image", output_type="label",

                             log_evaluation=True, labels=[str(i) for i in range(10)])])
# Extract the outputs of the layers

layer_outputs = [layer.output for layer in model[j].layers]

activation_model = keras.models.Model(inputs=model[j].input, outputs=layer_outputs)
# Get instance to plot activations for

img_tensor = X_val[4:5]

img_tensor.shape
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]

print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
# Create activation maps for all layers

classifier = model[j]

layer_names = []

for layer in classifier.layers[:12]:

    layer_names.append(layer.name)

    

images_per_row = 16



for layer_name, layer_activation in zip(layer_names, activations):

    n_features = layer_activation.shape[-1]

    size = layer_activation.shape[1]

    n_cols = n_features // images_per_row

    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols):

        for row in range(images_per_row):

            channel_image = layer_activation[0,

                                             :, :,

                                             col * images_per_row + row]

            channel_image -= channel_image.mean()

            channel_image /= channel_image.std()

            channel_image *= 64

            channel_image += 128

            channel_image = np.clip(channel_image, 0, 255).astype('uint8')

            display_grid[col * size : (col + 1) * size, # Displays the grid

                         row * size : (row + 1) * size] = channel_image

    scale = 1. / size

    plt.figure(figsize=(scale * display_grid.shape[1],

                        scale * display_grid.shape[0]))

    plt.title(layer_name)

    plt.grid(False)

    plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.savefig('img.png')