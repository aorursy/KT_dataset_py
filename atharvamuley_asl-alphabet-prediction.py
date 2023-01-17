#import libraries
import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt

from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.models import Sequential
from keras.optimizers import  Adam
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator

%config InlineBackend.figure_format = 'svg'
#Fetch Data from directory using ImageDataGenerator
data_dir = "/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train"
target_size = (128, 128)
target_dims = (128, 128, 3) # add channel for RGB
n_classes = 29
val_frac = 0.1
batch_size = 64

data_augmentor = ImageDataGenerator(samplewise_center=True, 
                                    samplewise_std_normalization=True, 
                                    validation_split=val_frac,
                                    )

train_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, shuffle=True, subset="training")
val_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, subset="validation")

# Define the model
model = Sequential()
model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu', input_shape=target_dims))
model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides = 2))
model.add(Conv2D(128, kernel_size=3, strides=1, activation='relu'))
model.add(Conv2D(128, kernel_size=3, strides=1, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides = 2))
model.add(Conv2D(256, kernel_size=3, strides=1, activation='relu'))
model.add(Conv2D(256, kernel_size=3, strides=1, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides = 2))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer = regularizers.l2()))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu', kernel_regularizer = regularizers.l2()))
model.add(Dense(n_classes, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
#Train the model
history = model.fit_generator(train_generator, epochs=15, validation_data=val_generator)
# Plot Model's Train v/s Validation Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Acuracy")
plt.legend(['Train', 'Validation'])
plt.show()
# Plot Model's Train v/s Validation Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(['Train', 'Validation'])
plt.show()