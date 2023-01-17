import numpy as np # linear algebra

import cv2



import os

import random

import pickle

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras.layers import (

    Conv2D, MaxPooling2D, 

    Dense, Dropout, 

    Flatten)

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.image import ImageDataGenerator
print(os.listdir("../input/cell_images/cell_images"))
IMG_SIZE = 50
CATEGORIES = ['Parasitized', 'Uninfected']

dataset = []



def generate_data():

    for category in CATEGORIES:

        path = f'../input/cell_images/cell_images/{category}'

        class_id = CATEGORIES.index(category)

        for image in os.listdir(path):

            try:

                image_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_COLOR)

                image_array = cv2.resize(image_array, (IMG_SIZE , IMG_SIZE))

                dataset.append([image_array, class_id])

            except Exception as e:

                print(e)

    random.shuffle(dataset)

                

generate_data()
print(len(dataset))
data = []

labels = []

for features, label in dataset:

    data.append(features)

    labels.append(label)
data = np.array(data)

data.reshape(-1, 50, 50, 3)
train_data, data, train_labels, labels = train_test_split(data, 

                                                          labels,

                                                          test_size=0.15)



test_data, validation_data, test_labels, validation_labels = train_test_split(data, 

                                                                    labels,

                                                                    test_size=0.7)
plt.figure(figsize=(10, 10))

i = 0

for i in range(25):

    plt.subplot(5, 5, i+1)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(test_data[i])

    if(test_labels[i] == 0):

        plt.xlabel('Infected')

    else:

        plt.xlabel('Uninfected')

    i += 1

plt.show()
datagen_train = ImageDataGenerator(rescale=1./255,

                            rotation_range=45,

                            width_shift_range=0.2,

                            height_shift_range=0.2,

                            shear_range=0.2,

                            zoom_range=0.2,

                            horizontal_flip=True)



datagen_test = ImageDataGenerator(rescale=1./255)

datagen_validation = ImageDataGenerator(rescale=1./255)
datagen_train.fit(train_data)

datagen_test.fit(test_data)

datagen_test.fit(validation_data)

model = Sequential([

    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),

    MaxPooling2D((2, 2)),

    

    Conv2D(64, (3, 3), activation="relu"),

    MaxPooling2D((2, 2)),

    

    Conv2D(128, (3, 3), activation="relu"),

    MaxPooling2D((2, 2)),

    

    Conv2D(256, (3, 3), activation="relu"),

    MaxPooling2D((2, 2)),

    

    Flatten(),

    Dense(256, activation="relu"),

    Dense(2, activation='softmax')

])
model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])
BATCH_SIZE = 32

epochs = 30

history = model.fit_generator(datagen_train.flow(train_data, train_labels, batch_size=BATCH_SIZE),

                   steps_per_epoch=len(train_data) / BATCH_SIZE,

                   epochs=epochs,

                   validation_data=datagen_validation.flow(validation_data, 

                                                     validation_labels, batch_size=BATCH_SIZE),

                    

                   )
accuracy = history.history['acc']

loss = history.history['loss']

val_accuracy = history.history['val_acc']

val_loss = history.history['val_loss']



print(f'Training Accuracy: {np.max(accuracy)}')

print(f'Training Loss: {np.min(loss)}')

print(f'Validation Accuracy: {np.max(val_accuracy)}')

print(f'Validation Loss: {np.min(val_loss)}')
epochs_range = range(epochs)



plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, accuracy, label="Training Accuracy")

plt.plot(epochs_range, val_accuracy, label="Validation Accuracy")

plt.legend(loc="lower right")

plt.title("Training and Validation Accuracy")



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label="Training Loss")

plt.plot(epochs_range, val_loss, label="Validation Loss")

plt.legend(loc="upper right")

plt.title("Training and Validation Loss")

plt.show()
class_names = ['Infected', 'Uninfected']

def plot_images(i, predictions_array, true_labels, images):

    predictions_array, true_label, img = predictions_array[i], true_labels[i],images[i]

    plt.grid(False)

    plt.xticks([])

    plt.yticks([])

    

    plt.imshow(img)

    

    predicted_label = np.argmax(predictions_array)

        

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],

                                        100*np.max(predictions_array),

                                        class_names[true_label]))
random.shuffle(test_data)

predictions = model.predict(test_data)
num_rows = 8

num_cols = 6

num_images = num_rows * num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):

    plt.subplot(num_rows, 2*num_cols, 2*i+1)

    plot_images(i, predictions, test_labels, test_data)