import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
train_data = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")



test_data = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_test.csv")

train_data.head()
print(train_data.shape)

print(test_data.shape)
# Converting dataframe values to array as keras expects inputs to be arrays



train_data = np.array(train_data, dtype = 'uint8')

test_data = np.array(test_data, dtype = 'uint8')
# Let's make validation set from test set

from sklearn.model_selection import train_test_split



test_images, val_images, test_labels, val_labels = train_test_split(test_data[:,1:], 

                                                        test_data[:,0], test_size = 0.3, 

                                                        random_state = 53, shuffle = True)
print(test_images.shape)

print(test_labels.shape)

print(val_images.shape)

print(val_labels.shape)
(train_images, train_labels) = train_data[:,1:],train_data[:,0]
train_images = train_images/255.0

val_images = val_images/255.0
# Let's see some of the images



class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



plt.figure(figsize = (10,10))

for i in range(15):

    ax = plt.subplot(5, 3, i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_images[i].reshape(28,28))

    plt.title(class_names[int(train_labels[i])])

    
# Setting image_shape and batch_size

image_shape = (28, 28, 1)

batch_size = 1000
# Reshaping the train, test and validation data

train_images = train_images.reshape(train_images.shape[0], *image_shape)

val_images = val_images.reshape(val_images.shape[0], *image_shape)

test_images = test_images.reshape(test_images.shape[0], *image_shape)
num_classes = 10



model = Sequential([

  layers.Conv2D(16, 3, padding='same', activation='relu', input_shape = image_shape),

  layers.MaxPooling2D(),

  layers.Conv2D(32, 3, padding='same', activation='relu'),

  layers.MaxPooling2D(),

  layers.Conv2D(64, 3, padding='same', activation='relu'),

  layers.MaxPooling2D(),

  layers.Dropout(0.2),

  layers.Flatten(),

  layers.Dense(128, activation='relu'),

  layers.Dense(num_classes, activation = 'softmax')

])
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
model.summary()
epochs = 20



history = model.fit(train_images, train_labels,

                    epochs = epochs, 

                    verbose = 1,

                    batch_size = batch_size,

                    validation_data = (val_images, val_labels))
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_range = range(epochs)



plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
predicted_labels = model.predict_classes(test_images)





true_labels = test_labels



from sklearn.metrics import classification_report

target_names = ["Class {}".format(i) for i in range(num_classes)]

print(classification_report(true_labels, predicted_labels, target_names = target_names))
fig, axes = plt.subplots(5, 5, figsize = (12,12))

axes = axes.ravel()



for i in np.arange(0, 25):  

    axes[i].imshow(test_images[i].reshape(28,28))

    axes[i].set_title(f"Predicted Class = {predicted_labels[i]: 0.1f}\n Original Class = {test_labels[i]: 0.1f}")

    axes[i].axis('off')



plt.subplots_adjust(wspace=0.5)