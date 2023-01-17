import numpy as np

import pandas as pd

%matplotlib inline

from matplotlib import pyplot as plt

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from keras import optimizers

from keras.utils import to_categorical

from sklearn.utils import shuffle

import random



np.random.seed(1337)



import os

print(os.listdir("../input"))
train_dir = "../input/"

train_img = np.load(train_dir + "kmnist-train-imgs.npz")['arr_0']

train_lbl = np.load(train_dir + "kmnist-train-labels.npz")['arr_0']

test_images = np.load(train_dir + "kmnist-test-imgs.npz")['arr_0']

test_labels = np.load(train_dir + "kmnist-test-labels.npz")['arr_0']

print(train_img.shape)

print(train_lbl.shape)

print(test_images.shape)

print(test_labels.shape)
train_img, train_lbl = shuffle(train_img, train_lbl, random_state = 0)

train_images = train_img[:50000]

train_labels = train_lbl[:50000]

validation_images = train_img[50000:]

validation_labels = train_lbl[50000:]
train_images = train_images.reshape(50000, 28, 28, 1)

validation_images = validation_images.reshape(10000, 28, 28, 1)

test_images = test_images.reshape(10000, 28, 28, 1)
train_labels = to_categorical(train_labels)

validation_labels = to_categorical(validation_labels)

test_labels = to_categorical(test_labels)
def plot_image(arr):

    plt.imshow(data, interpolation='nearest')

    plt.show()
def create_model():

    model = Sequential()

    

    model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (28,28,1)))

    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, (3,3), activation = 'relu'))

    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(128, (3,3), activation = 'relu'))

    model.add(MaxPooling2D((2,2)))

    

    model.add(Flatten())

    model.add(Dense(128, activation = 'relu'))

    model.add(Dense(10, activation = 'softmax'))

    

    return model
model = create_model()

model.summary()
opt = optimizers.RMSprop(lr = 1e-4)

model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['acc'])
history = model.fit(train_images, train_labels, epochs = 20, batch_size = 10, 

                    validation_data = (validation_images, validation_labels), shuffle = True)
model.save("kmnist_cnn.h5")
hist = history.history

print(hist.keys())
accuracy = hist['acc']

loss = hist['loss']

val_accuracy = hist['val_acc']

val_loss = hist['val_loss']
len(accuracy)
epochs = [i for i in range(1, 21)]
plt.plot(epochs, accuracy)

plt.plot(epochs, val_accuracy)

plt.title("Accuracy vs Val Accuracy")

plt.legend(["Accuracy", "Validation Accuracy"])

plt.show()
l, a = model.evaluate(test_images, test_labels)
print(l, a)