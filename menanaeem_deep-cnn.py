import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import os



train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')

train = np.array(train)

test  = np.array(test)
train_images = train[:35000,1:]

train_labels = train[:35000,0]

validation_images  = train[35000:,1:]

validation_labels  = train[35000:,0]





print(train_images.shape)

print(train_labels.shape)

print(validation_images.shape)

print(validation_labels.shape)
from keras import layers

from keras import models



model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))
from keras.utils import to_categorical



train_images = train_images.reshape((-1, 28, 28, 1))

train_images = train_images.astype('float32') / 255



validation_images = validation_images.reshape((-1, 28, 28, 1))

validation_images = validation_images.astype('float32') / 255



test = test.reshape((-1, 28, 28, 1))

test = test.astype('float32') / 255



train_labels = to_categorical(train_labels)

validation_labels = to_categorical(validation_labels)
model.compile(optimizer='rmsprop',

              loss='categorical_crossentropy',

              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, batch_size=64)
test_loss, test_acc = model.evaluate(validation_images, validation_labels)

test_acc
predicted = [i.argmax() for i in model.predict(test)]

output = pd.read_csv("../input/sample_submission.csv")

output['Label'] = predicted
output.tail()
for i in range(5):

    plt.subplot(1,5,i+1)

    plt.imshow(test[-5+i].reshape(28,28))


output.to_csv('sub.csv', index=False)
model.save_weights('model.h5')