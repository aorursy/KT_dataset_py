import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import Sequential
from tensorflow.keras.layers import Flatten, Dense

data = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()
train_images.shape, train_labels.shape, test_images.shape, test_labels.shape
plt.imshow(train_images[0])
train_images = train_images / 255.0
test_images  = test_images  / 255.0
model = Sequential()
model.add(Flatten(input_shape = (28,28)))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(10 , activation = 'softmax'))
model.compile(
                  optimizer = 'adam',
                  loss      = 'sparse_categorical_crossentropy',
                  metrics   = ['accuracy']
             )
model.fit(train_images, train_labels, epochs = 10, validation_data = (test_images, test_labels))
prediction = model.predict(test_images)
prediction[0]
np.argmax(prediction[0])
test_labels[0]
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i])
    plt.title(test_labels[i])
    plt.xlabel(np.argmax(prediction[i]))
plt.show()
