import numpy as np 
import pandas as pd 
import os
import keras
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras import backend as K
import matplotlib.pyplot as plt
from keras.preprocessing import image
%matplotlib inline
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 12

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


fig = plt.figure(figsize=(28,28))
for i in range (5):
    im = fig.add_subplot(1, 5, i+1, xticks=[], yticks=[])
    im.imshow(x_train[i], cmap="gray")
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels) #Converts a class vector (integers) to binary class matrix.
test_labels = to_categorical(test_labels)
X_train = train_images[6000:50000, :, :]
X_train_labels = train_labels[6000:50000, :]
X_valid = train_images[1000:6000, :, :]
X_valid_labels= train_labels[1000:6000, :]
X_test = test_images[:1000, :, :]
X_test_labels = test_labels[:1000, :]

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer=SGD(0.001),loss="categorical_crossentropy",metrics=["accuracy"])

model.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])
history = model.fit(X_train, X_train_labels, epochs=30, batch_size=64, validation_data=(X_valid, X_valid_labels))
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend(loc='best')

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc='best');
for i in range(1, 10):
    img = X_test[i]
    img_class = model.predict_classes(X_test)
    prediction = img_class[i]
    classname = img_class[i]
    print("Class: ",classname)
    img = img.reshape((28,28))
    plt.imshow(img)
    plt.title(classname)
    plt.show()