import numpy as np # linear algebra
import matplotlib.pyplot as plt 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, BatchNormalization

from PIL import Image, ImageOps
# kaggle/input/digits.jpeg
# DATA 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

plt.imshow(x_train[40], cmap="gray")
plt.show()
# MODEL 
model = Sequential()
model.add(Flatten())
model.add(Dense(392, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(196, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))
# from keras.preprocessing.image import ImageDataGenerator

# datagen = ImageDataGenerator(rotation_range=90)
# fit parameters from data
#datagen.fit(x_train.reshape((60000,28,28,1)))

# print(datagen)
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=20, shuffle=True, validation_split=0.25)
plt.figure(figsize=(10,6))
plt.subplot(2,2,1)
plt.plot(range(len(history.history['accuracy'])), history.history['accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.subplot(2,2,2)
plt.plot(range(len(history.history['loss'])), history.history['loss'])
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()
model.evaluate(x_test, y_test)
plt.imshow(x_test[88], cmap="gray")
pred = model.predict(x_test[88].reshape(1, 28, 28, 1))
print(pred.argmax())
img = Image.open("/kaggle/input/exampledigit2/ex_2.jpeg")
img = img.crop((80, 100, 200, 230))
plt.imshow(img, cmap="gray")
gray = ImageOps.grayscale(img)
inverted = ImageOps.invert(gray)
resized = inverted.resize((28, 28), Image.LANCZOS)
test_digit = np.asarray(resized)
plt.imshow(test_digit, cmap="gray")
pred = model.predict(test_digit.reshape(1, 28, 28, 1))
print(pred.argmax())
img = Image.open("/kaggle/input/example-digit/ex.jpeg")
img = img.crop((80, 90, 180, 250))
plt.imshow(img, cmap="gray")
gray = ImageOps.grayscale(img)
inverted = ImageOps.invert(gray)
resized = inverted.resize((28, 28), Image.LANCZOS)
test_digit = np.asarray(resized)
plt.imshow(test_digit, cmap="gray")
pred = model.predict(test_digit.reshape(1, 28, 28, 1))
print(pred.argmax())
img = Image.open("/kaggle/input/exampledigit3/ex_3.jpeg")
img = img.crop((50, 70, 150, 220))
plt.imshow(img, cmap="gray")
gray = ImageOps.grayscale(img)
inverted = ImageOps.invert(gray)
resized = inverted.resize((28, 28), Image.LANCZOS)
test_digit = np.asarray(resized)
plt.imshow(test_digit, cmap="gray")
pred = model.predict(test_digit.reshape(1, 28, 28, 1))
print(pred.argmax())
img = Image.open("/kaggle/input/tresdigit/tres.png")

gray = ImageOps.grayscale(img)
inverted = ImageOps.invert(gray)
test_digit = np.asarray(inverted)
plt.imshow(test_digit, cmap="gray")
plt.imshow(test_digit, cmap="gray")
pred = model.predict(test_digit.reshape(1, 28, 28, 1))
print(pred.argmax())