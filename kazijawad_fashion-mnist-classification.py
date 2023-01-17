from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


print("Training Data Shape:", x_train.shape)
print()
print("Training Sample Count:", len(x_train))
print("Training Label Count:", len(y_train))
print("Testing Sample Count:", len(x_test))
print("Testing Label Count:", len(y_test))
print()
print("Training Data Dimensions:", x_train[0].shape)
print("Training Data Labels:", y_train.shape[0])
print()
print("Testing Data Dimensions:", x_test[0].shape)
print("Testing Data Labels:", y_test.shape[0])
import matplotlib.pyplot as plt
import numpy as np

plt.subplot(331)
random_num = np.random.randint(0, len(x_train))
plt.imshow(x_train[random_num], cmap=plt.get_cmap("gray"))

plt.subplot(332)
random_num = np.random.randint(0, len(x_train))
plt.imshow(x_train[random_num], cmap=plt.get_cmap("gray"))

plt.subplot(333)
random_num = np.random.randint(0, len(x_train))
plt.imshow(x_train[random_num], cmap=plt.get_cmap("gray"))

plt.subplot(334)
random_num = np.random.randint(0, len(x_train))
plt.imshow(x_train[random_num], cmap=plt.get_cmap("gray"))

plt.subplot(335)
random_num = np.random.randint(0, len(x_train))
plt.imshow(x_train[random_num], cmap=plt.get_cmap("gray"))

plt.subplot(336)
random_num = np.random.randint(0, len(x_train))
plt.imshow(x_train[random_num], cmap=plt.get_cmap("gray"))

plt.show()
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
import keras

# Training Parameters
batch_size = 128
epochs = 3

img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)

# Convert to proper type
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Normalize values
x_train /= 255
x_test /= 255

# One-Hot Encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]

# Create model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=input_shape))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3,3), activation="relu"))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())

model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
print(model.summary())
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])