import tensorflow as tf

import matplotlib.pyplot as plt
# import pandas as pd
# import scipy.ndimage
import cv2
import numpy as np

# plt.rcParams["figure.figsize"] = (10, 5)
# plt.rcParams["figure.dpi"] = 300
inputs = tf.keras.layers.Input((512, 512, 1))
conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
conv4 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
drop4 = tf.keras.layers.Dropout(0.5)(conv4)
pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = tf.keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5 = tf.keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
drop5 = tf.keras.layers.Dropout(0.5)(conv5)

up6 = tf.keras.layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop5))
merge6 = tf.keras.layers.concatenate([drop4, up6], axis = 3)
conv6 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

up7 = tf.keras.layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv6))
merge7 = tf.keras.layers.concatenate([conv3, up7], axis = 3)
conv7 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

up8 = tf.keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv7))
merge8 = tf.keras.layers.concatenate([conv2, up8], axis = 3)
conv8 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

up9 = tf.keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv8))
merge9 = tf.keras.layers.concatenate([conv1, up9], axis = 3)
conv9 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv9 = tf.keras.layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv10 = tf.keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)

model = tf.keras.models.Model(inputs = inputs, outputs = conv10)
model
model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-4), \
              loss = tf.keras.losses.binary_crossentropy, \
              metrics = ['accuracy'])
image = plt.imread("../input/u-net-data/train.png")
plt.imshow(image)
polygons = []
with open("../input/u-net-data/contours.csv") as f:
    for row in f.readlines():
        row = row.replace("[", "").replace("]", "").replace("Point:", "")
        polygons.append(np.array([float(i) for i in row.split(",")]).round().reshape(-1, 2).astype(np.int32))

print(len(polygons))
plt.imshow(image)
for i in polygons:
    plt.gca().add_patch(plt.Polygon(i))
mask = np.zeros(image.shape[0: 2])
mask.shape
cv2.fillPoly(mask, polygons, color=(1, 1, 1))
plt.imshow(mask, cmap="gray")
plt.imsave("mask.png", mask, cmap="gray")
import random
def TrainingsetGenerator(X, Y, size):
    while True:
        row = random.randint(0, X.shape[0] - size[0])
        column = random.randint(0, X.shape[1] - size[1])
        yield X[row: row + size[0], column: column + size[1]],Y[row: row + size[0], column: column + size[1]]
g = TrainingsetGenerator(image[..., 0], mask, (512, 512))
xy = next(g)
plt.subplot(121)
plt.imshow(xy[0], cmap="gray")
plt.subplot(122)
plt.imshow(xy[1], cmap="gray")
x_train = []
y_train = []
trainingsetGenerator = TrainingsetGenerator(image[..., 0], mask, (512, 512))
for i in range(500):
    train = next(trainingsetGenerator)
    x_train.append(train[0].reshape(512, 512, 1))
    y_train.append(train[1].reshape(512, 512, 1))
x_train = np.stack(x_train)
y_train = np.stack(y_train)
handlerCheckpoint = tf.keras.callbacks.ModelCheckpoint("checkpoint.hdf5")

model.fit(x_train, y_train, epochs=5, batch_size=5, callbacks=[handlerCheckpoint])
model.save("hw0.hdf5")
testImage = plt.imread("../input/u-net-data/test.tif")
plt.subplot(121)
test = next(TrainingsetGenerator(testImage, mask, (512, 512)))[0]
plt.imshow(test, cmap="gray")

plt.subplot(122)
plt.imshow(model.predict(test.reshape(1, 512, 512, 1)).reshape(512, 512), cmap="gray")
# plt.colorbar()
x_test = []
y_test = []
testsetGenerator = TrainingsetGenerator(testImage, mask, (512, 512))
for i in range(1000):
    test = next(testsetGenerator)
    x_test.append(test[0].reshape(512, 512, 1))
    y_test.append(test[1].reshape(512, 512, 1))
x_test = np.stack(x_test)
y_test = np.stack(y_test)

model.evaluate(x_test, y_test)