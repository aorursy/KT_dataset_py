import numpy as np
import pandas as pd
import rasterio as rs
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, UpSampling2D


demo = np.array(rs.open("../input/water-scarcity/wsi_1981-1990_01.tif").read(1))
plt.figure(figsize=(20, 40))
plt.imshow(demo, cmap="Greys", vmin=0, vmax=1)
plt.show()
zeros = np.zeros((280, 720, 12))
zeros2 = np.zeros((280, 720, 12))

string = "../input/water-scarcity/wsi_1981-1990_0"
for z in range(9):
  test = string + str(z + 1) + ".tif"
  temp = rs.open(test)
  band1 = temp.read(1)
  arr = np.array(band1)
  zeros[:, :, z] = arr

zeros[:, :, 9] = np.array(rs.open("../input/water-scarcity/wsi_1981-1990_10.tif").read(1))
zeros[:, :, 10] = np.array(rs.open("../input/water-scarcity/wsi_1981-1990_11.tif").read(1))
zeros[:, :, 11] = np.array(rs.open("../input/water-scarcity/wsi_1981-1990_12.tif").read(1))

zeros = np.where(zeros == np.min(zeros), 0, zeros)

string = "../input/water-scarcity/wsi_2001-2010_0"
for z in range(9):
  test = string + str(z + 1) + ".tif"
  temp = rs.open(test)
  band1 = temp.read(1)
  arr = np.array(band1)
  zeros2[:, :, z] = arr

zeros2[:, :, 9] = np.array(rs.open("../input/water-scarcity/wsi_2001-2010_10.tif").read(1))
zeros2[:, :, 10] = np.array(rs.open("../input/water-scarcity/wsi_2001-2010_11.tif").read(1))
zeros2[:, :, 11] = np.array(rs.open("../input/water-scarcity/wsi_2001-2010_12.tif").read(1))

zeros2 = np.where(zeros2 == np.min(zeros2), 0, zeros2)

zeros_1 = np.zeros((12, 56, 144))
zeros_2 = np.zeros((12, 56, 144))
# for x in range(56):
#   for y in range(144):
#     for z in range(12):
#       zeros_1[z, x, y] = np.mean(zeros[5 * x:5 * (x + 1), 5 * y: 5 * (y + 1), z])
#       zeros_2[z, x, y] = np.mean(zeros2[5 * x:5 * (x + 1), 5 * y: 5 * (y + 1), z])


NewZeros = np.zeros((18, 280, 720, 3))
Y = np.zeros((18, 280, 720))

projection = np.zeros((1, 280, 720, 3))
for i in range(3):
    projection[0, :, :, i] = zeros2[:, :, 9 + i]

for z in range(9):
  Y[z, :, :] = zeros[:, :, z + 3]
  # Y[6, :, :] = zeros_1[9:, :, :]

for z in range(9):
  Y[z+9, :, :] = zeros2[:, :, z+3]

for z in range(9):
  for y in range(3):
    NewZeros[z, :, :, y] = zeros[:, :, z + y]

for z in range(9):
  for y in range(3):
    NewZeros[z+9, :, :, y] = zeros2[:, :, z + y]



# new_Y = np.round(Y)

# # FlattenedY = np.zeros((18, 8064))
# # for z in range(18):
# #   FlattenedY[z] = new_Y[z].reshape(8064)
model = tf.keras.Sequential()
model.add(Convolution2D(32, 3, input_shape = NewZeros.shape[1:], padding="same", kernel_initializer="he_normal"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Convolution2D(32, 3, padding="same", kernel_initializer="he_normal"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(UpSampling2D(size=(2, 2)))
model.add(Convolution2D(32, 3, padding="same", kernel_initializer="he_normal"))
model.add(Activation('relu'))
model.add(UpSampling2D(size=(2, 2)))
model.add(Convolution2D(1, 3, padding="same", kernel_initializer="he_normal"))
model.add(Activation('relu'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(NewZeros, Y, batch_size=None, epochs=100, verbose=1, steps_per_epoch=32)
image = model.predict(projection).reshape((280, 720))
print(np.unique(image))
plt.figure(figsize = (20, 40))
plt.imshow(image, cmap="Greys")