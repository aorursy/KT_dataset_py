import warnings, os

warnings.filterwarnings('ignore')
import numpy as np

import pandas as pd

from random import randint

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Convolution2D

from tensorflow.keras.layers import MaxPooling2D, BatchNormalization

from tensorflow.keras.layers import Activation, GlobalAveragePooling2D

from tensorflow.keras.optimizers import SGD, RMSprop

from keras.utils.vis_utils import model_to_dot

from IPython.display import SVG

from sklearn.model_selection import train_test_split
faces = np.moveaxis(np.load('../input/face_images.npz')['face_images'], -1, 0)

faces.shape
landmarks = pd.read_csv('../input/facial_keypoints.csv')

landmarks.shape
landmarks.head()
prediction_fields = list(landmarks)[:4]

print('Columns for eye detection:\n', prediction_fields)
cols_not_null = landmarks['left_eye_center_x'].notna() 

for i in prediction_fields[1:]:

    cols_not_null = cols_not_null & landmarks[i].notna()

is_select = np.nonzero(cols_not_null)[0]

is_select.shape
Y = np.zeros((is_select.shape[0], len(prediction_fields)))

for i in range(len(prediction_fields)):

    Y[:, i] = landmarks[prediction_fields[i]][is_select] / faces.shape[1]

Y.shape
X = np.zeros((is_select.shape[0], faces.shape[1], faces.shape[1], 1))

X[:, :, :, 0] = faces[is_select, :, :] / 255.0

X.shape
def get_coordinates(arr, scale = 96):

    x, y = [], []

    for i in range(len(arr)):

        if i % 2 == 0:

            x.append(arr[i] * scale)

        else:

            y.append(arr[i]* scale)

    return x, y
fig, axes = plt.subplots(nrows = 4, ncols = 4, figsize = (16, 16))

plt.setp(axes.flat, xticks = [], yticks = [])

for i, ax in enumerate(axes.flat):

    index = randint(0, 2167)

    img = X[index].reshape(96, 96)

    landmark_x, landmark_y = get_coordinates(Y[index])

    ax.imshow(img, cmap = 'gray')

    ax.scatter(landmark_x, landmark_y, c = 'r')

    ax.set_xlabel('Face_' + str(index))

plt.show()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
model_1 = Sequential([

    Conv2D(16, (3, 3), padding = 'same', activation = 'tanh', input_shape = (96, 96, 1)),

    MaxPooling2D(2, 2),

    Dropout(rate = 0.75),

    Conv2D(32, (3, 3), padding = 'same', activation = 'tanh', input_shape = (96, 96, 1)),

    MaxPooling2D(2, 2),

    Dropout(rate = 0.75),

    Flatten(),

    Dense(256, activation = 'tanh'),

    Dropout(rate = 0.75),

    Dense(len(prediction_fields), activation = 'sigmoid')

])



model_1.summary()
SVG(model_to_dot(model_1, show_shapes = True, show_layer_names = True).create(prog = 'dot', format = 'svg'))
model_1.compile(

    loss = 'mean_squared_error',

    optimizer = SGD(

        lr = 0.1,

        decay = 1e-6,

        momentum = 0.9,

        nesterov = True

    ),

    metrics = ['accuracy']

)
history = model_1.fit(X_train, Y_train, batch_size = 128, epochs = 8, validation_data = (X_test, Y_test), verbose = 1)
model_1.evaluate(X_test, Y_test, verbose = 1)
fig, axes = plt.subplots(nrows = 4, ncols = 4, figsize = (16, 16))

plt.setp(axes.flat, xticks = [], yticks = [])

for i, ax in enumerate(axes.flat):

    index = randint(0, 6329)

    img = X_train[index].reshape(96, 96)

    landmark_x_original, landmark_y_original = get_coordinates(Y_train[index])

    landmark_x, landmark_y = get_coordinates(model_1.predict(X_train[index].reshape(1, 96, 96, 1))[0])

    ax.imshow(img, cmap = 'gray')

    ax.scatter(landmark_x_original, landmark_y_original, c = 'r')

    ax.scatter(landmark_x, landmark_y, c = 'b')

    ax.set_xlabel('Results on Training by Model 1: Face_' + str(index))

plt.show()
fig, axes = plt.subplots(nrows = 4, ncols = 4, figsize = (16, 16))

plt.setp(axes.flat, xticks = [], yticks = [])

for i, ax in enumerate(axes.flat):

    index = randint(0, 704)

    img = X_test[index].reshape(96, 96)

    landmark_x_original, landmark_y_original = get_coordinates(Y_train[index])

    landmark_x, landmark_y = get_coordinates(model_1.predict(X_test[index].reshape(1, 96, 96, 1))[0])

    ax.imshow(img, cmap = 'gray')

    ax.scatter(landmark_x_original, landmark_y_original, c = 'r')

    ax.scatter(landmark_x, landmark_y, c = 'b')

    ax.set_xlabel('Results on Test Set by Model 1: Face_' + str(index))

plt.show()
model_1.save('model_1.h5')
model_2 = Sequential([

    BatchNormalization(input_shape = (96, 96, 1)),

    

    Conv2D(24, (5, 5), padding = 'same', activation = 'relu', input_shape = (96, 96, 1)),

    MaxPooling2D(2, 2, padding = 'valid'),

    Dropout(rate = 0.75),

    

    Conv2D(36, (5, 5), activation = 'relu'),

    MaxPooling2D(2, 2, padding = 'valid'),

    Dropout(rate = 0.75),

    

    Conv2D(48, (5, 5), activation = 'relu'),

    MaxPooling2D(2, 2, padding = 'valid'),

    Dropout(rate = 0.75),

    

    Conv2D(64, (3, 3), activation = 'relu'),

    MaxPooling2D(2, 2, padding = 'valid'),

    Dropout(rate = 0.75),

    

    Conv2D(64, (3, 3), activation = 'relu'),

    GlobalAveragePooling2D(),

    Dropout(rate = 0.75),

    

    Dense(500, activation = 'relu'),

    Dropout(rate = 0.75),

    Dense(90, activation = 'relu'),

    Dropout(rate = 0.75),

    Dense(4),

])
model_2.summary()
SVG(model_to_dot(model_2, show_shapes = True, show_layer_names = True).create(prog = 'dot', format = 'svg'))
model_2.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['accuracy'])
history_2 = model_2.fit(

    X_train, Y_train,

    validation_data = (

        X_test,

        Y_test

    ),

    batch_size = 20,

    epochs = 5,

    shuffle = True,

    verbose = 1

)
model_2.evaluate(X_test, Y_test, verbose = 1)
fig, axes = plt.subplots(nrows = 4, ncols = 4, figsize = (16, 16))

plt.setp(axes.flat, xticks = [], yticks = [])

for i, ax in enumerate(axes.flat):

    index = randint(0, 6329)

    img = X_train[index].reshape(96, 96)

    landmark_x_original, landmark_y_original = get_coordinates(Y_train[index])

    landmark_x, landmark_y = get_coordinates(model_2.predict(X_train[index].reshape(1, 96, 96, 1))[0])

    ax.imshow(img, cmap = 'gray')

    ax.scatter(landmark_x_original, landmark_y_original, c = 'r')

    ax.scatter(landmark_x, landmark_y, c = 'b')

    ax.set_xlabel('Results on Training by Model 2: Face_' + str(index))

plt.show()
fig, axes = plt.subplots(nrows = 4, ncols = 4, figsize = (16, 16))

plt.setp(axes.flat, xticks = [], yticks = [])

for i, ax in enumerate(axes.flat):

    index = randint(0, 704)

    img = X_test[index].reshape(96, 96)

    landmark_x_original, landmark_y_original = get_coordinates(Y_train[index])

    landmark_x, landmark_y = get_coordinates(model_2.predict(X_test[index].reshape(1, 96, 96, 1))[0])

    ax.imshow(img, cmap = 'gray')

    ax.scatter(landmark_x_original, landmark_y_original, c = 'r')

    ax.scatter(landmark_x, landmark_y, c = 'b')

    ax.set_xlabel('Results on Test Set by Model 2: Face_' + str(index))

plt.show()
model_2.save('model_2.h5')