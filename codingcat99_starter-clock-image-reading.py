from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print(os.listdir('../input'))
x = np.load("../input/clock_image.npy")

y = np.load("../input/clock_time.npy")
# print a few images

for i in range(5):

    print ("label", "{}:{}".format(y[i][0], y[i][1]))

    img = x[i]

    plt.imshow(img, origin="lower")

    plt.show()

# split training and validation set

def preprocess(x, y):

    x = x.reshape(x.shape[0], 64, 64, 1)

    share = 400

    x_train = x[:share]

    y_train = y[:share]

    x_val = x[share:]

    y_val = y[share:]

    return x_train, y_train, x_val, y_val

x_train, y_train, x_val, y_val = preprocess(x, y)
#build cnn model

import tensorflow as tf

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Dense, Conv2D,MaxPool2D,Flatten

from tensorflow.keras.optimizers import Adam



def build_model(x):

    opt = Adam(lr=0.001)

    

    model = Sequential()

    model.add(Conv2D(32, 5, padding = "valid", input_shape=x.shape[1:],activation="relu"))

    model.add(MaxPool2D(pool_size=(2,2), strides=2))

    model.add(Conv2D(64, 5, padding = "valid", activation="relu"))

    model.add(MaxPool2D(pool_size=(2,2), strides=2))

    model.add(Flatten())

    model.add(Dense(256,activation="relu"))

    model.add(Dense(2))

              

    model.compile(loss='mse',

              optimizer=opt,

              metrics=['mse'])

    return model



model = build_model(x_train)
print(model.summary())

def train(model, x, y, xval, yval):

    model.fit(x, y, batch_size=16, epochs=60, validation_data=(xval,yval))

train(model, x_train, y_train, x_val, y_val)
# How many cases are perfectly correct

def get_accuracy(x, ylabel):

    r = np.round(model.predict(x))

    diff = r - ylabel

    a = np.min(np.abs(diff), axis=1)

    return np.count_nonzero(a==0)/a.shape[0]



print("train accuracy: ", get_accuracy(x_train, y_train))

print("validation accuracy: ", get_accuracy(x_val, y_val))
#build activation model

from tensorflow.keras.models import  Model

layer_outputs = [layer.output for layer in model.layers] # Extracts the outputs of the top 12 layers

activation_model = Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(x_val[4:5])



layer_names = []

for layer in model.layers[:4]:

    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot

    

images_per_row = 8

for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps

    n_features = layer_activation.shape[-1] # Number of features in the feature map

    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).

    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix

    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols): # Tiles each filter into a big horizontal grid

        for row in range(images_per_row):

            channel_image = layer_activation[0,

                                             :, :,

                                             col * images_per_row + row]

            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable

            channel_image /= channel_image.std()

            channel_image *= 64

            channel_image += 128

            channel_image = np.clip(channel_image, 0, 255).astype('uint8')

            display_grid[col * size : (col + 1) * size, # Displays the grid

                         row * size : (row + 1) * size] = channel_image

    scale = 1. / size

    plt.figure(figsize=(scale * display_grid.shape[1],

                        scale * display_grid.shape[0]))

    plt.title(layer_name)

    plt.grid(False)

    plt.imshow(display_grid, aspect='auto', cmap='viridis')

import tensorflow as tf

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Dense, Conv2D,MaxPool2D,Flatten,UpSampling2D

from tensorflow.keras.optimizers import Adam



def build_auto_encoder(x):

    opt = Adam(lr=0.001)

    model = Sequential()

    

    model.add(Conv2D(32, 3, padding = "same", input_shape=x.shape[1:],activation="relu"))

    model.add(MaxPool2D(pool_size=(2,2), strides=2))

    model.add(Conv2D(32, 3, padding = "same", activation="relu"))

    model.add(MaxPool2D(pool_size=(2,2), strides=2))

    

    model.add(Conv2D(32, 3, padding = "same", input_shape=x.shape[1:],activation="relu"))

    model.add(UpSampling2D(size=(2,2)))

    model.add(Conv2D(32, 3, padding = "same", activation="relu"))

    model.add(UpSampling2D(size=(2,2)))

    model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

    

    model.compile(loss='binary_crossentropy',

              optimizer=opt)

    return model

auto_encoder = build_auto_encoder(x_train)



#train a regressor from encoded layer











auto_encoder.fit(x_train, x_train, batch_size=16, epochs=60, validation_data=(x_val,x_val))
import random

#visulaize difference

plt.figure()

n = 3

c = np.random.choice(x_val.shape[0], n)

xs = x_val[c]

xs_pred = auto_encoder.predict(xs)

for i in range(n):

    plt.subplot(n, 2, i*2 + 1)

    plt.imshow(xs[i].reshape([64,64]))

    plt.subplot(n, 2, i*2 + 2)

    plt.imshow(xs_pred[i].reshape([64,64]))

plt.show()
#build a regressor on top of the reduced representation

for layer in auto_encoder.layers:

    layer.trainable = False

    

encoded = auto_encoder.layers[3].output

f = Flatten()(encoded)

d = Dense(128, activation="relu")(f)

dout = Dense(2)(d)

model2 = Model(inputs = auto_encoder.inputs, outputs=dout)



model2.compile(loss='mse',

              optimizer=Adam(lr=0.001),

              metrics=["mse"])



#print (model2.summary())

model2.fit(x_train, y_train, batch_size=100, epochs=60, validation_data=(x_val,y_val))