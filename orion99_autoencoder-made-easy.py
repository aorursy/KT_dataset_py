import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

from keras.datasets import mnist

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose

from keras.models import Model

from keras.preprocessing import image
(X_train, _), (X_test, _) = mnist.load_data()
print("X_train", X_train.shape)

print("X_test", X_test.shape)
def pre_process(X):

    X = X/255.0

    X = X.reshape((len(X), 784))

    return X



X_train  =  pre_process(X_train)

X_test  =  pre_process(X_test)



print("X_train", X_train.shape)

print("X_test", X_test.shape)
def show_data(X, n=10, height=28, width=28, title=""):

    plt.figure(figsize=(10, 3))

    for i in range(n):

        ax = plt.subplot(2,n,i+1)

        plt.imshow(X[i].reshape((height,width)))

        plt.gray()

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)

    plt.suptitle(title, fontsize = 20)
show_data(X_train, title="train data")

show_data(X_test, title="test data")
input_dim, output_dim = 784, 784

encode_dim = 100

hidden_dim = 256
# encoder

input_layer = Input(shape=(input_dim,), name="INPUT")

hidden_layer_1 = Dense(hidden_dim, activation='relu', name="HIDDEN_1")(input_layer)



# code

code_layer = Dense(encode_dim, activation='relu', name="CODE")(hidden_layer_1)



# decoder

hidden_layer_2 = Dense(hidden_dim, activation='relu', name="HIDDEN_2")(code_layer)

output_layer = Dense(output_dim, activation='sigmoid', name="OUTPUT")(hidden_layer_2)
AE = Model(input_layer, output_layer)

AE.compile(optimizer='adam', loss='binary_crossentropy')

AE.summary()
AE.fit(X_train, X_train, epochs=10)
decoded_data = AE.predict(X_test)
get_encoded_data = Model(inputs=AE.input, outputs=AE.get_layer("CODE").output)
encoded_data = get_encoded_data.predict(X_test)
show_data(X_test, title="original data")

show_data(encoded_data, height=10, width=10, title="encoded data")

show_data(decoded_data, title="decoded data")
cat_train_path = "../input/cat-and-dog/training_set/training_set/cats/"

cat_test_path = "../input/cat-and-dog/test_set/test_set/cats/"



cat_train = []

for filename in os.listdir(cat_train_path):

    if filename.endswith(".jpg"):

        img = image.load_img(cat_train_path+filename, target_size=(128, 128))

        cat_train.append(image.img_to_array(img))

cat_train = np.array(cat_train)



cat_test = []

for filename in os.listdir(cat_test_path):

    if filename.endswith(".jpg"):

        img = image.load_img(cat_test_path+filename, target_size=(128, 128))

        cat_test.append(image.img_to_array(img))

cat_test = np.array(cat_test)
print("cat_train", cat_train.shape)

print("cat_test", cat_test.shape)
def show_cat_data(X, n=10, title=""):

    plt.figure(figsize=(15, 5))

    for i in range(n):

        ax = plt.subplot(2,n,i+1)

        plt.imshow(image.array_to_img(X[i]))

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)

    plt.suptitle(title, fontsize = 20)
show_cat_data(cat_train, title="train cats")

show_cat_data(cat_test, title="test cats")
input_layer = Input(shape=(128, 128, 3), name="INPUT")

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)

x = MaxPooling2D((2, 2))(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

x = MaxPooling2D((2, 2))(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)



code_layer = MaxPooling2D((2, 2), name="CODE")(x)



x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(code_layer)

x = UpSampling2D((2, 2))(x)

x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)

x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2,2))(x)

output_layer = Conv2D(3, (3, 3), padding='same', name="OUTPUT")(x)
cat_AE = Model(input_layer, output_layer)

cat_AE.compile(optimizer='adam', loss='mse')

cat_AE.summary()
cat_AE.fit(cat_train, cat_train,

                epochs=30,

                batch_size=32,

                shuffle=True,

                validation_data=(cat_test, cat_test))
cat_AE.save("cat_AE.h5")
get_encoded_cat = Model(inputs=cat_AE.input, outputs=cat_AE.get_layer("CODE").output)
encoded_cat = get_encoded_cat.predict(cat_test)

encoded_cat = encoded_cat.reshape((len(cat_test), 16*16*8))

encoded_cat.shape
reconstructed_cats = cat_AE.predict(cat_test)
show_cat_data(cat_test, title="original cats")

show_data(encoded_cat, height=32, width=64, title="encoded cats")

show_cat_data(reconstructed_cats, title="reconstructed cats")