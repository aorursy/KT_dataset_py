import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

from keras.datasets import mnist

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose

from keras.models import Model

from keras.preprocessing import image
train_path = "../input/sample-train/train_dataset/"

test_path = "../input/sample-test/test_dataset/"





train = []

for filename in os.listdir(train_path):

    if filename.endswith(".jpg"):

        img = image.load_img(train_path+filename, target_size=(128, 128))

        train.append(image.img_to_array(img))

        

train = np.array(train)

test = []

for filename in os.listdir(test_path):

    if filename.endswith(".jpg"):

        img = image.load_img(test_path+filename, target_size=(128, 128))

        test.append(image.img_to_array(img))

test = np.array(test)
print("train", train.shape)

print("test", test.shape)
def show_data(X, n=10, title=""):

    plt.figure(figsize=(15, 5))

    for i in range(n):

        ax = plt.subplot(2,n,i+1)

        plt.imshow(image.array_to_img(X[i]))

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)

    plt.suptitle(title, fontsize = 20)
show_data(train, title="train Data")

show_data(test, title="test Data")
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
AE = Model(input_layer, output_layer)

AE.compile(optimizer='adam', loss='mse')

AE.summary()
AE.fit(train, train,

                epochs=20,

                batch_size=32,

                shuffle=True,

                validation_data=(test, test))
AE.save("AE.h5")
get_encoded = Model(inputs=AE.input, outputs=AE.get_layer("CODE").output)
encoded_data = get_encoded.predict(test)

encoded_data = encoded_data.reshape((len(test), 16*16*8))

encoded_data.shape
reconstructed_data = AE.predict(test)
def show_data_1(X, n=10, height=28, width=28, title=""):

    plt.figure(figsize=(10, 3))

    for i in range(n):

        ax = plt.subplot(2,n,i+1)

        plt.imshow(X[i].reshape((height,width)))

        plt.gray()

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)

    plt.suptitle(title, fontsize = 20)
show_data(test, title="original images")

show_data_1(encoded_data, height=32, width=64, title="encoded images")

show_data(reconstructed_data, title="reconstructed images")
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=5).fit(encoded_data)

distances, indices = nbrs.kneighbors(np.array([encoded_data[-1]]))



print("distances, indices",distances, indices)

similar_images_list = []

for each_value in indices[0]:

    similar_images_list.append(each_value)

# Given input image for knn model

import matplotlib.pyplot as plt 

import matplotlib.image as img 

  

# reading png image file 

im = img.imread('../input/sample-test/test_dataset/4736.jpg') 

  

# show image 

plt.imshow(im) 
# Output indices images

show_data(test[similar_images_list], title="original images")