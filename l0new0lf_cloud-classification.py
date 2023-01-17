from PIL import Image

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import cv2

import keras

# For one-hot-encoding

from keras.utils import np_utils

# For creating sequenttial model

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

# For saving and loading models

from keras.models import load_model
'''

labels = pd.read_csv("/kaggle/input/understanding_cloud_organization/train.csv")

labels.head()

'''


'''

labels["label"] = labels["Image_Label"].map(lambda s: s.split("_")[1])

labels["image"] = labels["Image_Label"].map(lambda s: s.split("_")[0])

'''
'''

labels.head()

'''
'''

labels['label'].value_counts()

'''
'''

# numpy lists with image names

gravels = []

fishes = []

sugars = []

flowers = []

nans = []

for _ in labels["Image_Label"]:

    if _.split("_")[1] == "Gravel":

        gravels.append(_.split("_")[0])

    elif _.split("_")[1] == "Fish":

        fishes.append(_.split("_")[0])

    elif _.split("_")[1] == "Sugar":

        sugars.append(_.split("_")[0])

    elif _.split("_")[1] == "Flower":

        flowers.append(_.split("_")[0])

    else:

        nans.append(_.split("_")[0])

'''

'''

gravels[:5]

'''
'''

train_images_location = "/kaggle/input/understanding_cloud_organization/train_images/"

test_images_location = "/kaggle/input/understanding_cloud_organization/test_images/"

data = []

labels = []

'''
'''

N = 0

for cloud_type in [gravels, fishes, flowers, sugars]:

    for filename in cloud_type:

        try:

            image = cv2.imread(train_images_location + filename)

            image_from_numpy_array = Image.fromarray(image, "RGB")

            resized_image = image_from_numpy_array.resize((50,50))

            data.append(np.array(resized_image))

            

            if N == 0:

                labels.append(0)

            elif N == 1:

                labels.append(1)

            elif N == 2:

                labels.append(2)

            elif N == 3:

                labels.append(3)

            else:

                pass

            

        except:

            print("error occured for " + filename +". It isn't an image" )

    N=N+1

'''

'''

clouds = np.array(data)

labels = np.array(labels)

'''
'''

print(clouds.shape)

print(labels.shape)

'''
'''

np.save("all-clouds-as-rgb-image-arrays", clouds)

np.save("corresponding-labels-for-all-clouds-unshuffled", labels)

'''
!wget https://www.kaggleusercontent.com/kf/24253071/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..UBE9ENvOtVd5H0IKuCGF3g.1WLRXJgnZUzIQv4Xwts-9tSDQtAe_lzMv4eIq7_5G4Lm0EoBxRJa-txxI3nqPzQEM1YJPrDb4XDE_Pd3jB48ACcCeogiytpPHOwIj5y9O02Fnj2ZWDwmmaInJZ7JUeyT-2Tcy-hwGTxSe0JQ8uS4in8VLdPLU37KhG8J9msYcMw.JEO6ltHs0mYYV8K9W_Q4DA/corresponding-labels-for-all-clouds-unshuffled.npy

!wget https://www.kaggleusercontent.com/kf/24253071/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..sLTBUb9nfTuwKXtGM-ySlg.RmDjJtstKEkKzfjI63eov1VMyZfBuqzRLnqmv7G99wgxaFJJW1MtlSiNYq98Vp9KyK9RpSkXZYw-4xfxVADeUOWZr4dvhmqV0vq6OTZVa2cz0dhXkqwcbDz2KlSbhrAoIGazHxGEkW-oND_ygZKigvPXlaMctn7c1zE4RZUKczg.Pl8brjSq2VyWqt-gHqBp0g/all-clouds-as-rgb-image-arrays.npy
clouds = np.load("all-clouds-as-rgb-image-arrays.npy")

labels = np.load("corresponding-labels-for-all-clouds-unshuffled.npy")
np.save("all-clouds-as-rgb-image-arrays", clouds)

np.save("corresponding-labels-for-all-clouds-unshuffled", labels)
shuffle = np.arange(clouds.shape[0])

np.random.shuffle(shuffle)

clouds = clouds[shuffle]

labels = labels[shuffle]
num_classes = len(np.unique(labels)) 

len_data = len(clouds) 
(x_train,x_test)=clouds[(int)(0.1*len_data):],clouds[:(int)(0.1*len_data)]

(y_train,y_test)=labels[(int)(0.1*len_data):],labels[:(int)(0.1*len_data)]
# Normalizing data

x_train = x_train.astype("float32") / 255.0

x_test = x_test.astype("float32") / 255.0
# one hot encoding for keras

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)
x_train.shape
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=2, padding="same", activation="relu", input_shape=(50,50,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(500, activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(1000, activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(1000, activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(4, activation="softmax"))

model.summary()
model.compile(loss="categorical_crossentropy",

               optimizer="adam",

               metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=50, epochs=20, verbose=1)
accuracy =model.evaluate(x_test, y_test, verbose=1)

print(accuracy[1])
# save model weights

model.save("keras-malaria-detection-cnn.h5")