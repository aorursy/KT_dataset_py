# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from keras import models

from keras import layers

from keras import optimizers

from keras.utils import to_categorical
data = pd.read_csv("/kaggle/input/facial-expression-recognitionferchallenge/fer2013/fer2013/fer2013.csv")

data.head()
print("Datanın satır ve sütün sayıları = ", data.shape)

print("Sütünların ismi = ", data.columns)
data["Usage"].value_counts()
training = data.loc[data["Usage"] == "Training"]

public_test = data.loc[data["Usage"] == "PublicTest"]

private_test = data.loc[data["Usage"] == "PrivateTest"]



print("Eğitim seti = ", training.shape)

print("Genel test seti = ", public_test.shape)

print("Özel test seti = ", private_test.shape)
print("========================= Emotion Adetleri ===========================")

print("train adet = \n{}, \npublic adet = \n{}, \nprivate adet = \n{}".format(training["emotion"].value_counts(),

      public_test["emotion"].value_counts(), private_test["emotion"].value_counts()))

train_labels = training["emotion"]

train_labels = to_categorical(train_labels)



train_pixels = training["pixels"].str.split(" ").tolist()

train_pixels = np.uint8(train_pixels)

train_pixels = train_pixels.reshape((28709, 48, 48, 1))

train_pixels = train_pixels.astype("float32") / 255





private_labels = private_test["emotion"]

private_labels = to_categorical(private_labels)



private_pixels = private_test["pixels"].str.split(" ").tolist()

private_pixels = np.uint8(private_pixels)

private_pixels = private_pixels.reshape((3589, 48, 48, 1))

private_pixels = private_pixels.astype("float32") / 255





public_labels = public_test["emotion"]

public_labels = to_categorical(public_labels)



public_pixels = public_test["pixels"].str.split(" ").tolist()

public_pixels = np.uint8(public_pixels)

public_pixels = public_pixels.reshape((3589, 48, 48, 1))

public_pixels = public_pixels.astype("float32") / 255
import seaborn as sns

plt.figure(0, figsize=(12,6))

for i in range(1, 13):

    plt.subplot(3,4,i)

    plt.imshow(train_pixels[i, :, :, 0], cmap="gray")



plt.tight_layout()

plt.show()

model_1 = models.Sequential()



# Conv (evrişim katmanı)

model_1.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))

#Ortaklama katmanı

model_1.add(layers.MaxPooling2D(pool_size=(5,5), strides=(2, 2)))



model_1.add(layers.Conv2D(64, (3, 3), activation='relu'))

model_1.add(layers.Conv2D(64, (3, 3), activation='relu'))

model_1.add(layers.AveragePooling2D(pool_size=(3,3), strides=(2, 2)))



model_1.add(layers.Conv2D(128, (3, 3), activation='relu'))

model_1.add(layers.Conv2D(128, (3, 3), activation='relu'))

model_1.add(layers.AveragePooling2D(pool_size=(3,3), strides=(2, 2)))



model_1.add(layers.Flatten())



# Tam bağlantı katmanı

model_1.add(layers.Dense(1024, activation='relu'))

model_1.add(layers.Dropout(0.2))

model_1.add(layers.Dense(1024, activation='relu'))

model_1.add(layers.Dropout(0.2))



model_1.add(layers.Dense(7, activation='softmax'))

model_1.summary()
model_1.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])



hist = model_1.fit(train_pixels, train_labels, batch_size = 256, epochs = 30,

                validation_data = (private_pixels, private_labels))

acc = hist.history["accuracy"]

val_acc = hist.history["val_accuracy"]

loss = hist.history["loss"]

val_loss = hist.history["val_loss"]



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, "bo", label = "Eğitim Başarısı")

plt.plot(epochs, val_acc, "b", label = "Doğrulama Başarısı")

plt.title("Eğitim ve Doğrulama Başarısı")

plt.legend()



plt.figure()



plt.plot(epochs, loss, "bo", label = "Eğitim Kaybı")

plt.plot(epochs, val_loss, "b", label = "Doğrulama Kaybı")

plt.legend()





plt.show()
print("Teşekkürler!")