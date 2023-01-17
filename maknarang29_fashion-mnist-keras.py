# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

from tensorflow import keras
df = pd.read_csv("../input/fashion-mnist_train.csv")

train_labels = df["label"].values

df = df.drop("label", axis = 1)
train_data = df.values

train_data.shape
import matplotlib.pyplot as plt

%matplotlib inline



def show_image(n):

    img = train_data[n]

    plt.imshow(img.reshape(28,28))

    plt.grid(False)

    plt.show()

    

show_image(6)
plt.figure(figsize = (10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.imshow(train_data[i].reshape(28,28))

    plt.xticks([])

plt.show()
train_data = train_data.reshape(60000,28,28)
train_data = train_data/255.00
from keras import layers
model = keras.Sequential([

    keras.layers.Flatten(input_shape = (28,28)),

    keras.layers.Dense(128,activation = 'relu'),

    keras.layers.Dense(10,activation = 'softmax')

])
model.compile(optimizer='adam',metrics=["accuracy"],loss = "sparse_categorical_crossentropy")
model.fit(train_data,train_labels,epochs=10)
train_labels = train_labels.reshape(60000,)

train_labels = np.uint8(train_labels)
df_test = pd.read_csv("../input/fashion-mnist_test.csv")

test_labels = df_test["label"].values

df_test = df_test.drop("label", axis = 1)
test_images = df_test.values
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.imshow(test_images[0].reshape(28,28))
predictions = model.predict(test_images.reshape(-1,28,28))
preds = np.argmax(predictions,axis=1)
plt.figure(figsize=(15,15))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.imshow(test_images[i].reshape(28,28))

    plt.xlabel("pred:{} real: {} class: {:s}".format(preds[i],test_labels[i],classes[test_labels[i]]))

    plt.xticks([])
test_images = test_images.reshape(-1,28,28)

test_loss , test_accuracy = model.evaluate(test_images,test_labels)

print('\nTest Accuracy : ', test_accuracy*100,"%")