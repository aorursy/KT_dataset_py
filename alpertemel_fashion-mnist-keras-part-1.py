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
from keras import layers

from keras import models

from keras.utils import to_categorical
train_images = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")

train_labels = train_images.iloc[:, 0:1]

del train_images["label"]



test_images = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_test.csv")

test_labels = test_images.iloc[:, 0:1]

del test_images["label"]
train_images.head()
import matplotlib.pyplot as plt



clothing = {0 : 'T-shirt/top',

            1 : 'Trouser',

            2 : 'Pullover',

            3 : 'Dress',

            4 : 'Coat',

            5 : 'Sandal',

            6 : 'Shirt',

            7 : 'Sneaker',

            8 : 'Bag',

            9 : 'Ankle boot'}



fig, axes = plt.subplots(4, 4, figsize = (10,10))

for row in axes:

    for axe in row:

        index = np.random.randint(60000)

        img = train_images.values[index].reshape(28,28)

        cloths = train_labels['label'][index]

        axe.imshow(img, cmap='gray')

        axe.set_title(clothing[cloths])

        axe.set_axis_off()
train_images = np.uint8(train_images)

train_images = train_images.reshape((60000, 28, 28, 1))

train_images = train_images.astype("float32") / 255



test_images = np.uint8(test_images)

test_images = test_images.reshape((10000, 28, 28, 1))

test_images = test_images.astype("float32") / 255



train_labels = to_categorical(train_labels)

test_labels = to_categorical(test_labels)
train_images[3]
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (28, 28, 1)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation = "relu"))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation = "relu"))





model.add(layers.Flatten())

model.add(layers.Dense(64, activation = "relu"))

model.add(layers.Dense(10, activation = "softmax"))

model.summary()
model.compile(optimizer = "rmsprop",

              loss = "categorical_crossentropy",

              metrics = ["accuracy"])



model.fit(train_images, train_labels, epochs = 20, batch_size = 64)
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test Doğruluk oranı: ", round(test_accuracy,2))