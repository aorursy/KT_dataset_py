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
import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt
train_dataset = pd.read_csv('../input/fashion-mnist_train.csv')

test_dataset = pd.read_csv("../input/fashion-mnist_test.csv")

train_dataset.head()
# now I have to split up the label column and pixels column

y_train = train_dataset["label"].values

x_train = train_dataset.drop("label", axis=1)

y_test = test_dataset["label"]

x_test = test_dataset.drop("label", axis=1)
training_data = np.asarray(x_train).reshape(-1,28,28,1)

testing_data = np.asarray(x_test).reshape(-1,28,28,1)
import random

plt.imshow(random.choice(training_data).reshape(28,28),cmap="gray")
# now lets add some deep learning
from tensorflow.keras.layers import Dense, Flatten, Activation,Dropout,Conv2D, MaxPooling2D

from tensorflow.keras.models import Sequential

from keras.utils import np_utils
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', 

                        input_shape=(28, 28, 1)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.3))

model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=2))



model.add(Flatten())

model.add(Dense(30, activation='relu'))

model.add(Dense(20, activation='relu'))

model.add(Dense(10, activation='softmax'))
# compile the model

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(training_data, y_train, batch_size=128, epochs=10,verbose=1)
results = model.predict(testing_data)
labels = {0: "T-shirt/top" ,1: "Trouser", 2: "Pullover",

          3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot" }
fig = plt.figure(figsize=(20, 8))

for i, idx in enumerate(np.random.choice(testing_data.shape[0], size=24, replace=False)): 

    ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])

    ax.imshow(np.squeeze(testing_data[idx]),cmap='gray')

    pred_idx = np.argmax(results[idx])

    ax.set_title("{}".format(labels[pred_idx]))