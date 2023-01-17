# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle

from matplotlib import pyplot as plt

import csv

import keras 

from keras.models import Sequential

from keras.layers import Conv2D, Activation, Flatten, Dense, MaxPooling2D, Dropout

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
with open('/kaggle/input/simple-cnn/train.pkl', 'rb') as fhand:

    x_train = pickle.load(fhand)

y_train = pd.read_csv('/kaggle/input/simple-cnn/train.csv').to_numpy()
indices = np.random.choice(len(x_train), 10)

x_samples = x_train[indices]

y_samples = y_train[indices]

for idx, (im, label) in enumerate(zip(x_samples, y_samples)):

    plt.imshow(im)

    plt.title(label[1])

    plt.show()
classes = sorted(np.unique(y_train[:,1]))

le = LabelEncoder()

le.fit(classes)

print(le.classes_)

labels = le.transform(y_train[:,1])

y_train = keras.utils.to_categorical(labels, 10)
input_shape = (32, 32, 3)

model = Sequential([

    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),

    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),

    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),

    Dropout(0.5),

    Flatten(),

    Dense(128, activation='relu'),

    Dropout(0.25),

    Dense(10, activation='softmax')

])
model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])
train_x, test_x, train_y, test_y = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

model.fit(train_x, train_y,

          batch_size=16,

          epochs=3,

          verbose=1,

          validation_data=(test_x, test_y))
with open("/kaggle/input/simple-cnn/train.pkl","rb") as fhand:

    x_test = pickle.load(fhand)

y_pred = model.predict(x_test)
pred_label = np.argmax(y_pred, axis=-1)

pred_classes = le.inverse_transform(pred_label)
with open('submission.csv', 'w', newline='') as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow(["Id", "Category"])

    for idx, label in enumerate(pred_classes):

        writer.writerow([idx, label])