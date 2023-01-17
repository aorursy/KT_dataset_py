# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/cell_images/cell_images/Uninfected"))



# Any results you write to the current directory are saved as output.
import glob

import cv2



uninfected = [cv2.imread(file) for file in glob.glob("../input/cell_images/cell_images/Uninfected/*.png")]

infected = [cv2.imread(file) for file in glob.glob("../input/cell_images/cell_images/Parasitized/*.png")]
import matplotlib.pyplot as plt

plt.imshow(infected[0])
plt.imshow(uninfected[0])
infected[1].shape
resized_image = cv2.resize(infected[0], (50, 50)) 

plt.imshow(resized_image)
resized_infected = []

for img in infected:

    resized_infected.append(cv2.resize(img, (64, 64)))

    

resized_uninfected = []

for img in uninfected:

    resized_uninfected.append(cv2.resize(img, (64, 64)))
x_data = np.array(resized_uninfected + resized_infected)

x_data = x_data/255.



y_un = np.arange(len(resized_uninfected), dtype=int)

y_un = np.full_like(y_un, 0)

y_in = np.arange(len(resized_infected), dtype=int)

y_in = np.full_like(y_in, 1)

y_data = np.append(y_un, y_in)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle=True)
from keras.models import Sequential

from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
model = Sequential()

model.add(Conv2D(64, input_shape=(64,64,3), kernel_size=(5,5), activation="relu"))

model.add(MaxPool2D(2))

model.add(Conv2D(128, 3, activation="relu"))

model.add(Conv2D(256, 3, activation="relu"))

model.add(MaxPool2D(2))

model.add(Conv2D(128, 3, activation="relu"))

model.add(Conv2D(64, 3, activation="relu"))

model.add(MaxPool2D(2))

model.add(Flatten())

model.add(Dense(64, activation="relu"))

model.add(Dropout(rate=0.25))

model.add(Dense(1, activation="sigmoid"))



model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
history = model.fit(X_train, y_train, epochs=50)
plt.plot(history.history["acc"])
model.evaluate(X_test, y_test)