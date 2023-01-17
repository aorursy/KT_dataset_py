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
# load the data

import numpy as np

xtr = np.load('/kaggle/input/comp-850-project-1/xtr.npy')

xte = np.load('/kaggle/input/comp-850-project-1/xte.npy')

ytr = np.load('/kaggle/input/comp-850-project-1/ytr.npy')
import matplotlib.pyplot as plt

%matplotlib inline

fig, ax = plt.subplots(2, 4, figsize=(16, 6))

for i in range(2):

    for j in range(4):

        ax[i, j].imshow(xtr[i * 4 + j, :, :], cmap='gray')
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

from tensorflow.keras.optimizers import Adam
# build the model

model = Sequential()

model.add(Conv2D(16, 3, input_shape=(28, 28, 1), activation='relu', padding='same'))

model.add(Conv2D(16, 3, activation='relu', padding='same'))

model.add(MaxPool2D(2, 2))

model.add(Conv2D(32, 3, activation='relu', padding='same'))

model.add(Conv2D(32, 3, activation='relu', padding='same'))

model.add(MaxPool2D(2, 2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(10, activation='softmax'))



optimizer = Adam(learning_rate=3e-3)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
# preprocess the data

from tensorflow.keras.utils import to_categorical



xtr = np.expand_dims(xtr, -1).astype(np.float32) / 255

xte = np.expand_dims(xte, -1).astype(np.float32) / 255

ytr = to_categorical(ytr)
history = model.fit(xtr, ytr, validation_split=0.3, batch_size=128, epochs=5)
# make predictions

pred = model.predict(xte)

categorical_pred = np.argmax(pred, axis=1)
# write the predictions to a csv file

import pandas as pd

submission = pd.read_csv('/kaggle/input/comp-850-project-1/sample_submission.csv')

submission['Category'] = categorical_pred



submission.to_csv('my_first_submission.csv', index=False)