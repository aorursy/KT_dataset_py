# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.model_selection import train_test_split

from tensorflow import keras



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.deep_learning.exercise_8 import *

print("Setup Complete")



img_rows, img_cols = 28, 28

num_classes = 10



def prep_data(raw):

    y = raw[:, 0]

    out_y = keras.utils.to_categorical(y, num_classes)

    

    x = raw[:,1:]

    num_images = raw.shape[0]

    out_x = x.reshape(num_images, img_rows, img_cols, 1)

    out_x = out_x / 255

    return out_x, out_y



fashion_file = "/kaggle/input/digit-recognizer/train.csv"

fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')

x, y = prep_data(fashion_data)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout



batch_size = 16



fashion_model = Sequential()

fashion_model.add(Conv2D(16, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(img_rows, img_cols, 1)))

fashion_model.add(Conv2D(16, (3, 3), activation='relu',strides=2))

fashion_model.add(Flatten())

fashion_model.add(Dense(128, activation='relu'))

fashion_model.add(Dense(num_classes, activation='softmax'))



fashion_model.compile(loss="binary_crossentropy",

              optimizer='adam',

              metrics=['accuracy'])



fashion_model.fit(x, y,

          batch_size=batch_size,

          epochs=8,

          validation_split = 0.2)
