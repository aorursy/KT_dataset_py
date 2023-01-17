# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

"""

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

"""

# Any results you write to the current directory are saved as output.
import keras

from keras import layers

from keras import models

from keras.preprocessing import image

import os

import numpy as np

from sklearn.model_selection import train_test_split
data =[]

labels = []

data_dir = '../input/cell-images-for-detecting-malaria/cell_images/cell_images'

for img in os.listdir(data_dir+'/Parasitized'):

    if img.endswith(".png"):

        img_path = data_dir+'/Parasitized/'+img

        #print(img_path)

        image_data = image.load_img(img_path, target_size=(100, 100))

        image_data = image.img_to_array(image_data)

        data.append(image_data)

        labels.append(1)

    

for img in os.listdir(data_dir+'/Uninfected'):

    img_path = data_dir+'/Uninfected/'+img

    if img.endswith(".png"):

        #print(img_path)

        image_data = image.load_img(img_path, target_size=(100, 100))

        image_data = image.img_to_array(image_data)

        data.append(image_data)

        labels.append(0)

    
DATA = np.array(data)/255.

LABELS = np.array(labels).reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(DATA, LABELS, test_size = 0.3, shuffle =True)
model = models.Sequential()

model.add(layers.Conv2D(16, 3,input_shape = (X_train.shape[1], X_train.shape[2], 3), padding='SAME', activation='relu'))

model.add(layers.MaxPooling2D(2, padding='SAME'))

model.add(layers.Conv2D(32, 3, activation='relu'))

model.add(layers.MaxPooling2D(2))

model.add(layers.Dropout(0.25))



model.add(layers.Conv2D(64, 3, activation='relu'))

model.add(layers.MaxPooling2D(2))

model.add(layers.Conv2D(128, 3, activation='relu'))

model.add(layers.MaxPooling2D(2))

model.add(layers.Dropout(0.25))



model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dropout(0.25))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='Adam')

history = model.fit(X_train, y_train, batch_size= 128, epochs=20, validation_split = 0.2, shuffle=True )







model.evaluate(X_test, y_test)