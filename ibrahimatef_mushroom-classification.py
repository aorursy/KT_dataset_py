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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

ds= pd.read_csv('../input/mushroom-classification/mushrooms.csv')

ds_validation = ds.sample(frac=0.1 , random_state=123)

ds_train=ds.drop(ds_validation.index)

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from sklearn.preprocessing import LabelEncoder
def dataFrame_encoding(data):

    labelen=LabelEncoder()

    for i in range(len(data.columns)):           

        data.iloc[:,i]=labelen.fit_transform(data.iloc[:,i])

    return data
ds_train = dataFrame_encoding(ds_train)

ds_validation = dataFrame_encoding(ds_validation)
def get_dataset(data):

    data=data.copy()

    labels = data.pop('class').to_numpy()

    return data.to_numpy() , labels

    
x_train,y_train=get_dataset(ds_train)

x_valid,y_valid=get_dataset(ds_validation)
inputs = keras.Input(shape = (22))

x = layers.Dense(128, activation="relu")(inputs)

x = layers.Dropout(0.5)(x)

output = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, output)

model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

history=model.fit(x_train,y_train,epochs=10,validation_data=(x_valid,y_valid))
from matplotlib import pyplot as plt

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()