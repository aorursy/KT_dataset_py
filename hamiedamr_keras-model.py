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
df = pd.read_csv('../input/mushroom-classification/mushrooms.csv')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
val_dataframe = df.sample(frac=0.2, random_state=1337)
train_dataframe = df.drop(val_dataframe.index)
def dataframe_encoding(dataframe):
    le = LabelEncoder()
    for i in range(len(dataframe.columns)):
        le.fit(dataframe.iloc[:,i])
        dataframe.iloc[:,i]  = le.transform(dataframe.iloc[:,i])
    return dataframe
train_dataframe = dataframe_encoding(train_dataframe)
val_dataframe = dataframe_encoding(val_dataframe)
def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("class").to_numpy()
    return dataframe.to_numpy(),labels
x_train,y_train = dataframe_to_dataset(train_dataframe)
x_val,y_val = dataframe_to_dataset(val_dataframe)

inputs = keras.Input(shape = (22,))
x = layers.Dense(128, activation="relu")(inputs)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, output)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
model.summary()
history = model.fit(x_train,y_train,epochs=10,validation_data=(x_val,y_val))
from matplotlib import pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
