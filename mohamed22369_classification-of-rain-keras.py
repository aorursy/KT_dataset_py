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
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical, normalize

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

data=pd.read_csv('/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv')

data.head()

data.info()
data = data.dropna()

data.isnull().sum()
data['Date']=pd.to_datetime(data['Date'])

data.set_index('Date',inplace=True)

data.sort_index(inplace=True)

data.head()
data['Location'].unique().sum()
from sklearn.preprocessing import LabelEncoder

Lbnames=['Location','WindGusDir','WinDir9am','WinDir3pm','RainToday']



data['Location'] = data['Location'].astype('category').cat.codes

data['WindGustDir'] = data['WindGustDir'].astype('category').cat.codes

data['WindDir9am'] = data['WindDir9am'].astype('category').cat.codes

data['WindDir3pm'] = data['WindDir3pm'].astype('category').cat.codes

data['RainToday'] = data['RainToday'].astype('category').cat.codes

data['RainTomorrow'] = data['RainTomorrow'].astype('category').cat.codes
data.head()
data.info()
data.reset_index(drop=True, inplace=True)

X = data.drop('RainTomorrow', axis=1)

y = data['RainTomorrow']

X = X.values

X = normalize(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)

X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size = 0.5)

model = tf.keras.models.Sequential([

       

    tf.keras.layers.Dense(128, input_shape=(22,), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    

    tf.keras.layers.Dense(125, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),



            

    tf.keras.layers.Dense(1, activation='sigmoid')

])





model.compile(loss='binary_crossentropy',

              optimizer=Adam(0.00001),

              metrics=['acc'])
history = model.fit(X_train, y_train,

                    epochs=150,

                    validation_data=(X_val, y_val),

                    verbose=1,

                   )
loss, accuracy = model.evaluate(X_test, y_test)
