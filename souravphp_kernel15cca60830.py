# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

import os

from sklearn.metrics import accuracy_score

df = pd.read_csv('../input/zoo-animal-classification/zoo.csv')

df.head()

features = list(df.columns)
print(features)

features.remove('class_type')
features.remove('animal_name')

print(features)

X = df[features].values.astype(np.float32)
Y = df.class_type


print(X.shape)
print(Y.shape)



from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


model = tf.keras.Sequential()
model.add(keras.layers.Dense(units=5,  activation='relu', input_shape=[16,]))
model.add(keras.layers.Dense(units=8,kernel_initializer='he_normal',activation='softmax' ))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=400,verbose=1)
losst, acct = model.evaluate(X_test, Y_test, verbose=0)



print ('Test accuracy=',round(acct*100,2))

row=[1,0,0,1,0,0,1,1,1,1,0,0,4,0,0,1]
yhat=model.predict([row])
print('Predicted: %s (class=%d)' % (yhat, np.argmax(yhat)))





