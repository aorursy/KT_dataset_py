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
mushroom_data = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
mushroom_data.head()
null_sum = mushroom_data.isnull().sum()
null_sum
for name in mushroom_data.columns:

    print(name, mushroom_data[mushroom_data[name] == '?'].shape)
mushroom_data.drop(['stalk-root'], axis=1, inplace=True)
mushroom_data.shape
y = mushroom_data['class']

X = mushroom_data.drop(['class'], axis=1)
y.shape
X.shape
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder(sparse=False)

X_1hot = cat_encoder.fit_transform(X)
X_1hot[1:5,]
from sklearn.preprocessing import LabelEncoder

lbl_encoder = LabelEncoder()

y_1hot = lbl_encoder.fit_transform(y)
y_1hot
import tensorflow as tf

from tensorflow import keras
from sklearn.model_selection import train_test_split
X_train_full, X_test, y_train_full, y_test = train_test_split(X_1hot, y_1hot, test_size=0.33, random_state=1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
from keras.models import Sequential

from keras.layers import Dense
model = keras.models.Sequential()

model.add(Dense(30, activation="relu", input_shape=X_train.shape[1:]))

model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=20, batch_size=25, validation_data=(X_valid, y_valid))
model.evaluate(X_test, y_test)