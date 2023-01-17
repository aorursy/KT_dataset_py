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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
dataset = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")
dataset.head()
dataset.shape
dataset.info()
dataset.isnull().sum()
dataset.corr()['price'].sort_values()
dataset.describe().transpose()
plt.figure(figsize=(12,8))
sns.distplot(dataset['price'])
dataset

dataset.drop(columns =["id"], inplace = True) 
dataset.drop(columns =["zipcode"], inplace = True) 
dataset.info()
dataset
dataset.drop(columns =["date"], inplace = True) 
dataset
plt.figure(figsize=(6,5))
sns.countplot(x='bedrooms', data=dataset)
plt.figure(figsize=(10,5))
sns.set(style='darkgrid')
sns.countplot(y='bedrooms',data=dataset,order = dataset['bedrooms'].value_counts().index)
plt.show()
y = dataset['price']
y
X = dataset
X
X.drop(columns =["price"], inplace = True) 
X
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
X
y
plt.figure(figsize=(12,8))
sns.distplot(y)
y = np.log(y + 1)
y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X.shape
model = Sequential()

model.add(Dense(256, input_dim=17, kernel_initializer='normal', activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))

model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
model.fit(X_train, 
          y_train, 
          epochs=500, 
          batch_size=64,  
          verbose=1, 
          validation_split=0.2,
          callbacks = [early_stop])
# print(history.history.keys())

plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['loss', 'val_loss'], loc='upper right')
plt.show()
predictions = model.predict(X_test)
# Variance
explained_variance_score(y_test, predictions)

#Mean Absolute Error
mean_absolute_error(y_test, predictions)
#Mean Squared Error
mean_squared_error(y_test, predictions)
mean_squared_log_error(y_test, predictions)

