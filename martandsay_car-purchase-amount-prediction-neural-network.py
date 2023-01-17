# Welcome to the Car Purchase Amount Prediction case study. This is the very basic case study.

# Here we will simply see how to load data, Visualize data, clean data, train test split, prepare model, predict and evaluate.

# Upvote and share if your like it.

# https://www.facebook.com/codemakerz
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/car-purchase-data/Car_Purchasing_Data.csv', encoding='cp1252')
df.head()
sns.pairplot(df)
# We do not need customer name, email and country as these factors will not affect your purchase.

# also we will remove car purchase amount column as it is our target column and target column should not be in input matrix.

# in y we will only store our target column.



X = df.drop(["Customer Name", "Customer e-mail", "Country", "Car Purchase Amount"], axis=1)

y = df["Car Purchase Amount"]
X[0:10]
y[0:10]
print(X.shape)

print(y.shape)
# If you will not scale your data, model will predict terrible things. Your data should be in same scale. IT makes your model more accurate and faster.

# So lets scale our data with min-max scaler.

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

y = y.values.reshape(-1, 1)

y_scaled = scaler.fit_transform(y)
scaler.data_max_
from sklearn.model_selection import train_test_split
# Splitting our data. 30% test and 70% training data.

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3)
# importing tensorflow library to create our ANN model. If your model has more than 2 layers it is called deep learning network.

import tensorflow.keras

from keras.models import Sequential

from keras.layers import Dense
model = Sequential()

# input dim is number of inputs. It is always of the same size as feature size. (X)

# 40 is number of neurons or units.

model.add(Dense(40, input_dim=5, activation='relu'))

model.add(Dense(40, activation='relu'))

model.add(Dense(1, input_dim=5, activation='linear'))
# You can check your model details using summary function

model.summary()
model.compile(optimizer="adam", loss='mean_squared_error') # Compile your model.
epoch_hist = model.fit(X_train, y_train, epochs=100, verbose=1, validation_split=0.2, batch_size=25)
epoch_hist.history.keys()
epoch_hist.history
# So according to below figure you will see that for the first 20 epochs our model showing a big drop in loss or you can say improvement.

# but after that improvement is constant or not that much significant. So from here we can conclude that for given number of nuerons we can only use 20-25 epochs to get a good model.

plt.figure(figsize=(10, 8))

plt.plot(epoch_hist.history['loss'])

plt.plot(epoch_hist.history['val_loss'])

plt.legend(["Training Loss", "Validation Loss"])

plt.title("Training and Validation Loss Throughout the epochs")

plt.xlabel("Epochs")

plt.ylabel("Info Loss")

# Predict Test Records

y_pred = model.predict(X_test)

y_pred[0:10]
# Predict Custom Value

X_test_data = np.array([[1, 50, 45000, 3000, 500000]])

predict = model.predict(X_test_data)

print(predict)
# At the end i will suggest you to play with model's hyper paramters. You should try with different number or models and epochs and try to see what result you get. There is not best

# value. It all depends upon your data and your observations.
# Thank you ... upvote if you liked.