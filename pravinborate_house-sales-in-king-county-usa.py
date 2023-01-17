# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')
data.head()
data.isnull().sum()
data.describe().T
plt.figure(figsize=(12,8))

sns.distplot(data['price'])
plt.figure(figsize=(20,8))

sns.heatmap(data.corr(),annot = True)
data.corr()['price'].sort_values(ascending = False)
var = ['sqft_living','grade','sqft_above','sqft_living15','bathrooms','view','sqft_basement','bedrooms','lat','waterfront','floors','yr_renovated','sqft_lot','sqft_lot15','yr_built','condition','long',]
plt.figure(figsize = (12,6))

sns.scatterplot(x='price',y = 'sqft_living',hue = 'price',data = data)
data.head()
plt.figure(figsize = (12,6))

sns.countplot(x='bedrooms',data=data)
plt.figure(figsize = (12,6))

sns.scatterplot(x='long',y = 'lat',hue = 'price',data = data)
data = data.drop('id',axis = 1)
data.head()
data['date'] = pd.to_datetime(data['date'])
data.head()
data['year'] = data['date'].apply(lambda date : date.year)

data['month'] = data['date'].apply(lambda date : date.month)
data.head()
plt.figure(figsize=(12,8))

sns.boxenplot(x = 'month',y = 'price',data = data)
data.groupby('month').mean()['price']
data.groupby('month').mean()['price'].plot()
data.groupby('year').mean()['price'].plot()
data = data.drop('date',axis = 1)
data.head()
data = data.drop('zipcode',axis = 1)
data['yr_renovated'].value_counts()
data['sqft_basement'].value_counts()
X = data.drop('price',axis = 1).values

y = data['price'].values
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 101)
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)
X_train.shape
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(

    19,activation = 'relu'

))

model.add(Dense(

    19,activation = 'relu'

))

model.add(Dense(

    19,activation = 'relu'

))

model.add(Dense(

    19,activation = 'relu'

))



model.add(Dense(

    1

))
model.compile(optimizer = 'adam',loss='mse')
model.fit(X_train,y_train,validation_data=(X_test,y_test),

         batch_size=128,epochs = 400)
model_history = pd.DataFrame(model.history.history)

model_history
model_history.plot()
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error,explained_variance_score
y_pred = model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))
mean_absolute_error(y_test,y_pred)
explained_variance_score(y_test,y_pred)
plt.figure(figsize=(12,8))

plt.scatter(y_test,y_pred)

plt.plot(y_test,y_test,'r')