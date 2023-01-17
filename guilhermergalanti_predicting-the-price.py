# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')
# how many nulls?

df.isnull().sum()
df.describe().transpose()
plt.figure(figsize=(10,6))

sns.distplot(df['price'])
plt.figure(figsize = (10,6))

sns.countplot(df['bedrooms'])
df.corr()['price'].sort_values(ascending = False)
plt.figure(figsize = (13,8))

sns.heatmap(df.corr(), cmap='coolwarm')
plt.figure(figsize = (13,8))

sns.scatterplot(x = 'price', y = 'sqft_living', data = df)
plt.figure(figsize=(16,8))

sns.boxplot(x = 'bathrooms', y = 'price', data = df)
plt.figure(figsize = (13,8))

sns.scatterplot(x = 'long', y = 'lat', data=df, hue = 'price')
len(df)*0.01
# Removendo pontos distantes

# Cortaremos os preços em 3.000.000

non_top = df.sort_values('price', ascending = False).iloc[216:]
plt.figure(figsize = (13,8))

sns.scatterplot(x = 'long', y = 'lat', data=non_top, hue = 'price', edgecolor = None, alpha = 0.2, palette = 'RdYlGn')
plt.figure(figsize = (13,8))

sns.boxplot(x = 'waterfront', y='price', data = df)
df = df.drop('id', axis=1)
df['date'] = pd.to_datetime(df['date'])

df.head()
df['year'] = df['date'].apply(lambda date: date.year)

df['month'] = df['date'].apply(lambda date: date.month)
plt.figure(figsize = (10,6))

df.groupby('month').mean()['price'].plot()
df = df.drop('date', axis = 1)

df = df.drop('zipcode', axis = 1)
X = df.drop('price', axis = 1).values

y = df['price'].values
from sklearn.model_selection import train_test_split
#Separando treino e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
#Escalando valores

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
X_train.shape
model = Sequential()



#como temos 19 features, é interessante termos 19 neurônios

model.add(Dense(19, activation = 'relu'))

model.add(Dense(19, activation = 'relu'))

model.add(Dense(19, activation = 'relu'))

model.add(Dense(19, activation = 'relu'))



model.add(Dense(1))



model.compile(optimizer = 'adam', loss = 'mse')
model.fit(X_train, y_train, validation_data = (X_test, y_test), batch_size = 128, epochs = 400, verbose = 0)
loss = pd.DataFrame(model.history.history)
loss.plot()
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
predictions = model.predict(X_test)
np.sqrt(mean_squared_error(y_test, predictions))
mean_absolute_error(y_test, predictions)
explained_variance_score(y_test, predictions)
plt.figure(figsize = (13,8))

plt.scatter(y_test, predictions)

plt.xlabel('Valores de teste')

plt.ylabel('Valores preditos')