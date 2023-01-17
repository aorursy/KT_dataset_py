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
df = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")

df.head()
df.keys()
# Check null values

df.isnull().sum()
# Overview of dataset

df.describe().transpose()
# Since it is a continuous label, I would like to see a histogram/ distribution of the label

plt.figure(figsize=(10,6))

sns.distplot(df['price'])
# Categorical - Bedrooms

plt.figure(figsize=(10,6))

sns.countplot(df['bedrooms'])
df.corr().price.sort_values()
# Exploring highly correlated features with the label through SCATTERPLOT

plt.figure(figsize=(10,6))

sns.scatterplot(x='price', y='sqft_living', data=df)
# Boxplot of no. of bedrooms and the price

plt.figure(figsize=(10,6))

sns.boxplot(x='bedrooms',y='price',data=df)
# Longitude vs. Price

plt.figure(figsize=(12,8))

sns.scatterplot(x='price', y='long', data=df)
# Latitude vs. Price

plt.figure(figsize=(12,8))

sns.scatterplot(x='price', y='lat', data=df)
# Looking at both Lat and Long with a hue of Price

plt.figure(figsize=(12,8))

sns.scatterplot(x='long', y='lat', hue='price', data=df)
df.sort_values('price',ascending=False).head(20)
# Sample out top 1% of all houses

len(df)*(0.01)
bottom_99_perc = df.sort_values('price',ascending=False).iloc[216:]
plt.figure(figsize=(12,8))

sns.scatterplot(x='long',y='lat',

                data=bottom_99_perc,hue='price',

                palette='RdYlGn',edgecolor=None,alpha=0.2)
# Whether or not house is in front of waterfront

sns.boxplot(x='waterfront',y='price',data=df)
df = df.drop('id', axis=1)
# TO datetime object

df['date'] = pd.to_datetime(df['date'])
# New column Year

df['year'] = df['date'].apply(lambda date: date.year)
# New column Month

df['month'] = df['date'].apply(lambda date:date.month)
# Monthly

plt.figure(figsize=(10,6))

sns.boxplot(x='month',y='price',data=df)
# Mean Price varying throughout the months

df.groupby('month').mean()['price'].plot()
# Mean Price varying throught the years

df.groupby('year').mean()['price'].plot()
# Dropping the date

df = df.drop('date', axis=1)
df.columns
df['zipcode'].value_counts()
df = df.drop('zipcode',axis=1)
# could make sense due to scaling, higher should correlate to more value

df['yr_renovated'].value_counts()
df['sqft_basement'].value_counts()
X = df.drop('price',axis=1).values

y = df['price'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train.shape
X_test.shape
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation

from tensorflow.keras.optimizers import Adam
model = Sequential()



model.add(Dense(19,activation='relu'))

model.add(Dense(19,activation='relu'))

model.add(Dense(19,activation='relu'))

model.add(Dense(19,activation='relu'))

model.add(Dense(1))



model.compile(optimizer='adam',loss='mse')
model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), batch_size=128, epochs=400)

# smaller batch size, the longer the training is going to take, but the less likely that we are overfitting

# because we are not parsing in the entire training set at once, and only focusing on smaller batches
model.history.history 

# returns a dictionary
# But because we parsed in validation data tuple, and if we convert it to a df

# I will not only get Loss on training set, I will also get Loss on test set/ validation data



losses = pd.DataFrame(model.history.history)
losses.plot()

# We want in decrease in both and no increase in the validation set

# If val_loss begin to spike, it means overfitting, because we will have a much larger loss on validation data
# MSE, MAE

from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
predictions = model.predict(X_test)
predictions
# MSE

mean_squared_error(y_test,predictions)
# RMSE

np.sqrt(mean_squared_error(y_test,predictions))
# MAE

mean_absolute_error(y_test,predictions)

# Off by $100,000

# Is that good or bad? Need to see the actual data itself
df['price'].describe()
explained_variance_score(y_test, predictions)

# This tells us how much variance is being explained by our actual model

# Best score is 1, lower values are worse
plt.figure(figsize=(12,6))

plt.scatter(y_test, predictions)

plt.plot(y_test,y_test, 'r')
errors = y_test.reshape(6484, 1) - predictions
plt.figure(figsize=(12,8))

sns.distplot(errors)
single_house = df.drop('price',axis=1).iloc[0]
single_house = scaler.transform(single_house.values.reshape(-1,19))
model.predict(single_house)
df.head(1)