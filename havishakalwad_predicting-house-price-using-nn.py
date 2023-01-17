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
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
# importing the dataset
df # a look at the dataset
df.isnull().sum()
# checking for any missing data
df.describe().transpose()
# understanding the data
plt.figure(figsize=(10,6))
sns.distplot(df['price'])
# checking the price column distribution
# most of the house price fall between 0 - 1.5 million dollars
sns.countplot(df['bedrooms'])
# most of the houses have between 2 - 5 bedrooms on average
df.corr()
# correlations of all the columns with respect to each other
df.corr()['price'].sort_values()
# correlation of the price column alone with respect to other columns and sorted in ascending order
plt.figure(figsize=(10,6))
sns.scatterplot(x='price',y='sqft_living',data=df)
# a very linear relationship observed as they are highly correlated as seen from the above table
plt.figure(figsize=(10,6))
sns.scatterplot(x='price',y='bathrooms',data=df)
sns.boxplot(x='bedrooms',y='price',data=df)
# bedrooms and price correlation
plt.figure(figsize=(12,8))
sns.scatterplot(x = 'price', y = 'long', data = df)
# there is a lot of price variations based on the longitude of the location of the house
plt.figure(figsize=(12,8))
sns.scatterplot(x = 'price', y = 'lat', data = df)
# expensive housing areas in some particular latitudes
# at a certain combination of lat and long there seems to be an expensive neighbourhood
plt.figure(figsize = (15,12))
sns.scatterplot(x = 'long', y = 'lat', data = df, hue = 'price')
# the shape of the distribution matches the King county in Seattle
# darker points are the expensive neighborhoods
df.sort_values('price', ascending = False).head(20)
# only about 20 houses are in the range of 3 - 7 million dollars
# these can be considered as outliers in the dataset
len(df)
# 21613 houses in the dataset
# 1% of 21613 = 216 houses
bottom_99_percent = df.sort_values('price', ascending = False).iloc[216:]
# this drops all the really expensive houses which were the outliers
bottom_99_percent
plt.figure(figsize = (15,12))
sns.scatterplot(x = 'long', y = 'lat', data = bottom_99_percent, hue = 'price', palette = 'RdYlGn')
# a lot clearer color distribution of the expensive houses
sns.boxplot(x='waterfront',y='price',data=df)
# waterfront houses are more expensive
df
# can drop ID
df.drop('id', axis = 1, inplace = True)
df['date'] = pd.to_datetime(df['date'])
# convert the date column items into a date-time object. The formatting also changes.
# now its easier to extract info like the month and year automatically
df['date']
# feature engineering or feature extraction can be done on this object now
df['year'] = df['date'].apply(lambda date: date.year)
df['year']
df['month'] = df['date'].apply(lambda date: date.month)
df['month']
df
# the year and month columns are now added to this. Exploratory data analysis can be done on to see if they are useful.
df.groupby('month').mean()['price'].plot()
# to check if any significant relationship between month sold and price of the house
# about $60k price difference during the sprint and summer months.
df.groupby('year').mean()['price'].plot()
# sales increasing in price as the time goes by
df = df.drop('date', axis = 1)
df
df['zipcode'].value_counts()
# zipcodes cannot be left as numerical values. They have to be treated as a categorical variable.
# 70 categories of zipcodes have to be created to make dummy variables here
# 
df = df.drop('zipcode', axis = 1)
df

X = df.drop('price', axis =1).values
y = df['price']
X
y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)
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
model.add(Dense(1)) # output layer neuron

model.compile(optimizer='adam',loss='mse')
model.fit(x=X_train,y=y_train.values,
          validation_data=(X_test,y_test.values),
          batch_size=128,epochs=400)
losses = pd.DataFrame(model.history.history)
# Loss: loss on training set
# val_loss: loss on test set
losses.plot()
# both lines are close so no overfitting of the model
# decrease in both the training and validation loss
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
y_pred = model.predict(X_test)
mean_absolute_error(y_test,y_pred)
np.sqrt(mean_squared_error(y_test,y_pred))
explained_variance_score(y_test,y_pred)
df['price'].mean()
df['price'].median()
# Our predictions
plt.scatter(y_test,y_pred)

# Perfect predictions
plt.plot(y_test,y_test,'r')
