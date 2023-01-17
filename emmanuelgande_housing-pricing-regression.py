import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
df.head()
df.info()
df.shape
df.describe().transpose()
df.isnull().sum()
plt.figure(figsize=(12,8))

sns.distplot(df['price'])
sns.countplot(df['bedrooms'])
plt.figure(figsize=(12,8))

sns.scatterplot(x='price',y='sqft_living',data=df)
sns.boxplot(x='bedrooms',y='price',data=df)
plt.figure(figsize=(12,8))

sns.scatterplot(x='price',y='long',data=df)
plt.figure(figsize=(12,8))

sns.scatterplot(x='price',y='lat',data=df)
plt.figure(figsize=(12,8))

sns.scatterplot(x='long',y='lat',data=df,hue='price')

df.sort_values('price',ascending=False).head(20)
len(df)*0.01
non_top_1_perc = df.sort_values('price',ascending=False).iloc[216:]
plt.figure(figsize=(12,8))

sns.scatterplot(x='long',y='lat',

                data=non_top_1_perc,hue='price',

                palette='RdYlGn',edgecolor=None,alpha=0.2)
sns.boxplot(x='waterfront',y='price',data=df)
df.head()
df.info()
df = df.drop('id',axis=1)

df.head()
import datetime 

df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].apply(lambda time: time.month)
df['year'] = df['date'].apply(lambda time: time.year)
sns.boxplot(x='year',y='price',data=df)
sns.boxplot(x='month',y='price',data=df)
df.groupby('month').mean()['price'].plot()
df.groupby('year').mean()['price'].plot()
df = df.drop('date',axis=1)
df.columns
df['zipcode'].value_counts()
df = df.drop('zipcode',axis=1)
df.head()
df['sqft_basement'].value_counts()
df['yr_renovated'].value_counts()
X = df.drop('price',axis=1)

y = df['price']
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
model.fit(x=X_train,y=y_train.values,

          validation_data=(X_test,y_test.values),

          batch_size=128,epochs=400)
losses = pd.DataFrame(model.history.history)
losses.plot()
from sklearn.metrics import mean_squared_error, mean_absolute_error,explained_variance_score
X_test
predictions = model.predict(X_test)
mean_absolute_error(y_test,predictions)
np.sqrt(mean_squared_error(y_test,predictions))
explained_variance_score(y_test,predictions)
df['price'].mean()
df['price'].median()
# Our predictions

plt.scatter(y_test,predictions)



# Perfect predictions

plt.plot(y_test,y_test,'r')
errors = y_test.values.reshape(6484,1) - predictions
sns.distplot(errors)
single_house = df.drop('price',axis=1).iloc[0]
single_house = scaler.transform(single_house.values.reshape(-1, 19))
single_house
model.predict(single_house)
df.iloc[0]