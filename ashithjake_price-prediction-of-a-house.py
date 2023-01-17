#import libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score,r2_score

from tensorflow.keras.callbacks import EarlyStopping
#fetch data

house_data = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')

house_data.head()
house_data.columns
house_data.isnull().sum()
house_data.describe().transpose()
plt.figure(figsize=(10,6))

sns.distplot(house_data['price'])

plt.show()
plt.figure(figsize=(10,6))

sns.countplot(house_data['bedrooms'])

plt.show()
house_data.corr()['price'].sort_values(ascending=False)
plt.figure(figsize=(10,6))

sns.scatterplot(x='price',y='sqft_living',data=house_data)
plt.figure(figsize=(16,6))

sns.boxplot(x='bedrooms',y='price',data=house_data)
plt.figure(figsize=(16,6))

sns.scatterplot(x='long',y='lat',data=house_data,hue='price',edgecolor=None,alpha=0.2,palette='RdYlGn')

plt.show()
plt.figure(figsize=(16,6))

sns.boxplot(x='waterfront',y='price',data=house_data)

plt.show()
house_data.info()
# ID has no actual feature but just a unique value for each of the records

house_data.drop('id',axis=1,inplace=True)
#'date' feature is a string. Lets convert it to datetime object

house_data['date'] = pd.to_datetime(house_data['date'])

house_data['date'].head()
#lets extract year and month info from it

house_data['year'] = house_data['date'].apply(lambda date:date.year)

house_data['month'] = house_data['date'].apply(lambda date:date.month)

house_data[['year','month']].head()
plt.figure(figsize=(16,6))

sns.boxplot(x='month',y='price',data=house_data)
plt.figure(figsize=(16,6))

sns.boxplot(x='year',y='price',data=house_data)
plt.figure(figsize=(16,6))

house_data.groupby('month').mean()['price'].plot()
plt.figure(figsize=(16,6))

house_data.groupby('year').mean()['price'].plot()
house_data.drop('date',axis=1,inplace=True)
house_data['zipcode'].value_counts()
house_data.drop('zipcode',axis=1,inplace=True)
house_data.columns
X = house_data.drop('price',axis=1).values

y = house_data['price'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
X_train.shape
model = Sequential()

model.add(Dense(19,activation='relu'))

model.add(Dense(19,activation='relu'))

model.add(Dense(19,activation='relu'))

model.add(Dense(19,activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=128,epochs=800)
losses = pd.DataFrame(model.history.history)

losses.head()
losses.plot()
predictions = model.predict(X_test)
mean_squared_error(y_test,predictions)
np.sqrt(mean_squared_error(y_test,predictions))
mean_absolute_error(y_test,predictions)
explained_variance_score(y_test,predictions)
r2_score(y_test,predictions)
plt.figure(figsize=(12,6))

plt.scatter(y_test,predictions)

plt.plot(y_test,y_test,'r')