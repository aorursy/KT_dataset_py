import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
!ls
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/kc-housesales-data/kc_house_data.csv")
df.head()
# df.isnull().sum()
# df.describe().transpose()
plt.figure(figsize=(10,6))

sns.distplot(df['price'])
sns.countplot(df['bedrooms'])
df.corr()['price'].sort_values()
plt.figure(figsize=(8,6))

sns.scatterplot(x='price',y='sqft_living',data=df) 

plt.tight_layout()
plt.figure(figsize=(12,5))

sns.boxplot(x='bedrooms',y='price',data=df)
df.columns
plt.figure(figsize=(12,8))

sns.scatterplot(x='price',y='long',data=df)
plt.figure(figsize=(12,8))

sns.scatterplot(x='price',y='lat',data=df)
plt.figure(figsize=(12,8))

sns.scatterplot(x='long',y='lat',data=df,hue='price')
df.sort_values('price',ascending=False).head(20)
len(df) * 0.01
#removing 1% of most priced houses to get good color distribution
non_top_1_perc = df.sort_values('price',ascending=False).iloc[216:]
plt.figure(figsize=(12,8))

sns.scatterplot(x = 'long' ,y = 'lat' , data = non_top_1_perc ,hue = 'price',edgecolor=None,alpha=0.2,palette='RdYlGn')
plt.figure(figsize=(12,8))

sns.boxplot(x='waterfront',y='price',data=df)
df = df.drop('id',axis=1)
df['date'] = pd.to_datetime(df['date'])
df['date']
df['year'] = df['date'].apply(lambda date: date.year)

df['month'] = df['date'].apply(lambda date: date.month)
df.head()
plt.figure(figsize=(12,8))

sns.boxplot(x='month',y='price',data=df)
df.groupby('month').mean()['price'].plot()
df.groupby('year').mean()['price'].plot()
df = df.drop('date',axis=1)
df.columns
df.zipcode.value_counts()
df = df.drop('zipcode',axis=1)
df['yr_renovated'].value_counts()
df['sqft_basement'].value_counts()
X = df.drop('price',axis=1)

y = df['price'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
X_train.shape
model = Sequential()



model.add(Dense(19,activation='relu'))

model.add(Dense(19,activation='relu'))

model.add(Dense(19,activation='relu'))

model.add(Dense(19,activation='relu'))



model.add(Dense(1))



model.compile(optimizer='adam',loss='mse')
model.fit(x=X_train,y=y_train,

          validation_data=(X_test,y_test),

          batch_size=128,epochs=400)
losses = pd.DataFrame(model.history.history)
losses.plot()
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score
predictions = model.predict(X_test)
predictions
np.sqrt(mean_squared_error(y_test,predictions))
mean_absolute_error(y_test,predictions)
df['price'].describe()
5.402966e+05
explained_variance_score(y_test,predictions)
plt.figure(figsize=(12,6))

plt.scatter(y_test,predictions)

plt.plot(y_test,y_test,'r')
df.head()
single_house = df.drop('price',axis=1).iloc[0]
single_house = scaler.transform(single_house.values.reshape(-1,19))
model.predict(single_house)