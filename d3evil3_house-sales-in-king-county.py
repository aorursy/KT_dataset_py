import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
df.isnull().sum()
df.describe()
plt.figure(figsize=(10,6))
sns.distplot(df['price'])
sns.countplot(df['bedrooms'])
df.corr()
df.corr()['price'].sort_values()
plt.figure(figsize=(10,5))
sns.scatterplot(x='price',y='sqft_living',data=df)
plt.figure(figsize=(10,6))
sns.boxplot(x='bedrooms', y= 'price',data=df)
df.columns
plt.figure(figsize=(10,6))
sns.scatterplot(x='price',y='long',data=df)
plt.figure(figsize=(10,6))
sns.scatterplot(x='price',y='lat',data=df)
plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',data=df,hue='price')
non_top_1 = df.sort_values('price',ascending=False).iloc[216:]
plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',data=non_top_1,hue='price',edgecolor=None,alpha=0.2,palette='RdYlGn')
sns.boxplot(x='waterfront',y='price',data=df)
df = df.drop('id',axis=1)
df.head()
df['date'] = pd.to_datetime(df['date']) 
df.head()
def year_extraction(date):
    return date.year
df['year'] = df['date'].apply(lambda date: date.year)
df['month'] = df['date'].apply(lambda date: date.month)

df.head()
plt.figure(figsize=(10,6))
sns.boxplot(x='month',y='price',data=df)
df.groupby('month').mean()['price'].plot()
df.groupby('year').mean()['price'].plot()
df=df.drop('date',axis=1)
df.columns
df['zipcode'].value_counts()
df = df.drop('zipcode',axis=1)
df['yr_renovated'].value_counts()
df['sqft_basement'].value_counts()
X = df.drop('price',axis=1).values
y = df['price'].values
from sklearn.model_selection import train_test_split
train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler  = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error , explained_variance_score
predict = model.predict(X_test)
predict
np.sqrt(mean_squared_error(y_test,predict))
mean_absolute_error(y_test,predict)
explained_variance_score(y_test,predict)
plt.figure(figsize=(12,6))
plt.scatter(y_test,predict)
plt.plot(y_test,y_test,'r')
single_house = df.drop('price',axis=1).iloc[0]
single_house = scaler.transform(single_house.values.reshape(-1,19))
model.predict(single_house)
df.head(1)
