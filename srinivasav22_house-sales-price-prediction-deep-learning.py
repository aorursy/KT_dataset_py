# Import Main Libraries for Dataframe and Visualisation

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns

df = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")



df.head()
# Check the Shape of the Data 

df.shape
# Check the Column Names of the Data

df.columns
# check the Stats of the Data 

df.describe().transpose() # i have transpose because length of columns more
# Lets Check for the NULL Values in the Data 



df.isnull().sum()
# Lets Visualize Price Column to understand more 

plt.figure(figsize=(10,6))

sns.distplot(df['price'])
sns.countplot(df['bedrooms']) # Highest Sold bedrooms are 3
df.sort_values('price',ascending=False).head(20) # Max price 7700000
len(df)*(0.01)
non_top_1_perc = df.sort_values('price',ascending=False).iloc[216:]
plt.figure(figsize=(12,8))

sns.scatterplot(x='long',y='lat',

                data=non_top_1_perc,hue='price',

                palette='RdYlGn',edgecolor=None,alpha=0.2)
sns.boxplot(x='waterfront',y='price',data=df)
df = df.drop('id',axis=1)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].apply(lambda date:date.month)

df['year'] = df['date'].apply(lambda date:date.year)

sns.boxplot(x='year',y='price',data=df)
df = df.drop('date',axis=1)
df = df.drop('zipcode',axis=1)
# could make sense due to scaling, higher should correlate to more value

df['yr_renovated'].value_counts()
df['sqft_basement'].value_counts()
X = df.iloc[:,df.columns != 'price']

y = df.iloc[:,df.columns == 'price']
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



print(X_train.shape)

print(X_test.shape)
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

          batch_size=128,epochs=200)
losses = pd.DataFrame(model.history.history)

losses.plot()
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score

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
errors = y_test.values.reshape(6484, 1) - predictions
sns.distplot(errors)