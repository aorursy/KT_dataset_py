import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
df.head()
df.info()
sns.barplot(data=df,x='bedrooms',y='price')
df[df['bedrooms']==33]
df[df['bedrooms']==11]
sns.scatterplot(data=df,x='sqft_lot',y='price')
df[df['price']>7000000]
df[df['sqft_lot'] == df['sqft_lot'].max()]
sns.lmplot(data=df,x='bedrooms',y='bathrooms')
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df.drop(['id','date','zipcode','lat','long','price'],axis=1),df['price'],test_size=0.3)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
pred = model.predict(X_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error
sns.distplot(y_test-pred,bins=10,kde=True)
print('Mean Absolute Error: ', mean_absolute_error(y_test,pred))

print('Mean Squared Error:', mean_squared_error(y_test,pred))

print('Root Mean Squared Error', np.sqrt(mean_squared_error(y_test,pred)))