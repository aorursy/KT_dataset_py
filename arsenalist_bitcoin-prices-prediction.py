#Let's do the usual imports

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#Reading Our data

df= pd.read_csv('../input/crypto-markets.csv', parse_dates=['date'], index_col='date')
df.head()
df.tail()
btc=df[df['symbol']=='BTC']#Extracting the bitcoin data from the dataframe

btc.drop(['volume','symbol','name','ranknow','market'],axis=1,inplace=True)#Just dropping columns here!
btc.isnull().any()#We don't have any NaN values luckily
btc.shape #We can see that we have 1696 observations for bitcoin here 
btc.tail()#Our data is pretty up to date it seems! 
sns.set()

sns.set_style('whitegrid')

btc['close'].plot(figsize=(12,6),label='Close')

btc['close'].rolling(window=30).mean().plot(label='30 Day Avg')# Plotting the 

#rolling 30 day average against the Close Price

plt.legend()
#I will be adding a feature to improve the model.This feature is provided by Tafarel Yan in his Kernel



btc['ohlc_average'] = (btc['open'] + btc['high'] + btc['low'] + btc['close']) / 4
btc.head()
btc['Price_After_Month']=btc['close'].shift(-30) #This will be our label
btc.tail()#We basically moved all our values 30 lines up in our last cell
#Preprocessing

from sklearn import preprocessing

btc.dropna(inplace=True)

X=btc.drop('Price_After_Month',axis=1)

X=preprocessing.scale(X)#We need to scale our values to input them in our model

y=btc['Price_After_Month']



from sklearn import cross_validation

X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.ensemble import RandomForestRegressor

reg=RandomForestRegressor(n_estimators=200,random_state=101)

reg.fit(X_train,y_train)

accuracy=reg.score(X_test,y_test)

accuracy=accuracy*100

accuracy = float("{0:.4f}".format(accuracy))

print('Accuracy is:',accuracy,'%')#This percentage shows how much our regression fits our data
preds = reg.predict(X_test)

print("The prediction is:",preds[1],"But the real value is:" ,y_test[1])

#We can see that our predictions are kind of accurate but we still need to work on on them a lot. 
#Apply our model and get our prediction

X_30=X[-30:]#We'll take the last 30 elements to make our predictions on them

forecast=reg.predict(X_30)
#creating a new column which contains the predictions! 

#Proceed at your own risk!  

from datetime import datetime, timedelta

last_date=btc.iloc[-1].name

modified_date = last_date + timedelta(days=1)

date=pd.date_range(modified_date,periods=30,freq='D')

df1=pd.DataFrame(forecast,columns=['Forecast'],index=date)

btc=btc.append(df1)

btc.tail()
#Now we'll plot our forecast! 

btc['close'].plot(figsize=(12,6),label='Close')

btc['Forecast'].plot(label='forecast')

plt.legend()