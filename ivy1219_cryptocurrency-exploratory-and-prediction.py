# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv('../input/crypto-markets.csv',index_col = 'date',parse_dates=True)

df.head()

df.shape

# let us see the trends of Bitcoin price since 2013

Bitcoin_price = df[df['name']=='Bitcoin']

Bitcoin_price[['open','high','low','close']].plot(figsize=(16,8))

plt.show()

#since 2017, the Bitcoin price has a rocket speed,let us see closer

Bitcoin_price1718 = df[df['name']=='Bitcoin'].loc['2017':]

Bitcoin_price1718[['open','high','low','close']].plot(figsize=(16,8))

plt.show()

# Heightened volatility comes from Dec.2017.

#Frankly speaking,if I got on the bus since 2015 or Jan.2017,Now I would be very very very rich.



# I'd like to know the top 10 rank coins

df[df['ranknow'] <= 10].groupby('ranknow').name.unique()

# then know more about the top 5 coins

Bitcoin = df[df['name']=='Bitcoin'].loc['2017':]

Ethereum = df[df['name']=='Ethereum'].loc['2017':]

Ripple = df[df['name']=='Ripple'].loc['2017':]

Bitcoin_cash = df[df['name']=='Bitcoin Cash'].loc['2017':]

Cardano = df[df['name']=='Cardano'].loc['2017':]



# draw pic of MarketCap

plt.figure(figsize=(16,8))

(Bitcoin['market']/1000000).plot(color='red', label='Bitcoin')

(Ethereum['market']/1000000).plot(color='green', label='Ethereum')

(Ripple['market']/1000000).plot(color='blue', label='Ripple')

(Bitcoin_cash['market']/1000000).plot(color='yellow', label='Bitcoin Cash')

(Cardano['market']/1000000).plot(color='pink', label='Cardano')

plt.legend()

plt.title('Top5 Cryptocurrency Market Cap (Million USD)')

plt.show()



# draw pic or Volume

plt.figure(figsize=(16,8))

(Bitcoin['volume']/1000000).plot(color='red', label='Bitcoin')

(Ethereum['volume']/1000000).plot(color='green', label='Ethereum')

(Ripple['volume']/1000000).plot(color='blue', label='Ripple')

(Bitcoin_cash['volume']/1000000).plot(color='yellow', label='Bitcoin Cash')

(Cardano['volume']/1000000).plot(color='pink', label='Cardano')

plt.legend()

plt.title('Top5 Cryptocurrency Transactions Volume (Million Units)')

plt.show()



#I'd like to try some prediction now. I guess my dream is to be a Chief Fortuneteller Officer. 

#steps:create new dataframe, add new feature, model fit and optimize

Bitcoin_price['mean'] = (Bitcoin_price['open'] + Bitcoin_price['high'] + Bitcoin_price['low'] + Bitcoin_price['close']) / 4

Bitcoin_price['prediction'] = Bitcoin_price['close'].shift(-30)

# drop columns I don't use this time.

Bitcoin_price.drop(['slug','volume','symbol','name','ranknow','market','close_ratio','spread'],axis=1,inplace=True)

Bitcoin_price.tail() # just  check

#let us roll

from sklearn import preprocessing

Bitcoin_price.dropna(inplace=True)

X = Bitcoin_price.drop('prediction',axis = 1)

X = preprocessing.scale(X)

y= Bitcoin_price['prediction']



from sklearn import cross_validation

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size =0.1,random_state =101)



from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(n_estimators = 200,random_state = 101)

reg.fit(X_train,y_train)

accuracy = reg.score(X_test,y_test)

accuracy = accuracy*100

accuracy = float('{0:.2f}'.format(accuracy))

print ('Accuracy is: ',accuracy,'%')



preds = reg.predict(X_test)

print ('The predicted price is: ',preds[1],'The real price is:',y_test[1])

#So far,Accuracy looks good, predicted price sucks. C'est la vie and don't be fool or discouraged.

# It always happen in real world.



# Then, let us guess what will happen after 30 days?

X_30 =X[-30:]

forecast = reg.predict(X_30)



from datetime import datetime,timedelta

last_date = Bitcoin_price.iloc[-1].name

modified_date = last_date + timedelta(days=1)

date=pd.date_range(modified_date,periods=30,freq='D')

df1=pd.DataFrame(forecast,columns=['Forecast'],index=date)

Bitcoin_price=Bitcoin_price.append(df1)

Bitcoin_price.tail(30)

# In real world ,The price of Bitcoin reached 18870 in Dec.16th,my prediction is 18401;

# The next day,it went to 19458 in Dec.17th and my predicition is 18921;

# It looks smart.



#let us see what the baby looks like

Bitcoin_price['close'].plot(figsize=(16,8),label='Close',color = 'red')

Bitcoin_price['Forecast'].plot(label='forecast',color = 'blue')

plt.legend()

plt.show()



# I guess it will maintain the volatile markets for a while.

# Basically, I can use the model to predict any cryptocurrency's price as long as it has enough real-time data.

# I made this for fun, don't be fool or count on it to make a fortune.




