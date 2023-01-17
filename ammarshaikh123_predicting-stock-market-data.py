import pandas as pd
df=pd.read_csv('../input/sphist.csv')
df.head()
df['Date'] = pd.to_datetime(df.Date)
df.info()
df=df.sort_values('Date')

df.head()
# Lets Pick 3 indicators to compute, and generate a different column for each one.



df['Past 5 days mean_Close']= df.Close.rolling(window=6).mean()



df['Past 30 days mean_Close']= df.Close.rolling(window=31).mean()



df['Past 365 days mean_Close']= df.Close.rolling(window=366).mean()

    
df.head(366)
# Removing the data which older tha 1951-01-03



df=df[df["Date"] > '1951-01-02']



df.head()
#let drop the rows with missing values



df.dropna(axis=0,inplace=True)
train=df[df["Date"] < '2013-01-01']



test=df[df["Date"] > '2013-01-01']



train.columns
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression



model=LinearRegression()



X_train= train[['Past 5 days mean_Close', 'Past 30 days mean_Close',

       'Past 365 days mean_Close']]

y_train=train['Close']



X_test= test[['Past 5 days mean_Close', 'Past 30 days mean_Close',

       'Past 365 days mean_Close']]

y_test=test['Close']



model.fit(X_train,y_train)



predictions=model.predict(X_test)



print("Mean absolute error is {}".format(mean_absolute_error(y_test,predictions)))

data=pd.read_csv('../input/sphist.csv')



data['Date'] = pd.to_datetime(data.Date)
data=data.sort_values('Date')

data.head()
data['Past 5 days mean_Close']= data.Close.rolling(window=6).mean()



data['Past 30 days mean_Close']= data.Close.rolling(window=31).mean()



data['Past 365 days mean_Close']= data.Close.rolling(window=366).mean()



data['Past 5 days mean_Volume']= data.Volume.rolling(window=6).mean()



data['Past 365 days mean_Volume']= data.Volume.rolling(window=366).mean()



data['Ratio of mean Volume for 5 and 365 days']= data['Past 5 days mean_Volume']/data['Past 365 days mean_Volume']



data.head(366)
data=data[data["Date"] > '1951-01-02']



data.dropna(axis=0,inplace=True)



data.head()
train=data[data["Date"] < '2013-01-01']



test=data[data["Date"] > '2013-01-01']



train.columns

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression



model=LinearRegression()



X_train= train[['Past 5 days mean_Close', 'Past 30 days mean_Close',

       'Past 365 days mean_Close', 'Past 5 days mean_Volume',

       'Past 365 days mean_Volume','Ratio of mean Volume for 5 and 365 days']]

y_train=train['Close']



X_test= test[['Past 5 days mean_Close', 'Past 30 days mean_Close',

       'Past 365 days mean_Close', 'Past 5 days mean_Volume',

       'Past 365 days mean_Volume','Ratio of mean Volume for 5 and 365 days']]

y_test=test['Close']





model.fit(X_train,y_train)



predictions=model.predict(X_test)



print("Mean absolute error is {}".format(mean_absolute_error(y_test,predictions)))
