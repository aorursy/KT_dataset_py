# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

%pylab inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import preprocessing, cross_validation, svm

from sklearn.linear_model import LinearRegression

import math

from scipy import stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Initial exploration of the Dataset

BTC_Price = pd.read_csv("../input/BITCOIN-COINMARKETCAP.csv")

print(BTC_Price.head())

BTC_Price.describe()
# Identify the Missing values 

print(BTC_Price.tail(10))

print(BTC_Price.isnull().sum())

print(BTC_Price.info())
# Identify the Price variance for the each days

# Variance = Close price minus the Open price 

# Negative value indicate price has declined for that day and Positive value represent increase in price

BTC_Price['Variance'] = ((BTC_Price["Close"] - BTC_Price["Open"])/BTC_Price["Close"])*100



# Frequeny of change for a given day (High - Low)

BTC_Price['Freq'] = ((BTC_Price["High"] - BTC_Price["Low"])/BTC_Price["High"])*100

print (BTC_Price.head(10))
df = BTC_Price.filter(['Date','High','Low','Variance','Freq'], axis=1)

df['Date']=pd.to_datetime(df['Date'])



# Reverse the Dataframe to display the latest Price

df=df.iloc[::-1]

df.set_index('Date',inplace=True)

print(df.columns)
df['High'].plot(grid =True)

#pylab.rcParams['figure.figsize'] = (15, 9) 

plt.show()
df1 = BTC_Price.head(5000)

df1['Date']=pd.to_datetime(df1['Date'])

#plt.figure(figsize=(15,10))

ax= sns.barplot(x=df1['Date'], y=df1['High'])

plt.show()
f,ax = plt.subplots(figsize = (9,15))

sns.pointplot(x=df1['Date'],y=df1['Open'],data=df1,color='lime',alpha=0.8,label='Variance' )

sns.pointplot(x=df1['Date'],y=df1['Close'],data=df1,color='red',alpha=0.8,label='Freq' )

ax.legend(loc='lower right')

plt.grid()

plt.show()
df['Variance'].plot(grid = True)

#pylab.rcParams['figure.figsize'] = (15, 9) 

plt.show()
forecast_out = int(math.ceil(0.01 * len(BTC_Price)))

print(forecast_out)

BTC_Price['New Close'] = BTC_Price['Close'].shift(-forecast_out)

BTC_Price.tail()
print(BTC_Price.tail())

print(BTC_Price.describe())
BTC_Price.dropna(inplace=True)

print(BTC_Price.describe())

X=np.array(BTC_Price.drop(['Market Cap','Volume','Date'], axis=1))

X=preprocessing.scale(X)

print(stats.describe(X))

X_lately = X[-forecast_out:]

#X = X[:- forecast_out]

BTC_Price.dropna(inplace=True)

print(BTC_Price.describe())

y = np.array(BTC_Price['New Close'])

print(stats.describe(X))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)

clf.fit(X_train, y_train)

confidence = clf.score(X_test, y_test)

print(confidence)

forecast_set = clf.predict(X_lately)

print(forecast_set)