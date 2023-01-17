# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Get the data and have a look at it



url_stocks='../input/stock-time-series-20050101-to-20171231/all_stocks_2006-01-01_to_2018-01-01.csv'

#url_stocks_recent='../input/stock-time-series-20050101-to-20171231/all_stocks_2017-01-01_to_2018-01-01.csv'





stocks_df=pd.read_csv(url_stocks,index_col='Date',parse_dates=[0])

#stocks_recent_df=pd.read_csv(url_stocks_recent,index_col='Date',parse_dates=[0])

print(stocks_df.head())

#Slicing out the data we need

df=stocks_df[(stocks_df.Name=='AMZN') | (stocks_df.Name=='WMT')]

print(df.head())

#Checking for null values

print(df.info())

#Drop the single row with a Null value

df.dropna(axis=0,inplace=True)

print(df.isnull().sum())
#Separting the amazon and walmart data 

amzn_df=df[df.Name=='AMZN']

wmt_df=df[df.Name=='WMT']
#Basic statistics of amazon data

print(amzn_df.describe())

#Basic statistics of walmart data

print(wmt_df.describe())
#Basic Plots

plt.title("Opening price")

amzn_df['Open'].plot(label='AMZN')

wmt_df['Open'].plot(label='WMT')

plt.legend()

plt.show()

plt.title("Closing price")

amzn_df['Close'].plot(label='AMZN')

wmt_df['Close'].plot(label='WMT')

plt.legend()

plt.show()

plt.title("High price")

amzn_df['High'].plot(label='AMZN')

wmt_df['High'].plot(label='WMT')

plt.legend()

plt.show()

plt.title("Low price")

amzn_df['Low'].plot(label='AMZN')

wmt_df['Low'].plot(label='WMT')

plt.legend()

plt.show()

plt.title("Volume")

amzn_df['Volume'].plot(label='AMZN')

wmt_df['Volume'].plot(label='WMT')

plt.legend()

plt.show()

# We smoothen out the volume plot by taking rolling averages of 25 days

amzn_vol_mean=amzn_df['Volume'].rolling(window=25).mean()

wmt_vol_mean=wmt_df['Volume'].rolling(window=25).mean()





amzn_vol_mean.plot(label='AMZN')

wmt_vol_mean.plot(label='WMT')

plt.legend()

plt.show()



plt.figure(1)

plt.subplot(211)

amzn_df['Open'].hist()

plt.subplot(212)

amzn_df['Open'].plot(kind='kde')

plt.title("AMZN Open")

plt.show()



plt.figure(1)

plt.subplot(211)

amzn_df['Close'].hist()

plt.subplot(212)

amzn_df['Close'].plot(kind='kde')

plt.title("AMZN Close")

plt.show()
plt.figure(1)

plt.subplot(211)

wmt_df['Open'].hist()

plt.subplot(212)

wmt_df['Open'].plot(kind='kde')

plt.title("WMT Open")

plt.show()

plt.figure(1)

plt.subplot(211)

wmt_df['Close'].hist()

plt.subplot(212)

wmt_df['Close'].plot(kind='kde')

plt.title("WMT Close")

plt.show()
#Get non indexed version of data

data = pd.read_csv(url_stocks,parse_dates=[0])

data['Date'] = pd.to_datetime(data['Date'])

data['year'] = data['Date'].dt.year

data['month'] = data['Date'].dt.month

data.head()
amzn_data=data[data.Name == 'AMZN']

wmt_data=data[data.Name == 'WMT']
import seaborn as sns

variable = 'Open'

fig, ax = plt.subplots(figsize=(15, 6))

d=amzn_data

sns.lineplot(d['month'], d[variable], hue=d['year'])

ax.set_title('Seasonal plot of Open Price of AMZN', fontsize = 20, loc='center', fontdict=dict(weight='bold'))

ax.set_xlabel('Month', fontsize = 16, fontdict=dict(weight='bold'))

ax.set_ylabel('Open Price AMZN', fontsize = 16, fontdict=dict(weight='bold'))





fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))



sns.boxplot(d['year'], d[variable], ax=ax[0])

ax[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize = 20, loc='center', fontdict=dict(weight='bold'))

ax[0].set_xlabel('Year', fontsize = 16, fontdict=dict(weight='bold'))

ax[0].set_ylabel('Open Price of AMZN', fontsize = 16, fontdict=dict(weight='bold'))



sns.boxplot(d['month'], d[variable], ax=ax[1])

ax[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize = 20, loc='center', fontdict=dict(weight='bold'))

ax[1].set_xlabel('Month', fontsize = 16, fontdict=dict(weight='bold'))

ax[1].set_ylabel('Open price of AMZN', fontsize = 16, fontdict=dict(weight='bold'))

variable = 'Open'

fig, ax = plt.subplots(figsize=(15, 6))

d=wmt_data

sns.lineplot(d['month'], d[variable], hue=d['year'])

ax.set_title('Seasonal plot of Open Price of WMT', fontsize = 20, loc='center', fontdict=dict(weight='bold'))

ax.set_xlabel('Month', fontsize = 16, fontdict=dict(weight='bold'))

ax.set_ylabel('Open Price WMT', fontsize = 16, fontdict=dict(weight='bold'))





fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))



sns.boxplot(d['year'], d[variable], ax=ax[0])

ax[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize = 20, loc='center', fontdict=dict(weight='bold'))

ax[0].set_xlabel('Year', fontsize = 16, fontdict=dict(weight='bold'))

ax[0].set_ylabel('Open Price of WMT', fontsize = 16, fontdict=dict(weight='bold'))



sns.boxplot(d['month'], d[variable], ax=ax[1])

ax[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize = 20, loc='center', fontdict=dict(weight='bold'))

ax[1].set_xlabel('Month', fontsize = 16, fontdict=dict(weight='bold'))

ax[1].set_ylabel('Open price of WMT', fontsize = 16, fontdict=dict(weight='bold'))
#We try to predict the opening day prices

amzn_stock=amzn_df['Open']

wmt_stock=wmt_df['Open']
#Divide into testing and training sets

PERCENTAGE_TRAIN=0.8

train_size=int(PERCENTAGE_TRAIN*amzn_stock.shape[0])

print(train_size)



#We train the first PERCENTAGE_TRAIN% of our entries and predict the remaining

#Shifting by prices by one day

amzn_shifted=amzn_stock.shift(periods=1)

wmt_shifted=wmt_stock.shift(periods=1)

print(amzn_shifted.head())
amzn_train=amzn_stock[:train_size]

amzn_test=amzn_stock[train_size:]

wmt_train=wmt_stock[:train_size]

wmt_test=wmt_stock[train_size:]

test_size=amzn_test.size

print(amzn_test.size)
#RootMeanSqrError in baseline prediction

# Our predicted value at time t is just the known value at time t-1

from sklearn.metrics import mean_squared_error

amzn_naive_error=mean_squared_error(amzn_stock[train_size:],amzn_shifted[train_size:],squared=False)

wmt_naive_error=mean_squared_error(wmt_stock[train_size:],wmt_shifted[train_size:],squared=False)

print(amzn_naive_error,wmt_naive_error)
plt.plot(amzn_stock[train_size:],label="True")

plt.plot(amzn_shifted[train_size:],label="Predicted",color="red")

plt.title("AMZN")

plt.show()



plt.plot(wmt_stock[train_size:],label="True")

plt.plot(wmt_shifted[train_size:],label="Predicted",color="red")

plt.title("WMT")

plt.show()
k=25

amzn_mean=amzn_stock.rolling(window=k).mean()

wmt_mean=wmt_stock.rolling(window=k).mean()



amzn_stock.plot(label='Actual')

amzn_mean.plot(label='Predicted',color='red')

plt.title("AMZN Rolling Avgs")

plt.legend()

plt.show()



print("Error for k= ",k," AMZN is ", mean_squared_error(amzn_mean[k:],amzn_stock[k:],squared=False))







wmt_stock.plot(label='Actual')

wmt_mean.plot(label='Predicted',color='red')

plt.title("WMT Rolling Avgs")

plt.legend()

plt.show()



print("Error for k= ",k," WMT is ", mean_squared_error(wmt_mean[k:],wmt_stock[k:],squared=False))







k=100

amzn_mean=amzn_stock.rolling(window=k).mean()

wmt_mean=wmt_stock.rolling(window=k).mean()



amzn_stock.plot(label='Actual')

amzn_mean.plot(label='Predicted',color='red')

plt.title("AMZN Rolling Avgs")

plt.legend()

plt.show()



print("Error for k= ",k," AMZN is ", mean_squared_error(amzn_mean[k:],amzn_stock[k:],squared=False))







wmt_stock.plot(label='Actual')

wmt_mean.plot(label='Predicted',color='red')

plt.title("WMT Rolling Avgs")

plt.legend()

plt.show()



print("Error for k= ",k," WMT is ", mean_squared_error(wmt_mean[k:],wmt_stock[k:],squared=False))

# Rolling Medians
k=25

amzn_mean=amzn_stock.rolling(window=k).median()

wmt_mean=wmt_stock.rolling(window=k).median()



amzn_stock.plot(label='Actual')

amzn_mean.plot(label='Predicted',color='red')

plt.title("AMZN Rolling Avgs")

plt.legend()

plt.show()



print("Error for k= ",k," AMZN is ", mean_squared_error(amzn_mean[k:],amzn_stock[k:],squared=False))







wmt_stock.plot(label='Actual')

wmt_mean.plot(label='Predicted',color='red')

plt.title("WMT Rolling Avgs")

plt.legend()

plt.show()



print("Error for k= ",k," WMT is ", mean_squared_error(wmt_mean[k:],wmt_stock[k:],squared=False))







k=100

amzn_mean=amzn_stock.rolling(window=k).median()

wmt_mean=wmt_stock.rolling(window=k).median()



amzn_stock.plot(label='Actual')

amzn_mean.plot(label='Predicted',color='red')

plt.title("AMZN Rolling Avgs")

plt.legend()

plt.show()



print("Error for k= ",k," AMZN is ", mean_squared_error(amzn_mean[k:],amzn_stock[k:],squared=False))







wmt_stock.plot(label='Actual')

wmt_mean.plot(label='Predicted',color='red')

plt.title("WMT Rolling Avgs")

plt.legend()

plt.show()



print("Error for k= ",k," WMT is ", mean_squared_error(wmt_mean[k:],wmt_stock[k:],squared=False))

#Helper function to extract the needed data

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

    """

    Frame a time series as a supervised learning dataset.

    Arguments:

        data: Sequence of observations as a list or NumPy array.

        n_in: Number of lag observations as input (X).

        n_out: Number of observations as output (y).

        dropnan: Boolean whether or not to drop rows with NaN values.

    Returns:

        Pandas DataFrame of series framed for supervised learning.

    """

    n_vars = 1 if type(data) is list else data.shape[0]

    df = pd.DataFrame(data)

    cols, names = list(), list()

    # input sequence (t-n, ... t-1)

    for i in range(n_in, 0, -1):

        cols.append(df.shift(i))

        names += [(f"var_t-{i}") for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)

    for i in range(0, n_out):

        cols.append(df.shift(-i))

        if i == 0:

            names += [(f'var_t') for j in range(n_vars)]

        else:

            names += [(f'var_t+{i}') for j in range(n_vars)]

    # put it all together

    agg = pd.concat(cols, axis=1)

    agg.columns = names

    # drop rows with NaN values

    if dropnan:

        agg.dropna(inplace=True)

    return agg
from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(amzn_stock.values)
amzn_values = list(amzn_stock.values)

# We wish to do apply the rolling validation technique, hence the no of outputs will be 1

LAG_VARIABLES=1

amzn_supervised_data = series_to_supervised(amzn_values,LAG_VARIABLES,1)



print(amzn_supervised_data.shape)
#Split into train and test

TRAIN_PERCENTAGE=0.8

train_size=int(TRAIN_PERCENTAGE*amzn_supervised_data.shape[0])

print(train_size)

amzn_train_lin=amzn_supervised_data[:train_size]

amzn_test_lin=amzn_supervised_data[train_size:]

#print(amzn_test_lin)



train_price=amzn_train_lin['var_t']

amzn_train_data=amzn_train_lin.drop(columns='var_t')

#print(amzn_train_data)



test_price=amzn_test_lin['var_t']

#print(test_price)

amzn_test_data=amzn_test_lin.drop(columns='var_t')

print(amzn_test_data)
#Linear Model

from sklearn.linear_model import LinearRegression 

linear_model = LinearRegression()
print(amzn_train_data)

type(amzn_train_data)
history = amzn_train_data

#print(history)

y_pred = []

y_test = test_price.values

train_price=train_price.values

train_price=list(train_price)



for t in range(len(y_test)):

    model = linear_model

    model_fit = model.fit(history,train_price) #We make the model at every timestep

    x_val=[amzn_test_data.iloc[t,:]]

    yhat=model_fit.predict(x_val)

    

    history=history.append(x_val[0])

    y_pred.append(yhat)

    obs=y_test[t]

    train_price.append(obs)





error=mean_squared_error(y_pred,y_test,squared=False)

print("MSE is ",error)
plt.plot(y_test)

plt.plot(y_pred,color='red')

plt.show()
from statsmodels.graphics.tsaplots import plot_acf



#plot_acf helps finding q

plot_acf(amzn_train)

plt.show()
plt.plot(amzn_train)

plt.title("AMZN")

plt.show()

#amazon Arima p =1 q=0/1/2 d=1/2

from statsmodels.tsa.arima_model import ARIMA
#forecasting amzn stock

history=[x for x in amzn_train]

y_pred=[]

y_test=amzn_test.values

hyperparameters=(1,1,0)

for t in range(len(amzn_test)):

    model=ARIMA(history,order=hyperparameters)

    model_fit=model.fit(disp=0)

    output=model_fit.forecast()

    yhat=output[0]

    y_pred.append(yhat)

    obs=y_test[t]

    history.append(obs)

    print('predicted=%f, expected=%f' % (yhat, obs))



error=mean_squared_error(y_pred,y_test,squared=False)

print("MSE is ",error)
#forecasting amzn stock

history=[x for x in amzn_train]

y_pred=[]

y_test=amzn_test.values

hyperparameters=(1,1,1)

for t in range(len(amzn_test)):

    model=ARIMA(history,order=hyperparameters)

    model_fit=model.fit(disp=0)

    output=model_fit.forecast()

    yhat=output[0]

    y_pred.append(yhat)

    obs=y_test[t]

    history.append(obs)

    print('predicted=%f, expected=%f' % (yhat, obs))

error=mean_squared_error(y_pred,y_test,squared=False)

print("MSE for",hyperparameters, "is ",error)
#forecasting amzn stock

history=[x for x in amzn_train]

y_pred=[]

y_test=amzn_test.values

hyperparameters=(1,2,0)

for t in range(len(amzn_test)):

    model=ARIMA(history,order=hyperparameters)

    model_fit=model.fit(disp=0)

    output=model_fit.forecast()

    yhat=output[0]

    y_pred.append(yhat)

    obs=y_test[t]

    history.append(obs)

    print('predicted=%f, expected=%f' % (yhat, obs))

error=mean_squared_error(y_pred,y_test,squared=False)

print("MSE is ",error)