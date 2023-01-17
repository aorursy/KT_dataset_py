#importing all of the neccesary libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from statsmodels.tsa.arima_model import ARIMA

import pandas_datareader

import datetime



import pandas_datareader.data as web
#creating a start and an end dates:

start = datetime.datetime(2016,1,1)

end = datetime.datetime.today()
#JD.com

jd = web.DataReader('JD', 'yahoo', start, end)

jd.head()


jd['Open'].plot(label = 'JD.com')

plt.legend(loc = 'best');


jd['Volume'].plot(label = 'JD.com')

plt.legend();


jd['Total Traded'] = jd['Open']*jd['Volume']

jd['Total Traded'].plot(figsize = (16,8), label = 'JD.com')

plt.legend(loc = 'best');


jd['Avg'] = jd[['High', 'Low']].mean(axis=1)

jd['Total Traded New'] = jd['Avg']*jd['Volume']

jd['Total Traded New'].plot(figsize = (16,8), label = 'JD.com')

plt.legend(loc = 'best');




jd['MA50'] = jd['Open'].rolling(50).mean()

jd['MA200'] = jd['Open'].rolling(200).mean()
jd['Returns'] = jd['Close'].pct_change(1)
jd['Returns'].hist(bins = 50);


jd['Returns'].hist(bins = 100, figsize=(16,8), label = 'JD.com', alpha=0.4)

plt.legend();


jd['Returns'].plot(kind='kde', label = 'JD.com', figsize=(10,8))

plt.legend();
box_df = pd.concat([ jd['Returns']], axis = 1)

box_df.columns = [ 'JD Ret']

box_df.plot(kind='box', figsize = (8,10));
model = ARIMA(jd['Total Traded New'], order=(2,0,0)).fit()

y2 = model.predict(100)



y2.plot()