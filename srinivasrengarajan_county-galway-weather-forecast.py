# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns



df = pd.read_csv('/kaggle/input/irish-weather-hourly-data/hourly_irish_weather.csv').iloc[:,1:]



cols = df.columns.tolist()

cols[0] = 'date_stamp'

df.columns = cols



df.isna().sum()
df[['date','timestamp']] = df['date_stamp'].str.split(' ',expand=True)



df[['year','month','day']] = df['date'].str.split('-',expand=True)



df['hour'] = df['timestamp'].str[0:2]
df.head(5)
corr_matrx =df.corr()

corr_matrx
corr_matrx[(corr_matrx > 0.5) | (corr_matrx < - 0.5)]
galway_df = df[(df['county'] == 'Galway') ]

galway_df.count()
import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

%matplotlib inline

tmp_gpby =galway_df.groupby(['year','month']).agg({'temp':np.mean})



tmp_gpby[tmp_gpby == np.nan].sum()
rolling_index_diff_df = pd.DataFrame(pd.Series(galway_df[galway_df['temp'].isna()].index),columns=['index'])


check_list = galway_df['temp'].isna().index.tolist()



def impute():

    

    op = []

    

    for i in galway_df[galway_df['temp'].isna()].index:

        [year,month] = galway_df.loc[i,['year','month']].tolist()

        op.append(tmp_gpby.loc[(year,month)][0])

    return op



op = impute()



galway_df.loc[galway_df['temp'].isna(),'temp'] = op

galway_df.index = pd.DatetimeIndex(galway_df['date_stamp'])

#galway_df.drop(columns=['date_stamp','rolling_temp','rolling_sd'],inplace=True)
mean = galway_df['temp'].rolling(window = 24*30).mean()

sd =  galway_df['temp'].rolling(window = 24*30).std()



dummy = galway_df.copy()

dummy['rolling_temp'] = mean

dummy['rolling_sd'] =  sd



galway_df = dummy.copy()
galway_df

galway_daywise_df = galway_df.groupby(['date']).agg({'temp':np.mean})

galway_daywise_df.index = pd.DatetimeIndex(galway_daywise_df.index)

galway_df.drop(columns=['w','ww','sun','vis','clht','clamt'],inplace=True) #no captures for this variables [**GALWAY***]
import matplotlib.dates as mdates







ax = galway_daywise_df.plot(figsize=(20,10))



ax.xaxis.set_major_locator(mdates.YearLocator(1))

# set formatter

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))



ax
from statsmodels.tsa.stattools import adfuller

adfuller(galway_daywise_df['temp'])



#p-value is less than 0.05 (95% CI), so rejecting the null hypothesis, therefore at 95% confidence, we have enough evidence to

#support the claim that the data is stationary
#Checking for the relationship between the current temperature observations and the lagged observations - should indicate either strong 

#positive or strong negative correlation - for the data to be stationary

from pandas.plotting import lag_plot

lag_plot(galway_daywise_df['temp'])
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf



p = plot_acf(galway_daywise_df['temp'],lags=30)



p.set_size_inches(15, 8)
p = plot_pacf(galway_daywise_df['temp'],lags=30)



p.set_size_inches(15, 8)
from statsmodels.tsa.seasonal import seasonal_decompose



seasonal_decompose(galway_daywise_df['temp'],freq=30*12).plot();

galway_multivariate_df = galway_df.copy()

galway_multivariate_df.drop(columns=['station','county','longitude','latitude'],inplace=True)

galway_multivariate_df = galway_multivariate_df.iloc[:,0:14]

galway_multivariate_df.isna().sum()
multivariate_gpy = galway_multivariate_df.groupby(['year','month']).agg({'rain':'mean', 'temp':'mean', 'wetb':'mean', 'dewpt':'mean', 'vappr':'mean', 'rhum':'mean', 'msl':'mean', 'wdsp':'mean', 'wddir':'mean'})
galway_multivariate_df = galway_multivariate_df.interpolate(method='time')
galway_multivariate_df.isna().sum()
galway_multivariate_daywise_df = galway_multivariate_df.groupby(['date']).agg({'rain':'mean', 'temp':'mean', 'wetb':'mean', 'dewpt':'mean', 'vappr':'mean', 'rhum':'mean', 'msl':'mean', 'wdsp':'mean', 'wddir':'mean'})

galway_multivariate_daywise_df.index = pd.DatetimeIndex(galway_multivariate_daywise_df.index)
train = galway_multivariate_daywise_df.iloc[:4525,:]

test = galway_multivariate_daywise_df.iloc[4525:,:]



train.index  = pd.DatetimeIndex(train.index).to_period('D')
from statsmodels.tsa.statespace.sarimax import SARIMAX



model = SARIMAX(train['temp'],exog=train[['wdsp', 'vappr', 'wetb', 'dewpt', 'rhum', 'msl', 'rain', 'wddir']],order=(3, 0, 2), seasonal_order=(0,0,0,12))



model_fit = model.fit()



yhat = model_fit.predict(4525, 4525+364,exog=test[['wdsp', 'vappr', 'wetb', 'dewpt', 'rhum', 'msl', 'rain', 'wddir']])

print(yhat)

prediction_df = yhat.to_frame()

prediction_df['actual'] = test['temp'].values



prediction_df.rename(columns={0:'pred'},inplace=True)

prediction_df.index =  test.index





prediction_df.plot(figsize=(20,12))
model_fit.summary()

from sklearn.metrics import mean_squared_error,mean_absolute_error



print(mean_squared_error(prediction_df['actual'],prediction_df['pred']))

print(mean_absolute_error(prediction_df['actual'],prediction_df['pred']))