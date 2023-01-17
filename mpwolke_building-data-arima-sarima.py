# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from datetime import datetime

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from itertools import combinations



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from datetime import datetime

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.stattools import acf, pacf

from statsmodels.tsa.arima_model import ARIMA as ARIMA

import statsmodels.api as sm

import statsmodels.tsa.api as smt

pd.options.display.float_format = '{:.2f}'.format
df = pd.read_csv('../input/buildingdatagenomeproject2/solar_cleaned.csv')

df.head()
df.isnull().sum()
df.info()
# Lets first handle numerical features with nan value

numerical_nan = [feature for feature in df.columns if df[feature].isna().sum()>1 and df[feature].dtypes!='O']

numerical_nan
df[numerical_nan].isna().sum()
## Replacing the numerical Missing Values



for feature in numerical_nan:

    ## We will replace by using median since there are outliers

    median_value=df[feature].median()

    

    df[feature].fillna(median_value,inplace=True)

    

df[numerical_nan].isnull().sum()
cols_to_drop=['Bobcat_office_Justine']

df=df.drop(cols_to_drop,axis=1)

df.columns
def test_stationarity(timeseries):

    #Determing rolling statistics

    MA = timeseries.rolling(window=12).mean()

    MSTD = timeseries.rolling(window=12).std()



    #Plot rolling statistics:

    plt.figure(figsize=(15,5))

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(MA, color='red', label='Rolling Mean')

    std = plt.plot(MSTD, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)



    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)
def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):

    if not isinstance(y, pd.Series):

        y = pd.Series(y)

        

    with plt.style.context(style):    

        fig = plt.figure(figsize=figsize)

        layout = (2, 2)

        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)

        acf_ax = plt.subplot2grid(layout, (1, 0))

        pacf_ax = plt.subplot2grid(layout, (1, 1))

        

        y.plot(ax=ts_ax)

        p_value = sm.tsa.stattools.adfuller(y)[1]

        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))

        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)

        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)

        plt.tight_layout()
test_stationarity(df['Bobcat_education_Dylan'])
dec = sm.tsa.seasonal_decompose(df['Bobcat_education_Dylan'],period = 12).plot()

plt.show()
sns.distplot(df['Bobcat_education_Dylan'])
#log_df = np.log(df)

#log_df.head()
#test_stationarity(log_df['Bobcat_education_Dylan'])
#sns.distplot(log_df['Bobcat_education_Dylan'])
df_diff = df['Bobcat_education_Dylan'].diff()

df_diff = df_diff.dropna()

dec = sm.tsa.seasonal_decompose(df_diff,period = 12).plot()

plt.show()
test_stationarity(df_diff)
#log_df_diff = log_df['Bobcat_education_Dylan'].diff()

#log_df_diff = log_df_diff.dropna()

#dec = sm.tsa.seasonal_decompose(log_df_diff,period = 12)

#dec.plot()

#plt.show()
#test_stationarity(log_df_diff)
tsplot(df_diff)
model = ARIMA(df['Bobcat_education_Dylan'],order = (2,1,2))

model_fit = model.fit()

print(model_fit.summary())
df['FORECAST'] = model_fit.predict(start = 120,end = 144,dynamic = True)

df[['Bobcat_education_Dylan','FORECAST']].plot(figsize = (10,6))
exp = [df.iloc[i,0] for i in range(120,len(df))]

pred = [df.iloc[i,1] for i in range(120,len(df))]

df = df.drop(columns = 'FORECAST')

print(mean_absolute_error(exp,pred))
df_diff_seas = df_diff.diff(12)

df_diff_seas = df_diff_seas.dropna()

dec = sm.tsa.seasonal_decompose(df_diff_seas,period = 12)

dec.plot()

plt.show()
tsplot(df_diff_seas)
model = sm.tsa.statespace.SARIMAX(df['Bobcat_education_Dylan'],order = (2,1,2),seasonal_order = (1,1,2,12))

results = model.fit()

print(results.summary())
df['FORECAST'] = results.predict(start = 120,end = 144,dynamic = True)

df[['Bobcat_education_Dylan','FORECAST']].plot(figsize = (12,8))
exp = [df.iloc[i,0] for i in range(120,len(df))]

pred = [df.iloc[i,1] for i in range(120,len(df))]

data = df.drop(columns = 'FORECAST')

print(mean_absolute_error(exp,pred))
from pandas.tseries.offsets import DateOffset

future_dates = [df.index[-1] + DateOffset(months = x)for x in range(0,25)]

df = pd.DataFrame(index = future_dates[1:],columns = data.columns)
forecast = pd.concat([df,df])

forecast['FORECAST'] = results.predict(start = 144,end = 168,dynamic = True)

forecast[['Bobcat_education_Dylan','FORECAST']].plot(figsize = (12,8))