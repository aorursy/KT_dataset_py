# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv")
df.head, df.shape
df["Month"] = pd.to_datetime(df[ "Month" ]) #convert column to datetime

df.set_index("Month" , inplace= True)
plt.figure(figsize=(15 , 10))

plt.plot(df.index, df.Passengers, '--' , marker= '*')

plt.grid()

plt.xlabel('Year')

plt.ylabel('Passengers') 
#check for missing values

df.isnull().values.any()
#There are no missing values in our dataset however, in bid to explain how we handle

#missing values, we will make a copy of our dataset and delete some values at random.



df_copy = df.copy()



rows = df_copy.sample(frac= 0.1, random_state= 0)



rows['Passengers'] = np.nan

df_copy.loc[rows.index, 'Passengers'] = rows['Passengers']
df_copy.isnull().sum()
#There are now 14 missing values in the dataset 

#Filling missing data by imputation - Forward fill

df_copy_ffill = df_copy.fillna(method= 'ffill')



df_copy_ffill.isnull().sum() 
#Filling missing data by imputation - Backward fill

df_copy_bfill = df_copy.fillna(method= 'bfill')

df_copy_bfill.isnull().sum() 
#Filling missing data by interpolation

df_copy_LIF = df_copy.interpolate(method= 'linear' , limit_direction= 'forward')

df_copy_LIF.isnull().sum()



df_copy_LIB = df_copy.interpolate(method= 'linear' , limit_direction= 'backward')

df_copy_LIB.isnull().sum()
#Downsample to quarterly data points

df_quarterly = df.resample('3M').mean()



#Upsample to daily data points

df_daily = df.resample('D').mean() 
plt.figure(figsize=(15 , 10))

plt.plot(df_quarterly.index, df_quarterly.Passengers, '--' , marker= '*')

plt.grid()

plt.xlabel('Year')

plt.ylabel('Passengers') 
plt.figure(figsize=(15 , 10))

plt.plot(df_daily.index, df_daily.Passengers, '--' , marker= '*')

plt.grid()

plt.xlabel('Year')

plt.ylabel('Passengers') 
df_MA = df.copy()

MA = df_MA['Passengers'].rolling(12).mean()
plt.figure(figsize=(15 , 10))

plt.plot(df.index, df.Passengers, '--' , marker= '*')

plt.plot(MA, color="red")

plt.grid()

plt.xlabel('Year')

plt.ylabel('Passengers') 
import statsmodels.api as sm

from pylab import rcParams



rcParams['figure.figsize'] = 15, 8

decompose_series = sm.tsa.seasonal_decompose(df['Passengers'], model= 'additive')



decompose_series.plot()

plt.show()
#The decomposed time series show an obvious increasing trend and seasonality variations.

# Recall that we have initially plotted the moving average on the last 12 months which showed

# that it varies with time. This suggests that the data is not stationary. We will now perform

# an ADF test to confirm this speculation

from statsmodels.tsa.stattools import adfuller



adf_result = adfuller(df['Passengers'])

print(f'ADF Statistic: {adf_result[0]}')

print(f'p-value: {adf_result[1]}')

print(f'No. of lags used: {adf_result[2]}')

print(f'No. of observations used : {adf_result[3]}')

print('Critical Values:')





for k, v in adf_result[4].items():

    print(f'{k} : {v}')
#From the results obtained, the p-value is greater than the critical value at a 5%

#significance level and, the ADF statistic is greater that any of the critical values

# obtain. #This confirms that the series is indeed non-stationary. 
#Convert time series to stationary by removing trend and seasonality

#Transformation and Differencing



df_log = np.log(df)

df_diff = df_log.diff(periods = 1)



plt.plot(df_diff.index, df_diff.Passengers, '-')

plt.plot(df_diff.rolling(12).mean(), color= 'red') 
df_diff.fillna(0, inplace=True)
from statsmodels.tsa.stattools import acf, pacf

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



#ACF

plot_acf(df_diff, lags = range( 0 , 20 ))

plt.show()



#PACF

plot_pacf(df_diff, lags = range( 0 , 20 ))

plt.show() 
#The shaded regions in the plots are the confidence intervals. The lags where the PACF #and

# ACF charts cross this region are the values for p and q respectively. In both plots, #p=q=1.

#The shaded regions in the plots are the confidence intervals. The lags where the PACF and

#ACF charts cross this region are the values for p and q respectively.

#In the ACF plot, there is one lag that crosses the significance level hence, q=1. Similarly

#in the PACF plot, p=2 
#AR, MA and ARIMA

from statsmodels.tsa.arima_model import ARIMA



#(p,d,q)

AR_model = ARIMA(df_diff, order=( 2 , 0 , 0 ))

AR_model_results = AR_model.fit()

plt.plot(df_diff)

plt.plot(AR_model_results.fittedvalues, color= 'red' ) 
MA_model = ARIMA(df_diff, order=( 0 , 0 , 2 ))

MA_model_results = MA_model.fit()

plt.plot(df_diff)

plt.plot(MA_model_results.fittedvalues, color= 'red' ) 
ARIMA_model = ARIMA(df_diff, order=( 2 , 0 , 1 )) 

ARIMA_results = ARIMA_model.fit()

plt.plot(df_diff)

plt.plot(ARIMA_results.fittedvalues, color= 'red' ) 