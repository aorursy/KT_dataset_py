# Importing libraries for Loading Data
import numpy as np
import pandas as pd
# Importing libraries for data visualization 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
from statsmodels.tsa.arima_model import ARIMA 
# Loadig the given dataset
df=pd.read_csv( "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv" )
df.head()
# Convert column to datetime
df["Month"]=pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df
# Plotting a visualisation line graph
plt.figure(figsize=(10,6))
plt.plot(df.index, df.Passengers, '--' , marker= '*' , )
plt.grid()
plt.xlabel("Year")
plt.ylabel("Passenger")
# Check for missing values
df.isna().values.any()
# Missing values, we will make a copy of our dataset and delete some values at random.
# Now we xcan elaboratel check how many missing passenger data ......
df_copy = df.copy()
rows = df_copy.sample(frac= 0.1 , random_state= 0 )
rows[ 'Passengers' ] = np.nan
df_copy.loc[rows.index, 'Passengers' ] = rows[ 'Passengers' ]
df_copy.isnull().sum() 
# Filling missing data by imputation - Forward fill
df_copy_ffill = df_copy.fillna(method= 'ffill' )
df_copy_ffill.isnull().sum() 
# Filling missing data by imputation - Backward fill
df_copy_bfill = df_copy.fillna(method= 'bfill')
df_copy_bfill.isnull().sum()
# Filling missing data by interpolation

# Filling Linear Interpolation Forward (LIF)
df_copy_LIF = df_copy.interpolate(method='linear', limit_direction='forward')
df_copy_LIF.isnull().sum()

# Filling Linear Interpolation Barkward (LIB)
df_copy_LIB = df_copy.interpolate(method='linear', limit_direction='backward')
df_copy_LIB.isnull().sum()
# Downsampling and Upsampling

#Downsample to quarterly data points
df_quarterly = df.resample( '3M' ).mean()

#Upsample to daily data points
df_daily = df.resample( 'D' ).mean() 
df_MA = df.copy() 
MA = df_MA[ 'Passengers' ].rolling( 12 ).mean() 
# Time Series Specific Exploratory Methods 
rcParams[ 'figure.figsize' ] = 15 , 8
decompose_series = sm.tsa.seasonal_decompose(df[ 'Passengers' ], model= 'additive' )
decompose_series.plot()
plt.show()
adf_result = adfuller(df[ 'Passengers'])
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[ 1 ]}')
print(f'No. of lags used: {adf_result[ 2 ]}')
print(f'No. of observations used : {adf_result[ 3 ]}')
print('Critical Values:')
for k, v in adf_result[ 4 ].items():
 print( f' {k} : {v} ' ) 

# From the results obtained, the p-value is greater than the critical value at a 5%
#significance level and, the ADF statistic is greater that any of the critical values obtain. 
#This confirms that the series is indeed non-stationary
#Convert time series to stationary by removing trend and seasonality
#Transformation and Differencing
df_log = np.log(df)
df_diff = df_log.diff(periods= 1 )
plt.plot(df_diff.index, df_diff.Passengers, '-' )
plt.plot(df_diff.rolling( 12 ).mean(), color= 'red' ) 
df_diff = df_diff.fillna(method='bfill')
# Time Series Forecasting Using Stochastic Models
#ACF
plot_acf(df_diff, lags = range( 0 , 20 ))
plt.show()
#PACF
plot_pacf(df_diff, lags = range( 0 , 20 ))
plt.show() 
#(p,d,q)
AR_model = ARIMA(df_diff, order=( 2 , 0 , 0 ))
AR_model_results = AR_model.fit()
plt.plot(df_diff)
plt.plot(AR_model_results.fittedvalues, color= 'red') 
MA_model = ARIMA(df_diff, order=( 0 , 0 , 2 ))
MA_model_results = MA_model.fit()
plt.plot(df_diff)
plt.plot(MA_model_results.fittedvalues, color= 'red') 
ARIMA_model = ARIMA(df_diff, order=( 2 , 0 , 1 )) 
ARIMA_results = ARIMA_model.fit()
plt.plot(df_diff)
plt.plot(ARIMA_results.fittedvalues, color= 'red') 
