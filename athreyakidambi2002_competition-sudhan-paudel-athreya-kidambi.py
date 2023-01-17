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
df_pgen1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df_psense1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
df_pgen1['DATE_TIME'] = pd.to_datetime(df_pgen1['DATE_TIME'],format = '%d-%m-%Y %H:%M')
df_psense1['DATE_TIME'] = pd.to_datetime(df_psense1['DATE_TIME'],format = '%Y-%m-%d %H:%M')

df_pgen1['DATE'] = df_pgen1['DATE_TIME'].dt.date
df_pgen1['TIME'] = df_pgen1['DATE_TIME'].dt.time

df_psense1['DATE'] = df_psense1['DATE_TIME'].dt.date
df_psense1['TIME'] = df_psense1['DATE_TIME'].dt.time



df_pgen1['DATE'] = pd.to_datetime(df_pgen1['DATE'],format = '%Y-%m-%d')
df_psense1['DATE'] = pd.to_datetime(df_psense1['DATE'],format = '%Y-%m-%d')
df_pgen1['HOUR'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.hour
df_pgen1['MINUTES'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.minute

df_psense1['HOUR'] = pd.to_datetime(df_psense1['TIME'],format='%H:%M:%S').dt.hour
df_psense1['MINUTES'] = pd.to_datetime(df_psense1['TIME'],format='%H:%M:%S').dt.minute
df_pgen1.head(50)
df_psense1.head()
import matplotlib.pyplot as plt
import seaborn as sns
#for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_psense1.DATE_TIME,
        df_psense1.AMBIENT_TEMPERATURE.rolling(window=20).mean(),
        label='Ambient'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Ambient Temperature')
plt.xlabel('Date and Time')
plt.ylabel('Temperature')
plt.show()
print ("minimum="+ str(df_psense1.AMBIENT_TEMPERATURE.min()) )
print ("maximum="+ str(df_psense1.AMBIENT_TEMPERATURE.max()) )
print ("mean="+ str(df_psense1.AMBIENT_TEMPERATURE.mean()) )
#for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_psense1.DATE_TIME,
        df_psense1.MODULE_TEMPERATURE.rolling(window=20).mean(),
        label='Ambient'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Module Temperature over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('Temperature')
plt.show()

print ("minimum="+ str(df_psense1.MODULE_TEMPERATURE.min()) )
print ("maximum="+ str(df_psense1.MODULE_TEMPERATURE.max()) )
print ("mean="+ str(df_psense1.MODULE_TEMPERATURE.mean()) )
#FOR plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_psense1.DATE_TIME,
        df_psense1.AMBIENT_TEMPERATURE.rolling(window=20).mean(),
        label='Ambient'
       )

ax.plot(df_psense1.DATE_TIME,
        df_psense1.MODULE_TEMPERATURE.rolling(window=20).mean(),
        label='Module'
       )

ax.plot(df_psense1.DATE_TIME,
        (df_psense1.MODULE_TEMPERATURE-df_psense1.AMBIENT_TEMPERATURE).rolling(window=20).mean(),
        label='Difference'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Ambient Temperature and Module Temperature over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('Temperature')
plt.show()

# for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen1.DATE_TIME,
        df_pgen1.AC_POWER.rolling(window=500).mean(),
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC_POWER over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('POWER')
plt.show()

print ("minimum="+ str(df_pgen1.AC_POWER.min()) )
print ("maximum="+ str(df_pgen1.AC_POWER.max()) )
print ("mean="+ str(df_pgen1.AC_POWER.mean()) )
#for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen1.DATE_TIME,
        (df_pgen1.DC_POWER/10).rolling(window=500).mean(),
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC_POWER over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('POWER')
plt.show()

print ("minimum="+ str((df_pgen1.DC_POWER/10).min()) )
print ("maximum="+ str((df_pgen1.DC_POWER/10).max()) )
print ("mean="+ str((df_pgen1.DC_POWER/10).mean()) )
#for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen1.DATE_TIME,
        df_pgen1.AC_POWER.rolling(window=500).mean(),
        label='AC'
       )

ax.plot(df_pgen1.DATE_TIME,
       (df_pgen1.DC_POWER/10).rolling(window=500).mean(),
        label='DC'
       )

ax.plot(df_pgen1.DATE_TIME,
       ((df_pgen1.DC_POWER/10)-df_pgen1.AC_POWER).rolling(window=500).mean(),
        label='Difference'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC POWER and DC POWER over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('kW')
plt.show()

#for plant 1

df_data = df_psense1[df_psense1['DATE']=='2020-05-23T']

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_data.DATE_TIME,
        df_data.MODULE_TEMPERATURE,
        label="MODULE"
       )
ax.plot(df_data.DATE_TIME,
        df_data.AMBIENT_TEMPERATURE,
        label="AMBIENT"
       )

ax.plot(df_data.DATE_TIME,
      ((df_data.MODULE_TEMPERATURE)-(df_data.AMBIENT_TEMPERATURE)),
        label="Difference"
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Variance of Module Temperature with Ambient temperature on 23 May')
plt.xlabel('Date-Time')
plt.ylabel(' Temperature')
plt.show()
#for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_psense1.DATE_TIME,
        df_psense1.IRRADIATION.rolling(window=40).mean(),
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('IRRADIATION over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('IRRADIATION')
plt.show()

print ("minimum="+ str((df_psense1.IRRADIATION).min()) )
print ("maximum="+ str((df_psense1.IRRADIATION).max()) )
print ("mean="+ str((df_psense1.IRRADIATION).mean()) )
# for plant 1

df_data = df_psense1[df_psense1['DATE']=='2020-05-23T']

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_data.DATE_TIME,
        df_data.IRRADIATION.rolling(window=1).mean(),
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Variance of Irradiation on 23 May')
plt.xlabel('Date-Time')
plt.ylabel(' Temperature')
plt.show()
daily_yield=df_pgen1.groupby("DATE").agg(TODAY_YIELD=("DAILY_YIELD",max),
                                           DATE=("DATE",max)
                                        )
daily_yield
# for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(daily_yield.DATE,
        daily_yield.TODAY_YIELD
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DAILY YIELD over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('POWER')
plt.show()
print("maximum=" +str( daily_yield['TODAY_YIELD'].max()))
print("minimum=" +str( daily_yield['TODAY_YIELD'].min()))
print("mean=" +str( daily_yield['TODAY_YIELD'].mean()))
Inverters_performance=df_pgen1.groupby("SOURCE_KEY").agg(LIFETIME_YIELD=("TOTAL_YIELD",max),
                                           SOURCE_KEY=("SOURCE_KEY",max)
                                        )
Inverters_performance
# for plant 1

sns.barplot(x=Inverters_performance["SOURCE_KEY"], y=Inverters_performance["LIFETIME_YIELD"])

print("maximum=" +str(Inverters_performance['LIFETIME_YIELD'].max()))
print("minimum=" +str(Inverters_performance['LIFETIME_YIELD'].min()))
print("mean=" +str( Inverters_performance['LIFETIME_YIELD'].mean()))
#for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(df_psense1.AMBIENT_TEMPERATURE,
        df_psense1.MODULE_TEMPERATURE.rolling(window=5).mean(),
         marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
         )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Module Temperature varying with Ambient Temperature')
plt.xlabel('Ambient Temperature')
plt.ylabel('Module Temperature')
plt.show()
#for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))


df_data = df_psense1[df_psense1['DATE']=='2020-05-15']

ax.plot(df_data.AMBIENT_TEMPERATURE,
        df_data.MODULE_TEMPERATURE,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Module Temperature varies with Ambient Temperature')
plt.xlabel('Ambient Temperature')
plt.ylabel('Module Temperature')
plt.show()
#for plant1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_psense1['IRRADIATION'],
        df_psense1['MODULE_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='module temperature')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Irradiation vs. Module Tempreture')
plt.xlabel('Irradiation')
plt.ylabel('Module Tempreture')
plt.show()
#for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))


df_data = df_psense1[df_psense1['DATE']=='2020-05-23']

ax.plot(df_data.IRRADIATION,
        df_data.MODULE_TEMPERATURE,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title(' Module Temperature varying with Ambient Temperature on 23rd May')
plt.xlabel('IRRADIATION')
plt.ylabel('Module Temperature')
plt.show()
#for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_psense1['IRRADIATION'],
        df_psense1['AMBIENT_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Ambient temperature')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Irradiation vs. Module Temperature')
plt.xlabel('Irradiation')
plt.ylabel('Ambient Temperture')
plt.show()
#for plant 1


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen1.DC_POWER/10,
        df_pgen1.AC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='AC POWER'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How AC Power varies with DC Power')
plt.xlabel('DC Power')
plt.ylabel('AC Power')
plt.show()
comparision=df_pgen1.groupby("DATE").agg(DAILY_YIELD=("DAILY_YIELD",max),
                                         DC_POWER=("DC_POWER",sum),
                                         AC_POWER=("AC_POWER",sum),
                                         DATE=("DATE",max)
                                         )
comparision
# for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))



ax.plot(comparision.DC_POWER,
        comparision.DAILY_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC Power vs Daily Yield')
plt.xlabel('DC power')
plt.ylabel('daily yield')
plt.show()
# for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))



ax.plot(comparision.AC_POWER,
        comparision.DAILY_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC Power vs Daily Yield')
plt.xlabel('AC power')
plt.ylabel('daily yield')
plt.show()
plt.show()
# for plant 1

dates = comparision['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = comparision[comparision['DATE']==date]

    ax.plot(df_data.AC_POWER,
            df_data.DAILY_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC POWER and DAILY YIELD')
plt.xlabel('AC POWER')
plt.ylabel('Daily_Yield')
plt.show()
# for plant 1

dates = comparision['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = comparision[comparision['DATE']==date]

    ax.plot(df_data.DC_POWER,
            df_data.DAILY_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC POWER and DAILY YIELD')
plt.xlabel('DC POWER')
plt.ylabel('Daily_Yield')
plt.show()
result_outer1 = pd.merge(df_pgen1,df_psense1,on='DATE_TIME',how='outer')
# for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer1.IRRADIATION,
        result_outer1.DC_POWER/10,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DC POWER')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC Power vs. Irradiation')
plt.xlabel('Irradiation')
plt.ylabel('DC Power')
plt.show()
# for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer1.IRRADIATION,
        result_outer1.AC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='AC POWER')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC Power vs. Irradiation')
plt.xlabel('Irradiation')
plt.ylabel('AC Power')
plt.show()
# for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer1.MODULE_TEMPERATURE,
        result_outer1.DC_POWER/10,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DC POWER')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC Power vs. Module Temperature')
plt.xlabel('Temperature')
plt.ylabel('DC Power')
plt.show()
# for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer1.MODULE_TEMPERATURE,
        result_outer1.AC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DC POWER')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC Power vs. Temperature')
plt.xlabel('Temperature')
plt.ylabel('AC Power')
plt.show()
# for plant 1

dates = result_outer1['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer1[(result_outer1['DATE_x']==date)]

    ax.plot(data.MODULE_TEMPERATURE,
            data.DC_POWER,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC Power vs. Module Temperature')
plt.xlabel('Module Temperature')
plt.ylabel('DC Power')
plt.show()
# for plant 1

dates = result_outer1['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))



data = result_outer1[(result_outer1['DATE_x']=='2020-05-23')]

ax.plot(data.MODULE_TEMPERATURE,
        data.DC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC Power vs. Module Temperature')
plt.xlabel('Module Temperature')
plt.ylabel('DC Power')
plt.show()

#TIME SERIES FORECASTING USING FBPROPHET AND SARIMAX
#Import Libraries
import pandas as pd
import numpy as np
import seaborn as sb

import matplotlib.pyplot as plt
%matplotlib inline

from scipy.stats import norm, skew #for some statistics
from scipy import stats #qqplot
import statsmodels.api as sm #for decomposing the trends, seasonality etc.

from statsmodels.tsa.statespace.sarimax import SARIMAX 
df_dyield = df_pgen1.groupby('DATE')['DAILY_YIELD'].max().reset_index()
#This gives us the daily yield for the plant.
df_dyield
#Index the date
df_dyield = df_dyield.set_index('DATE')
df_dyield.index #Lets check the index
#We are using the one day as the timestamp so that we can take all 34 days of data.
y = df_dyield['DAILY_YIELD'].resample('D').mean()
y
y.plot(figsize=(12,5))
plt.show()

#Plotting daily yield with time to have a general idea of how daily yield varies over time.
#The best part about time series data and decomposition is that you can break down the data into the following:
#Time Series Decomposition.

from pylab import rcParams
import statsmodels.api as sm
rcParams['figure.figsize'] = 16, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()
#GRID SEARCH for Param Tuning.
#Sample params for seasonal arima. (SARIMAX).

#For each combination of parameters, we fit a new seasonal ARIMA model with the SARIMAX() function 

import itertools
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
#Get the best params for the data. Choose the lowest AIC.

# The Akaike information criterion (AIC) is an estimator of the relative quality of statistical models for a 
# given set of data. 
# AIC measures how well a model fits the data while taking into account the overall complexity of the model.
# Large AIC: Model fits very well using a lot of features.
# Small AIC: Model fits similar fit but using lesser features. 
# Hence LOWER THE AIC, the better it is.

#The code tests the given params using sarimax and outputs the AIC scores.

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
            
            
#Fit the model with the best parameters.
#ARIMA(1, 1, 1)x(1, 1, 1, 12)12 - AIC:


#The above output suggests that ARIMA(1, 1, 1)x(1, 1, 1, 12) yields the lowest AIC value: 119.76
#Therefore we should consider this to be optimal option.

from statsmodels.tsa.statespace.sarimax import SARIMAX
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1,0, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
#Plotting the diagnostics.

#The plot_diagnostics object allows us to quickly generate model diagnostics and investigate for any unusual behavior.
results.plot_diagnostics(figsize=(16, 8))
plt.show()

#What to look for?
#1. Residuals SHOULD be Normally Distributed ; Check
#Top Right: The (orange colored) KDE line should be closely matched with green colored N(0,1) line. This is the standard notation
#for normal distribution with mean 0 and sd 1.
#Bottom Left: The qq plot shows the ordered distribution of residuals (blue dots) follows the linear trend of the samples 
#taken from a standard normal distribution with N(0, 1). 

#2. #Residuals are not correlated; Check
#Top Left: The standard residuals don’t display any obvious seasonality and appear to be white noise. 
#Bottom Right: The autocorrelation (i.e. correlogram) plot on the bottom right, which shows that the time series residuals have 
#low correlation with its own lagged versions.

#Lets get the predictions and confidence interval for those predictions.
#Get the predictions. The forecasts start from the 1st of Jan 2017 but the previous line shows how it fits to the data.
pred = results.get_prediction(start=pd.to_datetime('2020-5-29'), dynamic=False) #false is when using the entire history.
#Confidence interval.
pred_ci = pred.conf_int()

#Plotting real and forecasted values.
ax = y['2020':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='blue', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Order_Demand')
plt.legend()
plt.show()

#Takeaway: The forecats seems to be fitting well to the data. The Blue/purple thicker plot shows the confidence level in the forecasts. 
#Getting the mean squared error (average error of forecasts).
y_forecasted = pred.predicted_mean
y_truth = y['2020-6-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('MSE {}'.format(round(mse, 2)))

#Smaller the better.
# root mean squared error
print('RMSE: {}'.format(round(np.sqrt(mse), 2)))
#Mean absolute percentage error

mape=(((y_forecasted - y_truth) / y_truth).mean())*100
print('MAPE: {}'.format(round(mape, 2)))
#The time can be changed using steps.
pred_uc = results.get_forecast(steps=25)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 8))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Order_Demand')
plt.legend()
plt.show()

#Far out values are naturally more prone to variance. The grey area is the confidence we have in the predictions.
df_dcpower = df_pgen1.groupby('DATE')['DC_POWER'].sum().reset_index()
#This gives us the total dc power generated on each day.
#Index the date
df_dcpower = df_dcpower.set_index('DATE')
df_dcpower.index #Lets check the index
# we are using the each day as the timestamp.
y = df_dcpower['DC_POWER'].resample('D').mean()
y.plot(figsize=(12,5))
plt.show()

# plotting DC_POWER vs DATE to have a general idea on how the data varies.
#The best part about time series data and decomposition is that you can break down the data into the following:
#Time Series Decomposition. 
from pylab import rcParams
import statsmodels.api as sm
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
            
from statsmodels.tsa.statespace.sarimax import SARIMAX
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(0 ,1 ,1),
                                seasonal_order=(1,0 ,1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

#Plotting the diagnostics.

#The plot_diagnostics object allows us to quickly generate model diagnostics and investigate for any unusual behavior.
results.plot_diagnostics(figsize=(16, 8))
plt.show()

#What to look for?
#1. Residuals SHOULD be Normally Distributed ; Check
#Top Right: The (orange colored) KDE line should be closely matched with green colored N(0,1) line. This is the standard notation
#for normal distribution with mean 0 and sd 1.
#Bottom Left: The qq plot shows the ordered distribution of residuals (blue dots) follows the linear trend of the samples 
#taken from a standard normal distribution with N(0, 1). 

#2. #Residuals are not correlated; Check
#Top Left: The standard residuals don’t display any obvious seasonality and appear to be white noise. 
#Bottom Right: The autocorrelation (i.e. correlogram) plot on the bottom right, which shows that the time series residuals have 
#low correlation with its own lagged versions.
#Lets get the predictions and confidence interval for those predictions.
#Get the predictions. The forecasts start from the 1st of Jan 2017 but the previous line shows how it fits to the data.
pred = results.get_prediction(start=pd.to_datetime('2020-5-29'), dynamic=False) #false is when using the entire history.
#Confidence interval.
pred_ci = pred.conf_int()
#Plotting real and forecasted values.
ax = y['2020':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='blue', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Order_Demand')
plt.legend()
plt.show()

#Takeaway: The forecats seems to be fitting well to the data. The Blue/purple thicker plot shows the confidence level in the forecasts. 


#Getting the mean squared error (average error of forecasts).
y_forecasted = pred.predicted_mean
y_truth = y['2020-6-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('MSE {}'.format(round(mse, 2)))

#Smaller the better.

print('RMSE: {}'.format(round(np.sqrt(mse), 2)))

mape=(((y_forecasted - y_truth) / y_truth).mean())*100
print('MAPE: {}'.format(round(mape, 2)))
#The time can be changed using steps.
pred_uc = results.get_forecast(steps=25)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 8))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Order_Demand')
plt.legend()
plt.show()

#Far out values are naturally more prone to variance. The grey area is the confidence we have in the predictions.
df_amb = df_psense1.groupby('DATE_TIME')['AMBIENT_TEMPERATURE'].max().reset_index()
#This gives us the total orders placed on each day.
#Index the date
df_amb =df_amb.set_index('DATE_TIME')
df_amb.index #Lets check the index
#Averages daily sales value for the month, and we are using the start of each month as the timestamp.
y = df_amb['AMBIENT_TEMPERATURE'].resample('15T').mean()

y
y.plot(figsize=(12,5))
plt.show()

#Get the best params for the data. Choose the lowest AIC.

# The Akaike information criterion (AIC) is an estimator of the relative quality of statistical models for a 
# given set of data. 
# AIC measures how well a model fits the data while taking into account the overall complexity of the model.
# Large AIC: Model fits very well using a lot of features.
# Small AIC: Model fits similar fit but using lesser features. 
# Hence LOWER THE AIC, the better it is.

#The code tests the given params using sarimax and outputs the AIC scores.

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
            
            
            
            
#Fit the model with the best params.
#ARIMA(1, 1, 1)x(1, 1, 1, 12)12 - AIC:


#The above output suggests that ARIMA(1, 1, 1)x(1, 1, 1, 12) yields the lowest AIC value: 119.76
#Therefore we should consider this to be optimal option.

from statsmodels.tsa.statespace.sarimax import SARIMAX
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(0,0, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

#Plotting the diagnostics.

#The plot_diagnostics object allows us to quickly generate model diagnostics and investigate for any unusual behavior.
results.plot_diagnostics(figsize=(16, 8))
plt.show()

#What to look for?
#1. Residuals SHOULD be Normally Distributed ; Check
#Top Right: The (orange colored) KDE line should be closely matched with green colored N(0,1) line. This is the standard notation
#for normal distribution with mean 0 and sd 1.
#Bottom Left: The qq plot shows the ordered distribution of residuals (blue dots) follows the linear trend of the samples 
#taken from a standard normal distribution with N(0, 1). 

#2. #Residuals are not correlated; Check
#Top Left: The standard residuals don’t display any obvious seasonality and appear to be white noise. 
#Bottom Right: The autocorrelation (i.e. correlogram) plot on the bottom right, which shows that the time series residuals have 
#low correlation with its own lagged versions.

#Lets get the predictions and confidence interval for those predictions.
#Get the predictions. The forecasts start from the 1st of Jan 2017 but the previous line shows how it fits to the data.
pred = results.get_prediction(start=pd.to_datetime('2020-5-29'), dynamic=False) #false is when using the entire history.
#Confidence interval.
pred_ci = pred.conf_int()

#Plotting real and forecasted values.
ax = y['2020':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='blue', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Ambient Temperature')
plt.legend()
plt.show()

#Takeaway: The forecats seems to be fitting well to the data. The Blue/purple thicker plot shows the confidence level in the forecasts. 



#Getting the mean squared error (average error of forecasts).
y_forecasted = pred.predicted_mean
y_truth = y['2020-6-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('MSE {}'.format(round(mse, 2)))

#Smaller the better.

print('RMSE: {}'.format(round(np.sqrt(mse), 2)))

mape=(((-y_forecasted + y_truth) / y_truth).mean())*100
print('MAPE: {}'.format(mape))

### seems to be problem here

#The time can be changed using steps.
pred_uc = results.get_forecast(steps=72)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 8))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Order_Demand')
plt.legend()
plt.show()
import fbprophet
##FB PROPHET
#predict AMBIENT TEMPERATURE
# renaming the columns 
sense1_renamed = df_psense1.rename(columns={'DATE_TIME':'ds', 'AMBIENT_TEMPERATURE':'y'})
# Make the prophet model and fit on the data
gm_prophet = fbprophet.Prophet(changepoint_prior_scale=0.25) # the parameter defines how tightly you want to fit your model

gm_prophet.fit(sense1_renamed)
# Make a future dataframe for 4 days
gm_forecast = gm_prophet.make_future_dataframe(periods=96, freq='H')

# Make predictions
gm_forecast = gm_prophet.predict(gm_forecast)
gm_prophet.plot(gm_forecast, xlabel = 'Date', ylabel = 'AMBIENT TEMPERATURE ')
plt.title('AMBIENT TEMPERATURE Prediction')
#predict daily yield(efficiency) for an inverter
df_pgen1['SOURCE_KEY'].unique()
df_temp=df_pgen1[df_pgen1['SOURCE_KEY']=='1BY6WEcLGh8j5v7']
df_temp
single_yield = df_pgen1.groupby('DATE').agg(TOTAL_YIELD = ('DAILY_YIELD', sum),
                                         DATE = ('DATE',max)
                                        )
single_yield
single_yield = single_yield.rename(columns={'DATE': 'ds', 'TOTAL_YIELD': 'y'})
single_yield
# Make the prophet model and fit on the data
gm_prophet = fbprophet.Prophet(changepoint_prior_scale=0.25) # the parameter defines how tightly you want to fit your model

gm_prophet.fit(single_yield)
# Make a future dataframe for 4 days
gm_forecast = gm_prophet.make_future_dataframe(periods=96, freq='H')

# Make predictions
gm_forecast = gm_prophet.predict(gm_forecast)
gm_prophet.plot(gm_forecast, xlabel = 'Date', ylabel = 'yield ')
plt.title('daily yield Prediction')
#predict DC_POWER 
temp_renamed = df_temp.rename(columns={'DATE_TIME':'ds', 'DC_POWER':'y'})
# Make the prophet model and fit on the data
gm_prophet = fbprophet.Prophet(changepoint_prior_scale=0.25) # the parameter defines how tightly you want to fit your model

gm_prophet.fit(temp_renamed)
# Make a future dataframe for 4 days
gm_forecast = gm_prophet.make_future_dataframe(periods=96, freq='H')

# Make predictions
gm_forecast = gm_prophet.predict(gm_forecast)
gm_prophet.plot(gm_forecast, xlabel = 'Date', ylabel = 'DC POWER ')
plt.title('DC_POWER Prediction')
