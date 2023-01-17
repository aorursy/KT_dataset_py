import numpy as np

import pandas as pd

import statsmodels.api as sm

import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.arima_model import ARIMA

from pandas.tseries.offsets import DateOffset



import matplotlib.pyplot as plt

%matplotlib inline



# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/industrial-production-index-in-usa/INDPRO.csv')

df.head()
df.columns = ['Date', 'IPI']

df.head()
# to check NAs

df.info()

df.isnull().sum()
df['Date'] = pd.to_datetime(df['Date'])
#setting up Date column as an Index

df.set_index ('Date', inplace = True)

df.index
#Date slicing

df_new = df['1998-01-01':]

df_new.tail()
df_new.describe().transpose()
f, ax = plt.subplots(figsize = (16,10))

ax.plot(df_new, c = 'r');
# rot = rotates labels at the bottom

# fontsize = labels size

# grid False or True



df_new.boxplot('IPI', rot = 80, fontsize = '12',grid = True);
time_series = df_new['IPI']

type(time_series)
time_series.rolling(12).mean().plot(label = '12 Months Rolling Mean', figsize = (16,10))

time_series.rolling(12).std().plot(label = '12 Months Rolling Std')

time_series.plot()

plt.legend();
result = adfuller(df_new['IPI'])
#to make it readable

def adf_check(time_series):

    

    result = adfuller(time_series)

    print('Augmented Dickey-Fuller Test')

    labels = ['ADF Test Statistic', 'p-value', '# of lags', 'Num of Obs used']

    

    print('Critical values:')

    for key,value in result[4].items():

        print('\t{}: {}'.format(key, value) )

    

    for value, label in zip(result,labels):

        print(label+ ' : '+str(value))

    

    if ((result[1] <= 0.05 and  result[0] <= result[4]['1%']) or

    (result[1] <= 0.05 and  result[0] <= result[4]['5%']) or

        (result[1] <= 0.05 and  result[0] <= result[4]['10%'])):

        print('Reject null hypothesis')

        print ('Data has no unit root and is stationary')

   

    else:

        print('Fail to reject null hypothesis')

        print('Data has a unit root and it is non-stationary')
adf_check(df_new['IPI'])
df_new['Dif_1'] = df_new['IPI'] - df_new['IPI'].shift(1)

df_new['Dif_1'].plot(rot = 80, figsize = (14,8));
adf_check(df_new['Dif_1'].dropna())
#If need to take a second difference



#df_new['Dif_2'] = df_new['Dif_1'] - df_new['Dif_1'].shift(1)

#adf_check(df_new['Dif_2'].dropna())
df_new['Dif_Season'] = df_new['IPI'] - df_new['IPI'].shift(12)

df_new['Dif_Season'].plot(rot = 80, figsize = (14,8));
adf_check(df_new['Dif_Season'].dropna())
df_new['Dif_Season_1'] = df_new['Dif_1'] - df_new['Dif_1'].shift(12)

df_new['Dif_Season_1'].plot(rot = 80, figsize = (14,8));
adf_check(df_new['Dif_Season_1'].dropna())
df_new['Dif_mean'] = df_new['IPI'] - df_new['IPI'].rolling(12).mean()

df_new['Dif_mean'].plot(rot = 80, figsize = (14,8));
adf_check(df_new['Dif_mean'].dropna())
#freq can be set to True or number of periods. Ex: 12 months

decomp = seasonal_decompose(time_series, freq = 12)

fig = decomp.plot()

fig.set_size_inches(15,8)
acf_seasonal = plot_acf(df_new['Dif_Season_1'].dropna(), lags = 40, color = "purple", marker = "^")

pacf_plot = plot_pacf(df_new['Dif_Season_1'].dropna(), lags = 30, color = "Green", marker = "*")
#model = ARIMA(df_new['IPI'], order = (14,1,12))

model = sm.tsa.statespace.SARIMAX(df_new['IPI'],order=(14,1,12), seasonal_order=(1,1,1,12))

model_result = model.fit()

print(model_result.summary());
#to plot residuals:

model_result.resid.plot(rot = 80);
#create additional future dates:

forecast_dates = [df_new.index[-1] + DateOffset(months=x) for x in range(1,24)]

df_future = pd.DataFrame(index=forecast_dates, columns = df_new.columns)

df_final = pd.concat([df_new, df_future])
df_final['Forecast'] = model_result.predict(start=220,end=280, alpha = 0.05)

df_final[['IPI','Forecast']].plot(figsize = (12,8));
y = model_result.plot_diagnostics(figsize = (10,6))
forecast = model_result.get_forecast(steps = 60)

conf_int = forecast.conf_int()
ax = df_final[['IPI','Forecast']].plot(figsize = (12,8))

ax.fill_between(conf_int.index,

               conf_int.iloc[:, 0],

               conf_int.iloc[:,1], color = 'grey', alpha = 0.5)

ax.set_xlabel('Time')

ax.set_ylabel('Industrial Production Index')

ax.set_title('SARIMA Forecast')

plt.legend();