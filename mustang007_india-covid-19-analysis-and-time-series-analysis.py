import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
# import os
# print(os.listdir("../input/time-series/case.xlsx"))
dataset = pd.read_csv('../input/datacovid19/case_time_series.csv')
dataset.info()
dataset['date'] = pd.to_datetime(dataset['Date'])
dataset.info()
import seaborn as sns
plt.figure(figsize=(20,10))
g = sns.jointplot(dataset['date'],dataset['Total Confirmed'], kind="kde", height=7, space=0)
plt.gcf().autofmt_xdate()
plt.show()
plt.figure(figsize=(15,10))
sns.set(style="darkgrid")
sns.lineplot(x='date', y='Total Confirmed',
             data=dataset,  linewidth=3)
plt.gcf().autofmt_xdate()
plt.show()
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime as dt
plt.figure(figsize=(15,10))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
plt.plot(dataset['date'],dataset['Total Confirmed'], linewidth=3)
plt.gcf().autofmt_xdate()
plt.show()
df = dataset.groupby(['date'])[['Daily Confirmed','Daily Recovered','Daily Deceased']].sum()

plt.figure(figsize=(15,10))
plt.title('DAILY CASES OF COVID-19 CASES IN INDIA', fontsize=30)
plt.xlabel('date')
plt.ylabel('NO OF CASES')
plt.plot(df.index,df['Daily Confirmed'], label='daily_confirmed', linewidth=3)
plt.plot(df.index,df['Daily Recovered'], label='daily_recovered', linewidth=3, color='green')
plt.plot(df.index,df['Daily Deceased'], label='daily_deceased', linewidth=3, color='red')
plt.bar(df.index,df['Daily Confirmed'], alpha=0.2, color='c')
plt.style.use('ggplot')
plt.legend()

# total cases
df_total = dataset.groupby(['date'])[['Total Confirmed','Total Recovered','Total Deceased']].sum()
plt.figure(figsize=(15,10))
plt.title('TOTAL CASES OF COVID-19 CASES IN INDIA', fontsize=30)
plt.xlabel('date', fontsize=20)
plt.ylabel('Total Number of Cases', fontsize=20)
plt.plot(df_total.index,df_total['Total Confirmed'], label='Total Confirmed',linewidth=3)
plt.plot(df_total.index,df_total['Total Recovered'], label='Total Recovered', linewidth=3)
plt.plot(df_total.index,df_total['Total Deceased'], label='Total Deceased', linewidth=3)
plt.bar(df_total.index,df_total['Total Confirmed'], label='Total Confirmed', alpha=0.2, color='c')
plt.style.use('ggplot')
plt.legend(loc='best')

#df
labels = 'Total Recovered','Total Deceased'
recovered = dataset['Total Recovered']
deceased = dataset['Total Deceased']
sizes = [recovered.sum() , deceased.sum()]
explode = [0,0.1]
colors = ['yellowgreen','lightcoral']
plt.figure(figsize=(10,20))

plt.title('Dsitribution of confirmed cases Till 4th MAY', fontsize=20)
plt.pie(sizes, autopct='%1.1f%%', labels=labels,explode=explode,colors=colors, shadow=True)
plt.legend(labels, loc='best')
plt.show()
df = pd.read_excel('../input/time-series/case.xlsx')
df.head()
df = df.set_index(['date'])
df.head()
plt.figure(figsize=(10,8))
plt.plot(df)
df.isnull().sum()
# rolling mean

rolmean = df.rolling(window=6).mean()
rolstd = df.rolling(window=6).std()
print(rolmean,rolstd)
plt.figure(figsize=(10,8))
plt.plot(df, color='blue', label='original cases')
plt.plot(rolmean, color='red', label='rolling mean')
plt.plot(rolstd, color='black', label='rolling standard deviation')
plt.legend(loc='best')
plt.show()
from statsmodels.tsa.stattools import adfuller
def test(data):
    rolmean = data.rolling(window=4).mean()
    rolstd = data.rolling(window=4).std()
    plt.figure(figsize=(10,8))
    plt.plot(data, color='blue', label='original cases')
    plt.plot(rolmean, color='red', label='rolling mean')
    plt.plot(rolstd, color='black', label='rolling standard deviation')
    plt.legend(loc='best')
    plt.show()
    
    dftest = adfuller(data['Total Confirmed'], autolag = 't-stat')
    dfoutput = pd.Series(dftest[0:4], index=['test statitics','p_value','lags used','number of observations'])
    for key,value in dftest[4].items():
        dfoutput['critcal value (%s)'%key] = value
        
    print(dfoutput)
test(df)
df_log = np.log(df)
plt.figure(figsize=(10,8))
plt.plot(df_log)
plt.gcf().autofmt_xdate()
plt.show()
df_log.tail()
test(df_log)
movingaverage = df_log.rolling(window=4).mean()
rolstd = df_log.rolling(window=4).std()
plt.figure(figsize=(10,8))
plt.plot(df_log, color='blue', label='original cases')
plt.plot(movingaverage, color='red', label='rolling mean')
plt.plot(rolstd, color='black', label='rolling standard deviation')
plt.legend(loc='best')
plt.show()
df_log_minus = df_log - movingaverage
df_log_minus.dropna(inplace=True)
df_log_minus.tail(12)
test(df_log_minus)
data_shift = df_log_minus - df_log_minus.shift()
plt.figure(figsize=(10,8))
plt.plot(data_shift)
data_shift.dropna(inplace=True)
test(data_shift)
# TO CALCULATE P AND Q VALUE FOR ARIMA MODEL
# TO CALCULATE ACF AND PACF

from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(data_shift, nlags=20)
lag_pacf = pacf(data_shift, nlags=20, method='ols')

plt.figure(figsize=(10,8))
#plot acf
plt.subplot(211)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data_shift)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(data_shift)), linestyle='--', color='gray')
plt.title('ACF')
plt.legend(loc='best')

#plot pacf
plt.subplot(212)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data_shift)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(data_shift)), linestyle='--', color='gray')
plt.title('PACF')
plt.legend(loc='best')
from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.filterwarnings('ignore')
model = ARIMA(df_log, order=(4,1,4))
result = model.fit(disp=-1)
plt.figure(figsize=(10,8))
plt.plot(data_shift)
plt.plot(result.fittedvalues, color='blue')
plt.title('RSS %-4F'% sum((result.fittedvalues- data_shift['Total Confirmed'])**2))
pred_arima_diff = pd.Series(result.fittedvalues, copy=True)
pred_arima_diff
# we will take cumsum
pred_arima_diff_cumsum = pred_arima_diff.cumsum()
pred_arima_diff_cumsum.tail()
prediction = pd.Series(df_log['Total Confirmed'].iloc[0], index=df_log.index)
prediction = prediction.add(pred_arima_diff_cumsum, fill_value=0)
prediction.tail()
pred = np.exp(prediction)
plt.figure(figsize=(10,8))
plt.plot(df)
plt.plot(pred, color='green')
pred.tail()
df

result.plot_predict(1,106)
plt.figure(figsize=(10,8))
x = result.forecast(steps=10)
x = np.exp(x[0])
for i in x:
    print(i)