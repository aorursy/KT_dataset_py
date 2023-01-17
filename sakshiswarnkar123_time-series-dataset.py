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
import numpy as np 
import pandas as pd 
data = pd.read_csv('/kaggle/input/time-series-datasets/Electric_Production.csv')
data.head()
data.dropna(inplace=True)
data
data.info()
#data['DATE'] = pd.to_datetime(data['DATE'],format='%d.%m.%Y')
df=data['DATE'].astype('datetime64[ns]')
df
data['DATE']=df
data.info()
data['DATE']
data['year']=data['DATE'].dt.year 
data['month']=data['DATE'].dt.month 
data['day']=data['DATE'].dt.day
data['dayofweek_num']=data['DATE'].dt.dayofweek  
data.head(18)
data['hour']=data['DATE'].dt.hour
data['minute']=data['DATE'].dt.minute
data['second']=data['DATE'].dt.second
data['day_of_year']=data['DATE'].dt.dayofyear
data['leap_year']=data['DATE'].dt.is_leap_year
data.head(18)
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(15,5)})
import seaborn as sns

# plt.figure(figsize=(15,5))
# sns.distplot(data['Sales_quantity'])
plt.plot(data.IPG2211A2N)
data['rolling_mean'] = data.IPG2211A2N.rolling( window=2).mean()
data['rolling_mean6'] = data.IPG2211A2N.rolling( window=6).mean()
plt.plot(data.IPG2211A2N, label='original')
plt.plot(data.rolling_mean, label = 'window =2')
plt.plot(data.rolling_mean6, label = 'window =6')
plt.legend(loc='best')
data.head(17)
data.index = data.DATE
from statsmodels.tsa.stattools import adfuller
dftest = adfuller(data.IPG2211A2N, autolag='AIC')
dftest
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
   dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)
data
data['#IPG2211A2N'] = data['IPG2211A2N'] - data['IPG2211A2N'].shift(1)
data['#IPG2211A2N'].dropna(inplace=True)
data['#IPG2211A2N'].dropna().plot()
dftest = adfuller(data['#IPG2211A2N'], autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
   dfoutput['Critical Value (%s)'%key] = np.round(value, 1)
print (dfoutput)

dftest[1].round(4)
data['#IPG2211A2N_sea'] = data['IPG2211A2N'] - data['IPG2211A2N'].shift(12)
data['#IPG2211A2N_sea'].dropna(inplace=True)
data['#IPG2211A2N_sea'].dropna().plot()
data['#IPG2211A2N_sea'] = data['IPG2211A2N'] - data['IPG2211A2N'].shift(5)
data['#IPG2211A2N_sea'].dropna(inplace=True)
data['#IPG2211A2N_sea'].dropna().plot()
dftest = adfuller(data['#IPG2211A2N_sea'], autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
   dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)
data['log'] = np.log(data.IPG2211A2N)
dftest = adfuller(data['log'], autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
   dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)
import statsmodels.api as sm
# multiplicative
res = sm.tsa.seasonal_decompose(data.IPG2211A2N,period=12,model="multiplicative")
#plt.figure(figsize=(16,12))
fig = res.plot()
#fig.show()
res = sm.tsa.seasonal_decompose(data.IPG2211A2N,period=12,model="additive")
#plt.figure(figsize=(16,12))
fig = res.plot()
#fig.show()
import statsmodels.api as sm
sm.graphics.tsa.plot_pacf(data['log'], lags=12, method='ols')
plt.show()
#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(data['log'], nlags=12)
lag_pacf = pacf(data['log'], nlags=12, method='ols')


plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data['log'])),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(data['log'])),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
import statsmodels.api as sm
sm.graphics.tsa.plot_acf(data['log'], lags=12)
plt.show()
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data['log'])),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(data['log'])),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(data['log'], order=(2,1, 2))  
results_AR = model.fit(disp=1)
results_AR.fittedvalues
predictions_ARIMA_diff = pd.Series(results_AR.fittedvalues, copy=True)
predictions_ARIMA_diff.head()
(predictions_ARIMA_diff)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_diff_cumsum.head()
(predictions_ARIMA_diff_cumsum)
predictions_ARIMA_log = pd.Series(data['log'].iloc[0], index=data.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head(49)
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(data['IPG2211A2N'])
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-data['IPG2211A2N'])**2)/len(data['IPG2211A2N'])))
x = results_AR.forecast(steps=10)
x = np.exp(x[0])
x
rng = pd.date_range('2020-04-01', periods=10, freq='M')
rng = pd.DataFrame(rng, columns=['DATE'])
rng
rng['future'] = x
rng.index = rng.DATE
plt.plot(data['IPG2211A2N'], label='original')
plt.plot(predictions_ARIMA, label='fitted Values')
plt.plot(rng['future'], label='Future Values')
plt.legend(loc='best')
import numpy as np
import pandas as pd 
df = pd.read_csv('/kaggle/input/time-series-datasets/monthly-beer-production-in-austr.csv')
df.head()
df.dropna(inplace=True)
df['Month'] = pd.to_datetime(df['Month'],format='%Y-%m')
df.head()
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(15,5)})
import seaborn as sns

# plt.figure(figsize=(15,5))
# sns.distplot(data['Sales_quantity'])
plt.plot(df['Monthly beer production'])
df.index = df.Month
# Log Transformation
data['log'] = np.log(df['Monthly beer production'])
import statsmodels.tsa.api as smt

fig, axes = plt.subplots(1, 2, sharey=False, sharex=False)
fig.set_figwidth(12)
fig.set_figheight(4)
smt.graphics.plot_acf(df['Monthly beer production'], lags=12, ax=axes[0])
smt.graphics.plot_pacf(df['Monthly beer production'], lags=12, ax=axes[1])
plt.tight_layout()
import itertools
import warnings
warnings.filterwarnings('ignore')
# Define the p, d and q parameters to take any value between 0 and 4
p = d = q = range(0, 4)

# Generate all different combinations of p, d and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
pdq
seasonal_pdq
#import statsmodels.api as sm
# import statsmodels.tsa.api as smt
# import statsmodels.formula.api as smf

# best_aic = np.inf

# best_pdq = None
# best_seasonal_pdq = None
# temp_model = None

# for param in pdq:
#     for param_seasonal in seasonal_pdq:        
#         try:
#             temp_model = sm.tsa.statespace.SARIMAX(df['Monthly beer production'],
#                                              order = param,
#                                              seasonal_order = param_seasonal,
#                                              enforce_stationarity=False,
#                                              enforce_invertibility=False)
#             results = temp_model.fit()
#             if results.aic < best_aic:
#                 best_aic = results.aic
#                 best_pdq = param
#                 best_seasonal_pdq = param_seasonal
#         except:
#             print('he')

# print("Best SARIMAX {} x {} 12 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))