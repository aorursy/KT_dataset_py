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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.parser import parse
df = pd.read_csv('/kaggle/input/timeseries/Train.csv', parse_dates = ['Datetime'], index_col = 'Datetime')
print(df.head())
df.sort_index(inplace = True)
print(df.shape)
print(df.info())
df_train = df[0:14630]
df_test = df[14630:]
df_train = df_train.asfreq('D')
df_test = df_test.asfreq('D')

df_train = df_train.dropna()
df_test = df_test.dropna()

df_pred = pd.read_csv('/kaggle/input/timeseries/Test.csv', parse_dates = ['Datetime'], index_col = 'Datetime')
print(df_pred.head())
print(df_pred.shape)
print(df_pred.info())
df_pred.drop('ID', 1, inplace = True)
df_pred = df_pred.asfreq('D')

df.drop('ID', 1, inplace = True)
df = df.asfreq('D')
df.plot()
plt.show()

df.isnull().any()
df = df.dropna()
from statsmodels.tsa.seasonal import seasonal_decompose
df = df.sort_index()
decompose_multiplicative = seasonal_decompose(df, model='multiplicative', period = 12)
decompose_additive = seasonal_decompose(df, model = 'additive', period = 12)

plt.rcParams.update({'figure.figsize': (10,10)})
decompose_additive.plot().suptitle('Additive Decompose', fontsize = 12)
decompose_multiplicative.plot().suptitle('Multiplicative Decompose', fontsize = 12)
plt.show()
from statsmodels.tsa.stattools import adfuller
result = adfuller(df['Count'], autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')
df["Count diff"] = df["Count"]- df["Count"].shift(1)
df = df.dropna()
result = adfuller(df["Count diff"], autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

acf_50 = acf(df['Count diff'], nlags=50)
pacf_50 = pacf(df['Count diff'], nlags=50)

print(acf_50)
print(pacf_50)

plot_acf(df["Count diff"], lags= 60, alpha=0.05);
plot_pacf(df["Count diff"], lags= 60, alpha=0.05);
#plotting data
df_train['Count'].plot(figsize=(15,8), title= 'Daily Commuters', fontsize=14)
df_test['Count'].plot(figsize=(15,8), title= 'Daily Commuters', fontsize=14)
plt.show()

#Holt's Winter

y_hat_avg = df_test.copy()
fit1 = ExponentialSmoothing(np.asarray(df_train['Count']) ,seasonal_periods=12, trend='add', seasonal='add',).fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(df_test))
plt.figure(figsize=(15,8))
plt.plot( df_train['Count'], label='Train')
plt.plot(df_test['Count'], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()
