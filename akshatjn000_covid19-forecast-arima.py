# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

%matplotlib inline

from matplotlib.pyplot import rcParams



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

dataset_Train = pd.read_csv ('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

dataset_Test = pd.read_csv ('/kaggle/input/covid19-global-forecasting-week-3/test.csv')

dataset_Train.head()

dataset_Train['Date'] = pd.to_datetime(dataset_Train['Date'], infer_datetime_format = True)

gb_Train = dataset_Train.groupby("Country_Region")

gb_Train.get_group('Afghanistan').set_index("Date").head()
confirmed_Cases = pd.pivot_table(dataset_Train, values="ConfirmedCases", index="Date", columns='Country_Region')

fatalities = pd.pivot_table(dataset_Train, values="Fatalities", index="Date", columns='Country_Region')           
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

plt.plot(confirmed_Cases['India'])

plt.plot(confirmed_Cases['Pakistan'])

plt.show()
confirmed_Cases.head()

fatalities.head()
sin_Confirmed_Cases =np.sin(confirmed_Cases)

plt.plot(sin_Confirmed_Cases['India'])

plt.show()

rolmean = sin_Confirmed_Cases.rolling(window=5).mean()

rolstd = sin_Confirmed_Cases.rolling(window=5).std()

rolmean.head()
rolmean.fillna(0, inplace=True)

rolstd.fillna(0, inplace=True)

rolmean.head()
plt.plot(sin_Confirmed_Cases['India'], color ='blue', label = 'Original')

plt.plot(rolmean['India'], color ='red', label = 'Rolling Mean')

plt.plot(rolstd['India'], color ='black', label = 'Rolling STD')

plt.show(block='False')
#

from statsmodels.tsa.stattools import adfuller

print('Results of Dicky-Fuller Test')

dftest = adfuller(sin_Confirmed_Cases['Afghanistan'], autolag='AIC')



dfoutput = pd.Series(dftest[0:5], index=['Test Statistics', 'P-Value', '#lags used', 'Number of observations used','Critical Value'])



    

print(dfoutput)
exponential_Decaying_Wtd_Avg = sin_Confirmed_Cases.ewm(halflife=5, min_periods=0, adjust=True).mean()

plt.plot(exponential_Decaying_Wtd_Avg['India'],color='red')

plt.plot(sin_Confirmed_Cases['India'], color='blue')

plt.show(block=False)
#Check Stationarity 



from statsmodels.tsa.stattools import adfuller

print('Results of Dicky-Fuller Test')

dftest = adfuller(exponential_Decaying_Wtd_Avg['Pakistan'], autolag='AIC')



dfoutput = pd.Series(dftest[0:5], index=['Test Statistics', 'P-Value', '#lags used', 'Number of observations used','Critical Value'])



    

print(dfoutput)

#ACF & PACF Graphs Decay WTD AVG

from statsmodels.tsa.stattools import acf, pacf

lag_Acf = acf(exponential_Decaying_Wtd_Avg['India'], nlags=10)

lag_Pacf = pacf(exponential_Decaying_Wtd_Avg['India'], nlags=10, method='ols')



plt.subplot(121)

plt.plot(lag_Acf)

plt.axhline(y=0, linestyle='--', color='grey')

plt.axhline(y=-1.96/np.sqrt(len(sin_Confirmed_Cases['India'])), linestyle='--', color='grey')

plt.axhline(y=1.96/np.sqrt(len(sin_Confirmed_Cases['India'])), linestyle='--', color='grey')

plt.title('Auto-Corelation Function')



plt.subplot(122)

plt.plot(lag_Pacf)

plt.axhline(y=0, linestyle='--', color='grey')

plt.axhline(y=-1.96/np.sqrt(len(sin_Confirmed_Cases['India'])), linestyle='--', color='grey')

plt.axhline(y=1.96/np.sqrt(len(sin_Confirmed_Cases['India'])), linestyle='--', color='grey')

plt.title('Partial Auto-Corelation Function')



plt.tight_layout()

#ACF & PACF Graphs Sin Function



from statsmodels.tsa.stattools import acf, pacf

lag_Acf = acf(sin_Confirmed_Cases['Afghanistan'], nlags=10)

lag_Pacf = pacf(sin_Confirmed_Cases['Afghanistan'], nlags=10, method='ols')



plt.subplot(121)

plt.plot(lag_Acf)

plt.axhline(y=0, linestyle='--', color='grey')

plt.axhline(y=-1.96/np.sqrt(len(sin_Confirmed_Cases['India'])), linestyle='--', color='grey')

plt.axhline(y=1.96/np.sqrt(len(sin_Confirmed_Cases['India'])), linestyle='--', color='grey')

plt.title('Auto-Corelation Function')



plt.subplot(122)

plt.plot(lag_Pacf)

plt.axhline(y=0, linestyle='--', color='grey')

plt.axhline(y=-1.96/np.sqrt(len(sin_Confirmed_Cases['India'])), linestyle='--', color='grey')

plt.axhline(y=1.96/np.sqrt(len(sin_Confirmed_Cases['India'])), linestyle='--', color='grey')

plt.title('Partial Auto-Corelation Function')



plt.tight_layout()
from statsmodels.tsa.arima_model import ARIMA



model = ARIMA(exponential_Decaying_Wtd_Avg['India'], order = (5,0,0))

result_AR = model.fit(disp=-1)

plt.plot(exponential_Decaying_Wtd_Avg['India'])

plt.plot(result_AR.fittedvalues, color='red')

plt.title('RSS: %4f'  %sum((result_AR.fittedvalues-exponential_Decaying_Wtd_Avg['India'])**2))

plt.show()



predictions_Arima_Decay = pd.Series(result_AR.fittedvalues, copy=True)

predictions_Arima_Decay.head()