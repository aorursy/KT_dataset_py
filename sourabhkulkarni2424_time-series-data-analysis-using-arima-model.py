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

%matplotlib inline
df = pd.read_csv('../input/delhi-city-temperature-time-series-data/datasets_312121_636393_DailyDelhiClimateTrain.csv',index_col='date',parse_dates=True)
df.head()
df.drop(['humidity','wind_speed','meanpressure'],axis=1,inplace=True)

df.head()
df.info()
plt.hist(df.meantemp,bins=20, rwidth=0.8)

plt.show()
df.meantemp.describe()
df.meantemp.plot(figsize=(20,10))

plt.show()
monthlytemp = df.resample('M').mean()

monthlytemp.head()
monthlytemp.plot(figsize=(20,10))

plt.show()
from statsmodels.tsa.stattools import adfuller



def adfuller_test(meantemp):

    result=adfuller(meantemp)

    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']

    for value,label in zip(result,labels):

        print(label+' : '+str(value) )

    if result[1] <= 0.05:

        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")

    else:

        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
adfuller_test(monthlytemp['meantemp'])
monthlytemp['first_difference'] = monthlytemp.meantemp-monthlytemp.meantemp.shift(1)

monthlytemp.head()
adfuller_test(monthlytemp['first_difference'].dropna())
monthlytemp.first_difference.plot(figsize=(20,10))
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_acf(monthlytemp['first_difference'].dropna())



plot_pacf(monthlytemp['first_difference'].dropna())
monthlytemp.shape
traindata= monthlytemp[0:30]

testdata= monthlytemp[31:]
traindata.head()
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(traindata.meantemp,order=(3,1,2), seasonal_order=(3,1,2,12), m=12)

results = model.fit()

results.summary()
start=len(traindata)+1

end=len(traindata)+len(testdata)

predictions = results.predict(start=start, end=end).rename('SARIMAX Predictions')

predictions
testdata['Predicted_value']=predictions

testdata
testdata.meantemp.plot(legend= True,figsize=(20,10))

predictions.plot(legend= True)

plt.show()
from statsmodels.tools.eval_measures import rmse

error = rmse(testdata.meantemp,predictions)

print('Root Mean Squared Error ',error)
model = SARIMAX(monthlytemp.meantemp,order=(3,1,2), seasonal_order=(3,1,2,12), m=12)

results = model.fit()

fcast = results.predict(len(monthlytemp)-1,len(monthlytemp)+11,typ='levels').rename('SARIMA Future Forecast')
fcast
monthlytemp.meantemp.plot(legend=True, figsize=(20,10))

fcast.plot(legend=True)