import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
test=pd.read_csv('../input/into-the-future/test.csv')
train=pd.read_csv('../input/into-the-future/train.csv')
train.info()
#parse string to date time type
train['time']=pd.to_datetime(train['time'],infer_datetime_format=True)
indexedDataset=train.set_index(['time'])
test['time']=pd.to_datetime(test['time'],infer_datetime_format=True)
indexedDataset=test.set_index(['time'])
from datetime import datetime
indexedDataset.tail(5)
indexedDataset.describe()

# plot graph
plt.xlabel('Time')
plt.ylabel('feature_2')
plt.plot(indexedDataset)
roll_mean=indexedDataset.rolling(window=10).mean()
roll_mean.plot()
from statsmodels.tsa.stattools import adfuller
dftest = adfuller(indexedDataset['feature_2'],autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value(%s)'%key] = value
    
print(dfoutput)
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
train.drop('feature_1',axis=1,inplace=True)
train.drop('id',axis=1,inplace=True)
indexedDataset.drop('id',axis=1,inplace=True)
plot_acf(indexedDataset) # to identify the value of q
plot_pacf(indexedDataset) # to identify the value of p
# p-value= 4   q=30   d=1
from statsmodels.tsa.arima_model import ARIMA
train.dropna()
train = pd.DataFrame(train.replace([np.inf, -np.inf], np.nan))
train = train.fillna(method='ffill')
train = train.fillna(method='bfill')
train = train.iloc[:, 0]
model = ARIMA(train, order=(2,1,2))

model_fit=model.fit()
model_fit.aic
model_forecast = model_fit.forecast
model_forecast
test.shape
np.sqrt(mean_squared_error(test,model_forecast))




