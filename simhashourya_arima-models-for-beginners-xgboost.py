# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Reading the Datasets
sales_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")
shops= pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")
item_categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
print(sales_train.shape)
sales_train.head()

sales_train.drop(sales_train[(sales_train['item_cnt_day']<=0)|(sales_train['item_price']<=0)].index ,axis=0,inplace=True)
# sales_train['date']=pd.to_datetime(sales_train['date'],dayfirst=True)

data=sales_train.groupby(["date","date_block_num","shop_id","item_id"])["item_cnt_day"].sum().reset_index()
ts=data.groupby(['date'])['item_cnt_day'].sum()
ts.astype('float')
plt.figure(figsize=(12,10))
plt.title('Total Sales of the item')
plt.xlabel('Month-Year')
plt.ylabel('Quantity of Sales')
plt.plot(ts)
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12,center=False).mean()
    rolstd = timeseries.rolling(window=12,center=False).std()#window=12, because of yearly trend for both mean and variance
#Plot rolling statistics:
    plt.figure(figsize=(15,10))
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
test_stationarity(ts)
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# def encoding_categorical(dataset):
#     categorical_columns=['shop_id','item_id']
#     for column in categorical_columns:
#         dataset[str(column)]=le.fit_transform(dataset[str(column)])
#     return dataset
# complete_data=encoding_categorical()


ts_data = pd.DataFrame(ts)

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_data,period=100)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
plt.figure(figsize=(15,8))
plt.subplot(411)
plt.plot(ts,label='Orginial')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.subplot(413)
plt.plot(seasonal,label='Seasonal')
plt.subplot(414)
plt.plot(residual,label='Residual')
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf,pacf

lag_acf=acf(ts,fft=False)
lag_pacf=pacf(ts,method='ols')
plt.figure(figsize=(11,8))
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
model=ARIMA(ts,order=(2,0,2))
result = model.fit(disp=-1)
plt.plot(ts,label="Original")
plt.plot(result.fittedvalues,color='red',label="Predicted")
forecast_errors = [ts[i]-result.fittedvalues[i] for i in range(len(ts))]
bias = sum(forecast_errors) * 1.0/len(ts)
print('Bias: %f' % bias)
test=pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
predictions = pd.DataFrame(result.fittedvalues).reset_index()
predictions.columns=["date","predictions"]
predictions.head()# Monthly sales forecasting

