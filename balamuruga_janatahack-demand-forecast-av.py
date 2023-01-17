#Kaggle data import code
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#necessary Imports
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.ar_model import AutoReg

from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.metrics import mean_squared_log_error
from fbprophet import Prophet
N = 13
#Load datasets and fill missing values by 0
full_train = pd.read_csv('/kaggle/input/demand-forecasting/train_0irEZ2H.csv', parse_dates=['week'])
full_test = pd.read_csv('/kaggle/input/demand-forecasting/test_nfaJ3J5.csv', parse_dates=['week'])


full_train.fillna(0, inplace=True)
full_test.fillna(0,inplace= True)
full_train.info()
#Get STORE_ITEM pair
dataset = [v for k, v in full_train.groupby(['sku_id','store_id'])]
test_dataset = [v for k, v in full_test.groupby(['sku_id','store_id'])]
len(test_dataset)
#Create units_sold column in test data with zeroes
for i in range(len(test_dataset)):
    test_dataset[i]['units_sold'] = np.zeros(len(test_dataset[0]))
test_dataset[0]
dataset[0][['week','units_sold']][:104]
len(dataset[i].iloc[104:])
train_list = []
validation_list = []
test_list =[]
leng = len(dataset)
for i in range(leng):
    train_list.append(dataset[i][['record_ID','week','units_sold']][:104])
    validation_list.append(dataset[i][['record_ID','week','units_sold']][104:])
    test_list.append(test_dataset[i][['record_ID','units_sold']])
    
    
test_list[0]
merged_train = pd.concat(train_list)
merged_train.to_csv("train_dataset_merged.csv")
merged_validation = pd.concat(validation_list)
merged_validation.to_csv("validation_dataset_merged.csv")
merged_test = pd.concat(test_list)
merged_test.to_csv("test_dataset_merged.csv")
data = pd.read_csv("./train_dataset_merged.csv")
testsdg = pd.read_csv("../input/demand-forecasting/train_0irEZ2H.csv",parse_dates=['week'])
future_pd = testsdg['week'].sort_values().unique()
future_pd = pd.DataFrame({'ds': future_pd})
future_pd.columns
train_dataset_merged = pd.read_csv("./train_dataset_merged.csv",parse_dates=['week'])
train_dates =  pd.DataFrame({'ds': train_dataset_merged['week'].sort_values().unique() }).squeeze()

train_dates
120120/104
start = 0
ds_y = pd.DataFrame()
ds_y['ds'] = train_dates
ds_y['y'] = data['units_sold'][start:start+104]
ds_y

def prophet(ds_y,future_pd):
    model = Prophet(
    interval_width=0.95,
    growth='linear',
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative')
    model.fit(ds_y)
    forecast_pd = model.predict(future_pd)
    print(forecast_pd)
    return forecast_pd['yhat']
def VARMAX(data):
    model = VARMAX(data)
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.forecast()
    return yhat

def ARIMA(data):
    model = ARMA(data, order=(1, 1))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(data), len(data)+12)
 
    return yhat

def Autoreg(data):
    model = AutoReg(data, lags=1)
    model_fit = model.fit()
    print(len(data))
    yhat = model_fit.predict(start="16/07/2013", end="01/10/2013")
    return yhat
start = 0 
pred_values_list =[]
for i in range(int(len(data)/104)):
    ds_y = pd.DataFrame()
    ds_y['ds'] = train_dates
    ds_y['y'] = data['units_sold'][start:start+104].values
    pred_values_list.append(prophet(ds_y,future_pd))
    start+=104
pred_values_list
x = pd.concat(pred_values_list)
x = x.apply(np.floor)
x = x.abs()
any(x<0)
x.to_csv("pred.csv")
