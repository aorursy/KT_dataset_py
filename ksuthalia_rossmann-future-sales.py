import os
for dirname, _, filenames in os.walk('/kaggle/input/rossmann-store-sales'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv')
test = pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv')
store = pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')
sample = pd.read_csv('/kaggle/input/rossmann-store-sales/sample_submission.csv')
train.head()
store.head()
train_open = train[train.Open == 1]
features = ["Store", "StoreType", "Assortment", "CompetitionDistance", "Promo2"]
store_selected = store[features]
train_open.info() # As you can see there is no null value in the train set
store_selected.info() 
#There are Null Values in "CompetitionDistance"
together = train_open.merge(store_selected)
together = together.drop(columns = ['Date']) # we don't need the Date column
together.info()
#There are Null Values in "CompetitionDistance"
# the number of null values
together.CompetitionDistance.isnull().sum()
mean_CD = round(together.CompetitionDistance.mean(),0)
mean_CD
together.CompetitionDistance[together.CompetitionDistance.isnull() == True] = mean_CD
together.info()
together.columns
X = together[['DayOfWeek', 'Open', 'Promo', 'Customers',
       'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment', ## there is no feaure 'Customers' in Test set
       'CompetitionDistance', 'Promo2']]
y = together[['Sales']]
X = X.reset_index()
del X['index']
y = y.reset_index()
del y['index']
X.StateHoliday.unique() 
# here, we have 0 and '0'. And it should be the same so we have to change the type of 0 to object
X.StateHoliday[X.StateHoliday == 0] = '0'
from sklearn.preprocessing import OneHotEncoder
X.columns
X_OH =  X[['DayOfWeek','StateHoliday','StoreType','Assortment']] # for OneHot Encoding
X_rest = X[['Customers', 'Promo','SchoolHoliday','CompetitionDistance', 'Promo2']] # we don't need open because it is always 1
OHencoder = OneHotEncoder(handle_unknown='ignore')
OH_result = pd.DataFrame(OHencoder.fit_transform(X_OH).toarray())
OH_result.columns = OHencoder.get_feature_names(['DayOfWeek','StateHoliday','StoreType','Assortment'])
OH_result.head()
X_final = pd.concat([X_rest,OH_result],axis = 1)
X_final
X_final.info()
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_final, y, test_size=0.33, random_state=42)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=0)
rf.fit(X_train, y_train)
rf.score(X_valid, y_valid)
X
predicted_RF = pd.DataFrame(rf.predict(X_valid))
X_ML = together.iloc[X_valid.index,:]
X_ML = X_ML.reset_index()
del X_ML['index']
All_RF = pd.concat([X_ML,predicted_RF], axis = 1)
All_RF = All_RF.rename(columns={0: "Predicted"})
All_RF
All_RF[['Sales','Predicted']] ## to compare
from sklearn.metrics import mean_absolute_error
mean_absolute_error(All_RF.Sales, All_RF.Predicted)
# MASE
MASE_RF = mean_absolute_error(All_RF.Sales, All_RF.Predicted)/len(All_RF.Sales)
MASE_RF
from sklearn.neural_network import MLPRegressor
NN = MLPRegressor(hidden_layer_sizes=(30,30,30),max_iter=30)
NN.fit(X_train, y_train)
NN.score(X_valid, y_valid)
predicted_NN = pd.DataFrame(NN.predict(X_valid))
All_NN = pd.concat([X_ML,predicted_NN], axis = 1)
All_NN = All_NN.rename(columns={0: "Predicted"})
All_NN
All_NN[['Sales','Predicted']] ## to compare
mean_absolute_error(All_NN.Sales, All_NN.Predicted)
# MASE
MASE_NN = mean_absolute_error(All_NN.Sales, All_NN.Predicted)/len(All_NN.Sales)
MASE_NN
week_mean_RF = All_RF.groupby('DayOfWeek').agg({'Sales':'mean','Predicted':'mean'})
week_mean_RF['Difference'] = week_mean_RF['Sales'] - week_mean_RF['Predicted']
week_mean_RF
plt.figure(figsize = (12,8))
sns.barplot(x = week_mean_RF.index, y = week_mean_RF.Sales, color = 'red', alpha = 0.3, label = 'Sales')
sns.barplot(x = week_mean_RF.index, y = week_mean_RF.Predicted, color = 'blue', alpha = 0.1, label = 'Predicted')
plt.legend()
plt.title('Random Forest weekly Comparison')
plt.figure(figsize = (12,8))
sns.barplot(x = week_mean_RF.index, y = week_mean_RF.Difference, color = 'gray')
plt.title('Difference Random Forest')
week_mean_NN = All_NN.groupby('DayOfWeek').agg({'Sales':'mean','Predicted':'mean'})
week_mean_NN['Difference'] = week_mean_NN['Sales'] - week_mean_NN['Predicted']
week_mean_NN
plt.figure(figsize = (12,8))
sns.barplot(x = week_mean_NN.index, y = week_mean_NN.Sales, color = 'red', alpha = 0.3, label = 'Sales')
sns.barplot(x = week_mean_NN.index, y = week_mean_NN.Predicted, color = 'blue', alpha = 0.1, label = 'Predicted')
plt.legend()
plt.title('Neural Network weekly Comparison')
plt.figure(figsize = (12,8))
sns.barplot(x = week_mean_NN.index, y = week_mean_NN.Difference, color = 'gray')
plt.title('Difference Neural Network')
print('MASE calculated by Random Forest: ', MASE_RF)
print('MASE calculated by Neural Network: ', MASE_NN)
test.head()
train_open.head()
train_open.Date = pd.to_datetime(train_open.Date)
train_for_ts = train_open[['Store','Date','Customers']]
# Splitting based on Store Number
ts_stores = {}
for i, g in train_for_ts.groupby('Store'):
    ts_stores.update({i : g.reset_index(drop=True)})
ts_stores[1]
from statsmodels.tsa.arima_model import ARIMA
ARIMA_model = ARIMA(ts_stores[1].Customers, order=(1,0,2))
ARIMA_model_fit = ARIMA_model.fit()
ARIMA_predicted = ARIMA_model_fit.predict()
gr = pd.concat([ts_stores[1],ARIMA_predicted],axis = 1)
gr = gr.rename(columns={0: "Predicted"})
gr
# To check whether ARIMA Model works well
plt.figure(figsize = (16,8))

sns.lineplot(x = gr.Date[0:360], y = gr.Customers[0:360], label = 'Customers')
sns.lineplot(x = gr.Date[0:360], y = gr.Predicted[0:360], label = 'Predicted')
result_ARIMA = {}
result_ARIMA = pd.DataFrame(result_ARIMA)

for index,value in ts_stores.items():
    
    ARIMA_model = ARIMA(value.Customers, order=(1,0,2))
    ARIMA_model_fit = ARIMA_model.fit()
    ARIMA_predicted = ARIMA_model_fit.predict()
    ARIMA_forecast = ARIMA_model_fit.forecast(41)
    
    
    tmp = pd.concat([value,ARIMA_predicted],axis = 1)
    tmp = tmp.rename(columns={0: "Predicted"})
    
    
    result_ARIMA = result_ARIMA.append(tmp, ignore_index=True)
    
    
result_ARIMA.info()
result_ARIMA
len(test[ (test.Store == 1) & (test.Open == 1)])
## we have to forecast 41 days!
ARIMA_model = ARIMA(ts_stores[1].Customers, order=(1,0,2))
ARIMA_model_fit = ARIMA_model.fit()
ARIMA_predicted = ARIMA_model_fit.predict()
ARIMA_forecast = ARIMA_model_fit.forecast(41)[1]
tmp = pd.concat([ts_stores[1],ARIMA_predicted],axis = 1)
tmp = tmp.rename(columns={0: "Predicted"})
tmp
from datetime import datetime
times = pd.date_range(start="2015-08-01",end="2015-09-17")
weekday = times.weekday
times = pd.DataFrame(times)
times = times.rename(columns={0: "Date"})
weekday = pd.DataFrame(weekday)
weekday = weekday.rename(columns={0: "Open"})
x = pd.concat([times,weekday],axis = 1)
x['Open'][x['Open'] != 6] = 1
x['Open'][x['Open'] == 6] = 0
ARIMA_forecast = pd.DataFrame(ARIMA_forecast)
ARIMA_forecast = ARIMA_forecast.rename(columns={0: "Forecast"})
base = x[x['Open'] == 1]
base = base.reset_index()
del base['index']
f = []
for i in range(1,1116):
    
    for ii in range(1,42):
        
        f.append([i,ii])
              
f = pd.DataFrame(f)
f
base2 = base.copy()
for i in range(115):
    
    base2 = pd.concat([base2,base2],axis = 0)
base2.head(45)
g = pd.concat([ base,ARIMA_forecast ],axis = 1)
g = pd.concat([g, ])
