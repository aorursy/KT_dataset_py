# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy.ma as ma
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn import preprocessing
import os #modulos de gestion de directorios
import glob #modulo de visualización de directorios
#import xgboost as xgb
color = sns.color_palette()
import sys
%matplotlib inline

pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.max_columns = 999
#cambiamos al directorio de trabajo donde tenemos los datos
#os.chdir("../input")
os.getcwd() 
print(glob.glob("../input/*.*"))
train_df = pd.read_csv("../input/train.csv", \
                      parse_dates=True, index_col=0)
test_df = pd.read_csv("../input/test.csv", \
                     parse_dates=True, index_col=0 )
print("Train shape : ", train_df.shape)
print("Test shape : ", test_df.shape)
train_df=train_df.reset_index()
#we create some new fields to easy manipulate
#forecasting probably should be at item-store because demand pattens could vary much dep. items and store 
train_df['weekday']=pd.DatetimeIndex(train_df['date']).weekday
train_df['month']=pd.DatetimeIndex(train_df['date']).month 
train_df['year']=pd.DatetimeIndex(train_df['date']).year
train_df['itemstore']=train_df.item.astype(str)+"-"+train_df.store.astype(str)
#overview of data
print("number of different items: %i" %(len(np.unique(train_df.item))))
print("number of different stores: %i" %(len(np.unique(train_df.store))))
print("number of different dates: %i" %(len(np.unique(train_df.date))))
print("maximun date in data: %s" %(max(train_df.date)))
print("minimum date in data: %s" %(min(train_df.date)))
print("number of different itemstore: %i" %(len(np.unique(train_df.itemstore))))
#create some lists to see range of unique values
stores = list(set(train_df.store))
item = list(set(train_df.item))
itemstore = list(set(train_df.itemstore))
#we check anual sales profile comparing stores
c=train_df.groupby(['year','store']).sum()
plt.figure(figsize=(15,10))
d=c.unstack()
d.plot(y='sales')
#we check seasonal sales profile comparing stores
c=train_df.groupby(['month', 'store']).sum()
plt.figure(figsize=(15,10))
d=c.unstack()
d.plot(y='sales')
#we check seasonal sales profile comparing stores
c=train_df.groupby(['weekday', 'store']).sum()
plt.figure(figsize=(15,10))
d=c.unstack()
d.plot(y='sales')
#we evaluate increase in anual sales at itemstore level
b =train_df.drop(columns=['store', 'item','weekday','date','month'])
c=b.groupby(['year', 'itemstore']).sum()
d=c.unstack()
sales_itemstore_year=d.T
sales_itemstore_year['delta_2014/2013']=((sales_itemstore_year[2014]-sales_itemstore_year[2013])/sales_itemstore_year[2013])*100
sales_itemstore_year['delta_2015/2014']=((sales_itemstore_year[2015]-sales_itemstore_year[2014])/sales_itemstore_year[2014])*100
sales_itemstore_year['delta_2016/2015']=((sales_itemstore_year[2016]-sales_itemstore_year[2015])/sales_itemstore_year[2015])*100
sales_itemstore_year['delta_2017/2016']=((sales_itemstore_year[2017]-sales_itemstore_year[2016])/sales_itemstore_year[2016])*100
sales_itemstore_year_deltas =sales_itemstore_year.drop(columns=[2013, 2014, 2015, 2016, 2017], axis=1)
sales_itemstore_year_deltas =sales_itemstore_year.drop(columns=[2013, 2014, 2015, 2016, 2017], axis=1)
#heat-maps to compare deltas anual and bet. itemstore each year
sales_itemstore_year_deltas=sales_itemstore_year_deltas.sort_values('delta_2014/2013')
plt.figure(figsize=(8,10))
sns.heatmap(sales_itemstore_year_deltas)
plt.title("Percentage variation sales-itemstore. Sort 2014/2013", fontsize=15)
plt.show()
sales_itemstore_year_deltas=sales_itemstore_year_deltas.sort_values('delta_2017/2016')
plt.figure(figsize=(8,10))
sns.heatmap(sales_itemstore_year_deltas)
plt.title("Percentage variation sales-itemstore. Sort 2017/2016", fontsize=15)
plt.show()
#we pivot, group to weeks
train_df['date'] = pd.to_datetime(train_df['date'])
train_df_train=train_df.pivot(index='date', columns='itemstore', values='sales')
train_df_train=train_df_train.resample('W').sum()
train_df_train = train_df_train[:-1]
train_df_train_V1 = train_df_train

# we search ARIMA parameters for item 1 store 1 with 52 weeks differentation for stationary hipotesis
import warnings
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas import tseries

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)

def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order):
# prepare training dataset
    X = X.astype('float32')
    train_size = int(len(X) * 0.7)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
    # difference data
        weeks_in_year = 52
        diff = difference(history, weeks_in_year)
        model = ARIMA(diff, order=arima_order)
        model_fit = model.fit(trend='nc', disp=0)
        yhat = model_fit.forecast()[0]
        yhat = inverse_difference(history, yhat, weeks_in_year)
        predictions.append(yhat)
        history.append(test[t])
        # calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
#evaluate models
p_values = range(0, 6)
d_values = range(0, 2)
q_values = range(0, 6)
t = '1-1'

warnings.filterwarnings("ignore")

evaluate_models(train_df_train_V1[t].values, p_values, d_values, q_values)
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from sklearn.metrics import mean_squared_error
from math import sqrt

#Procedure to predict values
def do_predictions_join_unpack(t, database1, database2, dictionary, database3):
    X = database1[itemstore[t]].values
    X = X.astype('float32')
    weeks_in_year = 52
    diff = difference(X, weeks_in_year)
    model = ARIMA(diff, order=(1,0,1))
    model_fit = model.fit(trend='nc', disp=0)
    # bias constant, could be calculated from in-sample mean residual
    bias = 0
    # save model
    #model_fit.save('model.pkl')
    #np.save('model_bias.npy', [bias])    
    # load and prepare datasets
    X = database1[itemstore[t]].values.astype('float32')
    history = [x for x in X]
    weeks_in_year = 52
    y = database2[itemstore[t]].values.astype('float32')
    # load model
    #model_fit = ARIMAResults.load('model.pkl')
    #bias = np.load('model_bias.npy')
    # forecast 13 periods
    predictions = list()
    forecast = model_fit.forecast(steps=13)[0]
    for yhat in forecast:
        yhat = bias + inverse_difference(history, yhat, weeks_in_year)
        history.append(yhat)
        predictions.append(yhat)
    #turn to daily with weekly pattern and copy in summary
    database2 = database2.reset_index()
    predictions = pd.DataFrame(predictions)
    train_df_test_V1_pred = pd.concat([database2['date'], database2[itemstore[t]], predictions], axis=1)
    train_df_test_V1_pred['date'] = pd.to_datetime(train_df_test_V1_pred['date'])
    train_df_test_V1_pred=train_df_test_V1_pred.set_index('date')
    new_dates = pd.date_range('2018-01-01', '2018-04-01', name='date')
    train_df_test_V1_pred_daily = train_df_test_V1_pred.reindex(new_dates, method='ffill')
    for k in range (13):
        for j in range (7):
            train_df_test_V1_pred_daily[0][(k*7)+j] = round(train_df_test_V1_pred_daily[0][(k*7)+j]*dictionary[itemstore[t]][j])
    database3[[itemstore[t]]] = train_df_test_V1_pred_daily[0]
    return database3, train_df_test_V1_pred_daily, predictions, train_df_test_V1_pred
train_df = train_df.set_index('date')
#we asign in a dictionary for each item-store the de-composition of sales for SUN-MON-TUE.......-SAT-SUMA. we use 2017 weekly pattern
dictionary_week_sales_itemstore={}
dictionary_week_sales_itemstore_reparto={}
for i in range (len(itemstore)):
    dictionary_week_sales_itemstore.update({itemstore[i]:[0, 0, 0, 0, 0, 0, 0, 0]})
    dictionary_week_sales_itemstore_reparto.update({itemstore[i]:[0, 0, 0, 0, 0, 0, 0, 0]})

#Now we group sales at item-store level and week-day    
#train_df=train_df.set_index('date')
train_sales_weekday=train_df['01-01-2013':'31-12-2017'].groupby(['weekday', 'itemstore']).sum()
#def update_dictionary_week_sales_itemstore(itemstore, train_sales_weekday)
for i in range (len(itemstore)):
    for j in range (0,7):
        dictionary_week_sales_itemstore[itemstore[i]][j]= train_sales_weekday.loc[(j, itemstore[i]),['sales']][0]
    dictionary_week_sales_itemstore[itemstore[i]][7]= sum(dictionary_week_sales_itemstore[itemstore[i]][0:7])   
    
#Now we update second dictionary dictionary_week_sales_itemstore_reparto={}
for i in range (len(itemstore)):
    for j in range (0,7):
        dictionary_week_sales_itemstore_reparto[itemstore[i]][j]= (dictionary_week_sales_itemstore[itemstore[i]][j]/   \
            dictionary_week_sales_itemstore[itemstore[i]][7])
#we prepare dataframe for integrate all results
test_df['itemstore']=test_df.item.astype(str)+"-"+test_df.store.astype(str)
test_df['sales'] = 0
train_df_test_V2  = test_df.drop(columns=['store', 'item'])
train_df_test_V2['date'] = pd.to_datetime(train_df_test_V2['date'])
train_df_test_V2 = train_df_test_V2.pivot(index='date', columns='itemstore', values='sales')
train_df_test_V1 = train_df_test_V2.resample('W').sum()
#calculation of all itemstore predictions
predictions = list()
for t in range (len(itemstore)):
        do_predictions_join_unpack(t, train_df_train_V1, train_df_test_V1, dictionary_week_sales_itemstore_reparto, train_df_test_V2)
    
#copy same pattern for first week
train_df_test_V2.ix['2018-01-01']=train_df_test_V2.ix['2018-01-08']
train_df_test_V2.ix['2018-01-02']=train_df_test_V2.ix['2018-01-09']
train_df_test_V2.ix['2018-01-03']=train_df_test_V2.ix['2018-01-10']
train_df_test_V2.ix['2018-01-04']=train_df_test_V2.ix['2018-01-11']
train_df_test_V2.ix['2018-01-05']=train_df_test_V2.ix['2018-01-12']
train_df_test_V2.ix['2018-01-06']=train_df_test_V2.ix['2018-01-13']

for i in range (len(test_df)):
    test_df['sales'][(i)] = train_df_test_V2.loc[test_df['date'][(i)], test_df['itemstore'][(i)]]
submission = test_df.drop(columns=['date', 'store', 'item', 'itemstore'])
submission_1= submission.reset_index()
submission_1.to_csv('submissionFCTl.csv', index=False)