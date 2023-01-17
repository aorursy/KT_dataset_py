import time
Acc_Start = time.time()

# Viz
import matplotlib.pyplot as plt # basic plotting
import seaborn as sns # for prettier plots


from fbprophet import Prophet

plt.rcParams['figure.figsize'] = (16, 9)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

## Load Data

Path="../input/"
#os.listdir(f'{Path}')


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# TIME SERIES
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

# Basic packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.display.max_columns = 99
import random as rd # generating random numbers
import datetime # manipulating date formats
import time
import os
import shutil
%matplotlib inline

#Data Preparation
  #loading data, don't load sample submission
data = {
    'item_cat': pd.read_csv(f'{Path}item_categories.csv'),
    'items': pd.read_csv(f'{Path}items.csv'),
    'sales_train': pd.read_csv(f'{Path}sales_train.csv'),
    'shops': pd.read_csv(f'{Path}shops.csv'),
    'test': pd.read_csv(f'{Path}test.csv'),
    }

Block_End=time.time()-Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End,2) +"secs\n")
print("Accumulate use " + "%0.2f" % round(Block_End,2) +"secs")
Block_Start=time.time()
data['sales_train']['date'] = pd.to_datetime(data['sales_train']['date'])
data['sales_train']['date'] = data['sales_train']['date'].dt.date

Block_End=time.time()-Block_Start
Acc_End=time.time()-Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End,2) +"secs\n")
print("Accumulate use " + "%0.2f" % round(Acc_End,2) +"secs")
Block_Start=time.time()
data['sales_train']['date'] = pd.to_datetime(data['sales_train']['date'])
data['sales_train']['dow'] = data['sales_train']['date'].dt.dayofweek
data['sales_train']['year'] = data['sales_train']['date'].dt.year
data['sales_train']['month'] = data['sales_train']['date'].dt.month

data['sales_train'] = pd.merge(data['sales_train'], data['items'] , how='left', on=['item_id'])
data['sales_train'].drop(['item_name'], axis=1, inplace=True)

Block_End=time.time()-Block_Start
Acc_End=time.time()-Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End,2) +"secs\n")
print("Accumulate use " + "%0.2f" % round(Acc_End,2) +"secs")
Block_Start=time.time()

# Lable encoder for categorical variables
lbl = LabelEncoder()
data['sales_train']['item_id'] = lbl.fit_transform(data['sales_train']['item_id'])
data['sales_train']['item_category_id'] = lbl.fit_transform(data['sales_train']['item_category_id'])

Block_End=time.time()-Block_Start
Acc_End=time.time()-Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End,2) +"secs\n")
print("Accumulate use " + "%0.2f" % round(Acc_End,2) +"secs")
data['sales_train'].head(20)
cols = data['sales_train'].columns.tolist()
cols
cols=['date',
     'date_block_num',
     'year',
     'month',
     'dow',
     'shop_id',
     'item_category_id',
     'item_id',
     'item_price',
     'item_cnt_day',
    ]
data['sales_train']=data['sales_train'][cols]
data['sales_train'].head()
Block_Start=time.time()
data['test'] = pd.merge(data['test'], data['items'], how='left', on=['item_id'])
data['test'].drop(['item_name'], axis=1, inplace=True)
data['test'].head(20)
Block_End=time.time()-Block_Start
Acc_End=time.time()-Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End,2) +"secs\n")
print("Accumulate use " + "%0.2f" % round(Acc_End,2) +"secs")
Block_Start=time.time()
#Split the test set not in train set
train_items = data['sales_train'].item_id.unique()
test_old= data['test'][~data['test'] .item_id.isin(train_items)]
test_items_not_in_train = data['test'][~data['test'].item_id.isin(train_items)].item_id.unique()
print('%d items in test data not found in train data' % len(test_items_not_in_train))
print("\n")
Block_End=time.time()-Block_Start
Acc_End=time.time()-Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End,2) +"secs\n")
print("Accumulate use " + "%0.2f" % round(Acc_End,2) +"secs")
test_new=data['test'][~data['test'].item_id.isin(train_items)]
test_old= data['test'][data['test'].item_id.isin(train_items)]

y=len(data['test'])
x=len(test_old)+len(test_new)
x==y #True means that we have succesfully splitted the test_set with 

#Block_Start=time.time()
import pandas_profiling
pandas_profiling.ProfileReport(data['sales_train'])
#Block_End=time.time()-Block_Start


Acc_End=time.time()-Acc_Start
#print("Execute block use " + "%0.2f" % round(Block_End,2) +"secs\n")
print("Accumulate use " + "%0.2f" % round(Acc_End,2) +"secs")
ts=data['sales_train'].groupby(["date_block_num"])["item_cnt_day"].sum()
ts.astype('float')
plt.figure(figsize=(16,8))
plt.title('Total Sales of the company')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts);
plt.figure(figsize=(16,6))
plt.plot(ts.rolling(window=12,center=False).mean(),label='Rolling Mean');
plt.plot(ts.rolling(window=12,center=False).std(),label='Rolling sd');
plt.legend();
import statsmodels.api as sm
# multiplicative
res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="multiplicative")
#plt.figure(figsize=(16,12))
fig = res.plot()
#fig.show()
# Additive model
res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="additive")
#plt.figure(figsize=(16,12))
fig = res.plot()
#fig.show()
# Stationarity tests
def test_stationarity(timeseries):
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

test_stationarity(ts)
# to remove trend
from pandas import Series as Series
# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced forecast
def inverse_difference(last_ob, value):
    return value + last_ob


ts=data['sales_train'].groupby(["date_block_num"])["item_cnt_day"].sum()
ts.astype('float')
plt.figure(figsize=(16,16))
plt.subplot(311)
plt.title('Original')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts)
plt.subplot(312)
plt.title('After De-trend')
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts=difference(ts)
plt.plot(new_ts)
plt.plot()

plt.subplot(313)
plt.title('After De-seasonalization')
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts=difference(ts,12)       # assuming the seasonality is 12 months long
plt.plot(new_ts)
plt.plot()
test_stationarity(new_ts)
proph_results = test_old.reset_index()
proph_results['item_cnt_day'] = 0
test_old.drop(['item_category_id'], axis=1, inplace=True)
test_old.head()
cols = test_old.columns.tolist()
cols
cols=['date','ID', 'shop_id', 'item_id']

test_old=test_old[cols]

test_old.head()
test_old['date'] = pd.to_datetime(test_old['date'], format="%m/%d/%Y")
test_old.set_index('date', inplace=True)
train.drop(['date_block_num','item_price'], axis=1, inplace=True)
train.head()
train.dropna(axis=1, how='all') 
test_old.dropna(axis=1, how='all') 
train['item_cnt_day'].describe()
test_old.head()
tic = time.time()

for s in proph_results['shop_id'].unique():
    for i in proph_results['item_id'].unique():
        proph_train = train.loc[(train['shop_id'] == s) & (train['item_id'] == i)].reset_index()
        proph_train.rename(columns={'date': 'ds', 'item_cnt_day': 'y'}, inplace=True)
        
        m = Prophet()
        m.fit(proph_train[['ds', 'y']])
        future = m.make_future_dataframe(periods=len(test_old.index.unique()), include_history=False)
        fcst = m.predict(future)
        
        proph_results.loc[(proph_results['shop_id'] == s) & (proph_results['item_id'] == i), 'sales'] = fcst['yhat'].values
        
        toc = time.time()
        if i % 10 == 0:
            print("Completed store {} item {}. Cumulative time: {:.1f}s".format(s, i, toc-tic))
proph_results.drop(['date', 'store', 'item'], axis=1, inplace=True)
proph_results.head()
proph_results = np.clip(proph_results,0.,20.)
proph_results.to_csv('proph_results.csv', index=False)