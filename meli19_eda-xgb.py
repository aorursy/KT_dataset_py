#### Necessary libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
#### Let's import our data

data = pd.read_csv('../input/CryptocoinsHistoricalPrices.csv')
#### and test if everything OK

data.head()
#### rename the first column

data= data.rename(index=str, columns={"Unnamed: 0": "id"})

data.head()
#### check for NAs in sense Pandas understands them

data.isnull().sum()
data.info()
data.describe()
### Now let's prepare lists of numeric, categorical and binary columns

### Assuming we wanted to predict the 'Close' Value

# All features

target = data['Close'].fillna(-1)

features = data.columns.tolist() 

features.remove('Close')
### set plot size

plt.rcParams["figure.figsize"] = (18,9)

plt.rc('xtick', labelsize=15) 

plt.rc('ytick', labelsize=10)
### Target variable exploration

x = range(len(target))

y = target

_ = plt.scatter(x,y,alpha=0.5);

plt.xlabel('id', fontsize=30)

plt.ylabel('close value', fontsize=30)

plt.show()
### split date

import datetime as dt

time_format = '%Y-%m-%d'



def split_date(data):

    data_time = pd.to_datetime(data.Date, format=time_format)

    data['Year']= data_time.dt.year

    data['Month'] = data_time.dt.month

    data['DayOfYear'] = data_time.dt.dayofyear

    data['DayOfMonth'] = data_time.dt.day

    

    return data



data = split_date(data)
_ = sns.barplot(x='DayOfMonth', y='Close', data=data, hue='Year')

plt.xlabel('month days', fontsize=30)

plt.ylabel('close value', fontsize=30)

plt.show()
_ = sns.pointplot(x='DayOfMonth', y='Close',hue="Year", data=data)

plt.xlabel('months', fontsize=30)

plt.ylabel('close value', fontsize=30)

plt.show()
_ = sns.barplot(x='Month', y='Close', data=data, hue='Year')

plt.xlabel('months', fontsize=30)

plt.ylabel('close value', fontsize=30)

plt.show()
_ = sns.pointplot(x='Month', y='Close',hue="Year", data=data)

plt.xlabel('months', fontsize=30)

plt.ylabel('close value', fontsize=30)

plt.show()
from sklearn.cross_validation import train_test_split

import xgboost as xgb
df_train = data.drop(['id','Date'], axis=1)

df_train.head()
df_train.info()
df_train['Volume'] = df_train['Volume'].apply(lambda x: x.replace(',',''))

df_train['Volume'] = df_train['Volume'].apply(lambda x: x.replace('0000',''))

df_train['Market.Cap'] = df_train['Market.Cap'].apply(lambda x: x.replace(',',''))

df_train['Market.Cap'] = df_train['Market.Cap'].apply(lambda x: x.replace('000000',''))
df_train.info()
df_train.head(10)
df_train['Volume'] = df_train['Volume'].convert_objects(convert_numeric=True)

df_train['Market.Cap'] = df_train['Market.Cap'].convert_objects(convert_numeric=True)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df_train['coin'] = le.fit_transform(df_train['coin'])
features = df_train.columns.tolist() 

features.remove('Close')



train = df_train



train.head(5)



X_train, X_valid = train_test_split(train, test_size=0.2, random_state=42)

y_train = (X_train.Close)

y_valid = (X_valid.Close)

dtrain = xgb.DMatrix(X_train[features], y_train)

dvalid = xgb.DMatrix(X_valid[features], y_valid)
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
num_boost_round = 20 # 5000



params = {'objective': 'reg:linear',

          'min_child_weight': 1, 

          'booster' : 'gbtree',

          'eta': 0.001,

          'alpha': 0,

          'gamma': 0,

          'max_depth': 8,

          'subsample': 0.9,

          'colsample_bytree': 0.9,

          'silent': 1,

          'seed': 1301,

#           'tree_method': 'gpu_hist',

#           'max_bin': 1000

          }



gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \

  early_stopping_rounds=2000,  

  verbose_eval=True)