import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import datetime as dt

from sklearn import metrics

from sklearn.model_selection import train_test_split

import warnings; warnings.simplefilter('ignore')
data=pd.read_csv("../input/chh-ola/train.csv")

test=pd.read_csv("../input/chh-ola/test.csv")
data.isnull().sum()
data.columns
data.rename(columns = {'vendor+AF8-id': "vendor_id",

                        'pickup+AF8-loc': "pickup_loc",

                        'drop+AF8-loc' : "drop_loc",

                        'driver+AF8-tip': "driver_tip", 

                        'mta+AF8-tax' : "mta_tax",

                        'pickup+AF8-time' : "pickup_time",

                        'drop+AF8-time' : "drop_time", 

                        'num+AF8-passengers' : "num_passengers",

                        'toll+AF8-amount' : "toll_amount",

                        'payment+AF8-method' : "payment_method",

                        'rate+AF8-code' : "rate_code",

                        'stored+AF8-flag' : "stored_flag",

                        'extra+AF8-charges' : "extra_charges",

                        'improvement+AF8-charge' : "improvement_charge",

                        'total+AF8-amount': "total_amount"

                        }, inplace = True)
data.shape
data['pickup_time']=pd.to_datetime(data['pickup_time'])
data['drop_time']=pd.to_datetime(data['drop_time'])
data['trip_duration']=data['drop_time']-data['pickup_time']
data['trip_duration'] = data['trip_duration'].astype('timedelta64[s]')
data.drop(['pickup_time', "drop_time",'ID', "vendor_id", "drop_loc", "pickup_loc", "stored_flag", "mta_tax", "improvement_charge" ], axis = 1, inplace = True) 
data.shape
data.info()
data.head()
data.columns
data.isnull().sum()
data.isnull().sum().sum()
cat_cols = ["driver_tip", "toll_amount", "extra_charges", "total_amount"]

num_cols = [c for c in data.columns if c not in cat_cols]

print (num_cols)
for c in num_cols :

  data[c].fillna((data[c].mean()),inplace=True)
data.isnull().sum()
data.info()
for c in cat_cols :

  data[c] = pd.to_numeric(data[c], errors='coerce')
data.info()
data.isnull().sum().sum()
data.dropna(inplace= True)
data.shape
y=data['total_amount']
data.drop(["total_amount"], axis=1, inplace = True)
data.shape
from xgboost import XGBRegressor

train_x,test_x,train_y,test_y=train_test_split(data,y,test_size=.075,random_state=42)
model = XGBRegressor(max_depth=8, n_estimators = 750, learning_rate= 0.025,random_state=42)

model.fit(train_x,train_y)
model.score(train_x,train_y)
model.score(test_x,test_y)
test=pd.read_csv("../input/chh-ola/test.csv")
test['pickup_time']=pd.to_datetime(test['pickup_time'])

test['drop_time']=pd.to_datetime(test['drop_time'])

test['trip_duration']=test['drop_time']-test['pickup_time']

test['trip_duration'] = test['trip_duration'].astype('timedelta64[s]')
test.shape
test.info()
test.drop(['pickup_time', "drop_time",'ID', "vendor_id", "drop_loc", "pickup_loc", "stored_flag", "mta_tax", "improvement_charge" ], axis = 1, inplace = True)
test.shape
test.info()
predict = model.predict(test)
ID = np.arange(154235)
len(ID)
df = pd.DataFrame({"ID" : ID,

                   "total_amount" : predict}, columns = ["ID", "total_amount"]).set_index("ID")
df.head()