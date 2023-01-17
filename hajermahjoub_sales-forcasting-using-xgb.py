from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import gc
import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from subprocess import check_output
print(check_output(["ls", "../input/train"]).decode("utf8"))



df=pd.read_csv('../input/train/train_Numerical_data.csv')
pred=pd.read_csv('../input/test/test.csv')
stores = pd.read_csv('../input/Stores_description.csv')


pred['Store_id'] = pred.Id.map(lambda x: (x.split('|')[0]))
pred['Item_id'] = pred.Id.map(lambda x: (x.split('|')[1]))
pred['date'] = pred.Id.map(lambda x: (x.split('|')[2]))
pred = pred.drop('Id',axis=1)
stores.head()
df = df.join(stores.set_index('Store_id'),on='Store_id')
pred = pred.join(stores.set_index('Store_id'),on='Store_id')
#Clean the df set 
#df
df = df.drop('Date',axis=1)
df['is_franchise ']=df['is_franchise '].map(lambda x:1if x else 0)
df[['Sale_space_surface']]=df[['Sale_space_surface']]/88
df[['Seating capacity']]=df[['Seating capacity']]/200
df[['Total surface']] = df[['Total surface']]/289
df[['surface_outside_tables']]=df[['surface_outside_tables']]/162
#Pred
pred['is_franchise ']= pred['is_franchise '].map(lambda x:1if x else 0)
pred[['Sale_space_surface']]=pred[['Sale_space_surface']]/88
pred[['Seating capacity']]=pred[['Seating capacity']]/200
pred[['Total surface']] = pred[['Total surface']]/289
pred[['surface_outside_tables']]=pred[['surface_outside_tables']]/162

#Split the data on Train  and test
from sklearn.model_selection import train_test_split
train,test= train_test_split(df, test_size=0.40)
train.head()
X_train= train[['Store_id','Item_id','date','City','Seating capacity','is_franchise ','Total surface','Sale_space_surface','surface_inside tables','surface_outside_tables']]
X_test= test[['Store_id','Item_id','date','City','Seating capacity','is_franchise ','Total surface','Sale_space_surface','surface_inside tables','surface_outside_tables']]
Y_train = train[['Quantity']]
Y_pred = pred[['Quantity']]
X_pred = pred[['Store_id','Item_id','date','City','Seating capacity','is_franchise ','Total surface','Sale_space_surface','surface_inside tables','surface_outside_tables']]
Y_test = test[['Quantity']]
#clean date feature
def CleanDate(df):
    df['date'] = df['date'].astype('datetime64[ns]')
    df['dayofmonth'] = df.date.dt.day
    df['dayofyear'] = df.date.dt.dayofyear
    df['dayofweek'] = df.date.dt.dayofweek
    df['month'] = df.date.dt.month
    df['year'] = df.date.dt.year
    df['weekofyear'] = df.date.dt.weekofyear
    df['is_month_start'] = (df.date.dt.is_month_start).astype(int)
    df['is_month_end'] = (df.date.dt.is_month_end).astype(int)
    #scaling date
    df[['dayofyear']]=df[['dayofyear']]/365
    df[['dayofweek']]=df[['dayofweek']]/7
    df[['month']]=df[['month']]/12
    df[['weekofyear']]=df[['weekofyear']]/52
    df[['dayofmonth']]=df[['dayofmonth']]/31
    return df
X_train = CleanDate(X_train)
X_test = CleanDate(X_test)
X_pred = CleanDate(X_pred)

X_train=X_train.replace({'year': {2016:1,2017 :2,2018:3}})
X_test=X_test.replace({'year': {2016:1,2017 :2,2018:3}})
X_pred = X_pred.replace({'year': {2016:1,2017 :2,2018:3}})
X_train = X_train.drop('date',axis=1)
X_test = X_test.drop('date',axis=1)
X_pred = X_pred.drop('date',axis=1)

#turn Item_id and Store_id to numeric feature
X_train.Store_id = X_train.Store_id.map(lambda x: int(x.split('_')[1]))
X_train.Item_id = X_train.Item_id.map(lambda x: int(x.split('_')[1]))
X_test.Store_id = X_test.Store_id.map(lambda x: int(x.split('_')[1]))
X_test.Item_id = X_test.Item_id.map(lambda x: int(x.split('_')[1]))
X_pred.Store_id= X_pred.Store_id.map(lambda x: int(x.split('_')[1]))
X_pred.Item_id = X_pred.Item_id.map(lambda x: int(x.split('_')[1]))
#training the Model "XGBRegressor"
import xgboost as xgb

reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, Y_train,
        eval_set=[(X_train, Y_train), (X_test, Y_test)],
        early_stopping_rounds=50,
       verbose=False)

test_predictions = reg.predict(X_test)
print(test_predictions)
#Error Function MSE
from sklearn.metrics import mean_absolute_error,mean_squared_error
print("Mean Absolute Error : " + str(mean_squared_error(test_predictions, Y_test)))

#predict the submition
predictions = reg.predict(X_pred)
print(predictions)

#Submission
test=pd.read_csv('../input/test/test.csv')

my_submission = pd.DataFrame({'Id': test.Id, 'Quantity': predictions})

my_submission.to_csv('submission.csv', index=False)
