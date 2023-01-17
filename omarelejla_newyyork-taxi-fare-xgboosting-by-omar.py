# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor

import pandas as pd 

import matplotlib.pyplot as plt







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#loading dataset

train_iop_path='/kaggle/input/new-york-city-taxi-fare-prediction/train.csv'

test_iop_path='/kaggle/input/new-york-city-taxi-fare-prediction/test.csv'

dataset_train=pd.read_csv(train_iop_path, nrows=1000000, index_col='key')

dataset_test=pd.read_csv(test_iop_path, nrows=1000000, index_col='key')



print("dataset_test old size", len(dataset_test))



dataset_test = dataset_test[dataset_test.dropoff_longitude != 0]

print("new size", len(dataset_test))

dataset_test.head(50)
print("old size", len(dataset_train))



dataset_train = dataset_train[dataset_train.dropoff_longitude != 0]

print("new size", len(dataset_train))
from datetime import datetime as dt

import warnings

warnings.filterwarnings("ignore")

def preparedataset(datasetname):

    datasetname['pickup_year']=0

    datasetname['pickup_month']=0

    datasetname['pickup_day']=0

    datasetname['pickup_hour']=0

  #  datasetname['pickup_minute']=0

 #   datasetname['pickup_second']=0

    datasetname['dis'] =0

    datasetname['x_dis']=0

    datasetname['y_dis']=0

    

    datasetname.head()

#print(datetime.strptime(df['pickup_datetime'][0].replace("UTC",''),"%Y-%m-%d %H:%M:%S "))



    for k in range(len(datasetname.index)):

        datetime=dt.strptime(datasetname['pickup_datetime'][k].replace("UTC",''),"%Y-%m-%d %H:%M:%S ")

        datasetname['pickup_year'][k]=datetime.year

        datasetname['pickup_month'][k]=datetime.month

        datasetname['pickup_day'][k]=datetime.day

        datasetname['pickup_hour'][k]=datetime.hour

      # datasetname['pickup_minute'][k]=datetime.minute

      # datasetname['pickup_second'][k]=datetime.second

        

    datasetname['x_dis'] = (datasetname['dropoff_longitude'] - datasetname['pickup_longitude'])  

    datasetname['y_dis'] = (datasetname['dropoff_latitude'] - datasetname['pickup_latitude']) 

    datasetname['dis'] = ((datasetname['dropoff_longitude'] - datasetname['pickup_longitude'])**2 + (datasetname['dropoff_latitude']-datasetname['pickup_latitude'])**2)**.5 

    datasetname=datasetname.drop(['pickup_datetime'],axis=1)

    datasetname=datasetname.drop(['pickup_longitude'],axis=1)

    datasetname=datasetname.drop(['dropoff_latitude'],axis=1)

    datasetname=datasetname.drop(['dropoff_longitude'],axis=1)

    datasetname=datasetname.drop(['pickup_latitude'],axis=1)



    return datasetname





df=preparedataset(dataset_train)

df.head(50)

df.to_csv('train_preprocessed.csv')
test_df=preparedataset(dataset_test)

test_df.head(50)
test_df.to_csv('test_preprocessed.csv')
#spliting the dataset 

from sklearn.model_selection import train_test_split

y = df.fare_amount

X=df.drop('fare_amount',axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y)
#applying XCbossting 

import warnings

warnings.filterwarnings("ignore")

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

result={}

best_istemator=0

best_learing_rate=0

best_mae=100000000000

for lr in [X / 100 for X in range(10,50, 5)]:

    for ns in range(200,701,50):

        #print("n_estimators",ns)

        #print("learning_rate",lr)

        my_model = XGBRegressor(n_estimators=ns, learning_rate=lr,n_jobs=4)

        my_model.fit(X_train, y_train)



        predictions = my_model.predict(X_valid)



        mae=mean_absolute_error(predictions, y_valid)

        if(mae < best_mae):

            best_mae=mae

            best_istemator=ns

            best_learing_rate=lr

            print("better found")

            print(ns , lr, mae)

        result[(ns,lr)]=mae

my_model_2 = XGBRegressor(n_estimators=best_istemator, learning_rate=best_learing_rate, n_jobs=4)

my_model_2.fit(X_train,y_train)

predictions_2 = my_model_2.predict(X_valid)

mae_2 = mean_absolute_error( y_valid, predictions_2) 

# Uncomment to print MAE

print("best_istemator:" , best_istemator)

print("best_learing_rate:" , best_learing_rate)

print("Mean Absolute Error:" , mae_2)

from xgboost import XGBRegressor

my_model_2 = XGBRegressor(n_estimators=700, learning_rate=0.2, n_jobs=4)

my_model_2.fit(X_train,y_train)



test_preds = my_model_2.predict(test_df)



output = pd.DataFrame({'key': test_df.index,

                      'fare_amount': test_preds})

output.to_csv('submission.csv', index=False)