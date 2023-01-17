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
df.head()
df.tail()
df.to_csv('train_prepared.csv')
test_df=preparedataset(dataset_test)
test_df.head()
test_df.tail()
test_df.to_csv('test_prepared.csv')
n_esitmators = list(range(100, 1001, 100))

print('n_esitmators', n_esitmators)

learning_rates = [x / 100 for x in range(5, 101, 5)]

print('learning_rates', learning_rates)
parameters = [{'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 

                     'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 

                                       0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

                    }]
#spliting the dataset 

from sklearn.model_selection import train_test_split

y_train = df.fare_amount

X_train =df.drop('fare_amount',axis=1)
from sklearn.model_selection import GridSearchCV

gsearch = GridSearchCV(estimator=XGBRegressor(),

                       param_grid = parameters, 

                       scoring='neg_mean_absolute_error',

                       n_jobs=4,cv=5)



gsearch.fit(X_train, y_train)
gsearch.best_params_.get('n_estimators'), gsearch.best_params_.get('learning_rate')
final_model = XGBRegressor(n_estimators=gsearch.best_params_.get('n_estimators'), 

                           learning_rate=gsearch.best_params_.get('learning_rate'), 

                           n_jobs=4)
final_model.fit(X_train, y_train)
test_preds = final_model.predict(test_df)
output = pd.DataFrame({'key': test_df.index,

                      'fare_amount': test_preds})

output.to_csv('submission.csv', index=False)

print('done')