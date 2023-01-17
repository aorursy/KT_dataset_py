# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_bm = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv',encoding='utf-8')

data_wtrain = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv',encoding='utf-8')

data_wtest = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv',encoding='utf-8')

data_train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv',encoding='utf-8')

data_test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv',encoding='utf-8')

sample_submission = pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv',encoding='utf-8')
data_train=data_train.head(10108050)

print("building_metadata:",data_bm.shape,'\n'"weather_train:",data_wtrain.shape,'\n'"weather_test:",data_wtest.shape,'\n'

      "train:",data_train.shape,'\n'"data_test:",data_test.shape)
data_bm['primary_use']=data_bm.primary_use.map({'Education':1,'Lodging/residential':2,'Entertainment/public assembly':3,

                                           'Public services':4,'Office':5,'Technology/science':6,'Utility':7,

                                           'Parking':8,'Other':9,'Healthcare':10,'Manufacturing/industrial':11})
data_bm["building_age"]=data_bm["year_built"]-data_bm["year_built"].min()
for column in data_bm.iloc[:,1:7]:

    plt.figure()

    data_bm.boxplot([column])

print("building_metadata:",data_bm.memory_usage().sum() / 1024**2,'\n'"weather_train:",data_wtrain.memory_usage().sum()/ 1024**2,

      '\n'"weather_test:",data_wtest.memory_usage().sum() / 1024**2,'\n'"train:",data_train.memory_usage().sum() / 1024**2,'\n'"data_test:",

      data_test.memory_usage().sum() / 1024**2)
def reduce_mem_usage(dataframe):

    start_mem_usg = dataframe.memory_usage().sum() / 1024**2 

    print("Memory usage of dataframe is :",start_mem_usg," MB")

    NAlist = [] 

    for col in dataframe.columns:

        if dataframe[col].dtype != object:

            print("******************************")

            print("Column: ",col)

            print("dtype before: ",dataframe[col].dtype)

            IsInt = False

            mx = dataframe[col].max()

            mn = dataframe[col].min()

            print("max value of a column:",mx)

            print("min value of a column:",mn)

            if not np.isfinite(dataframe[col]).all(): 

                NAlist.append(col)

                #dataframe[col].fillna(mn-1,inplace=True)

                dataframe[col].fillna(0,inplace=True)

                asint = dataframe[col].fillna(0).astype(np.int64)

                result = (dataframe[col] - asint)

                result = result.sum()

                print("result:",result)

                if result > -0.01 and result < 0.01:

                    IsInt = True

            if IsInt:

                if mn >= 0:

                    if mx < 255:

                        dataframe[col] = dataframe[col].astype(np.uint8)

                        print("in < 255 loop:",col)

                    elif mx < 65535:

                        dataframe[col] = dataframe[col].astype(np.uint16)

                        print("in < 65535 loop:",col)

                    elif mx < 4294967295:

                        dataframe[col] = dataframe[col].astype(np.uint32)

                        print("in < 4294967295 loop:",col)

                    else:

                        dataframe[col] = dataframe[col].astype(np.uint64)

                        print("in uint64 loop:",col)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        dataframe[col] = dataframe[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        dataframe[col] = dataframe[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        dataframe[col] = dataframe[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        dataframe[col] = dataframe[col].astype(np.int64)  

            else:

                dataframe[col] = dataframe[col].astype(np.float32)

            

            # Print new column type

        print("dtype after: ",dataframe[col].dtype)

        print("******************************")

            

    print("___MEMORY USAGE AFTER COMPLETION:___")

    mem_usg = dataframe.memory_usage().sum() / 1024**2 

    print("Memory usage is: ",mem_usg," MB")

    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")                                                

    return data_bm, NAlist
dataframe, NAlist = reduce_mem_usage(data_bm)

print("_________________")

print("")

print("Warning: the following columns have missing values filled with '0': ")

print("_________________")

print("")

print(NAlist)
dataframe, NAlist = reduce_mem_usage(data_wtrain)

print("_________________")

print("")

print("Warning: the following columns have missing values filled with '0': ")

print("_________________")

print("")

print(NAlist)
dataframe, NAlist = reduce_mem_usage(data_wtest)

print("_________________")

print("")

print("Warning: the following columns have missing values filled with '0': ")

print("_________________")

print("")

print(NAlist)
dataframe, NAlist = reduce_mem_usage(data_train)

print("_________________")

print("")

print("Warning: the following columns have missing values filled with ''0'': ")

print("_________________")

print("")

print(NAlist)
dataframe, NAlist = reduce_mem_usage(data_test)

print("_________________")

print("")

print("Warning: the following columns have missing values filled with ''0'': ")

print("_________________")

print("")

print(NAlist)
datamerg=pd.merge(data_train,data_bm,on='building_id',how='left')
train=pd.merge(datamerg,data_wtrain,how='left')
train["timestamp"] = pd.to_datetime(train["timestamp"])

train["hour"] = train["timestamp"].dt.hour

train["day"]=train["timestamp"].dt.day

#train["weekday_name"]=train["timestamp"].dt.weekday_name

train["weekday"]=train["timestamp"].dt.weekday

train["month"]=train["timestamp"].dt.month
train=train.drop(["timestamp"],axis=1)
train.head(10)
dataframe, NAlist = reduce_mem_usage(train)

print("_________________")

print("")

print("Warning: the following columns have missing values filled with ''0'': ")

print("_________________")

print("")

print(NAlist)
train=train[(train.square_feet>0) & (train.year_built>0) & (train.floor_count >0) & (train.day >0) & (train.month >0)

           & (train.primary_use >0)]
train.shape
def correlation_heatmap(data):

    correlation=train.corr()

    

    fig,ax=plt.subplots(figsize=(10,10))

    sns.heatmap(correlation, vmax=1.0, center=0, fmt='.2f',

                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})

    plt.show();
correlation_heatmap(train)


del data_wtrain

del data_train
#features=train.drop(['meter_reading','precip_depth_1_hr','weekday','wind_direction','wind_speed','day','hour'],axis=1)

#features=train.drop(['meter_reading','precip_depth_1_hr','weekday','wind_direction','wind_speed'],axis=1)

#features=train.drop(['meter_reading','weekday','wind_direction','wind_speed','building_id','site_id'],axis=1)

features=train.drop(['meter_reading','weekday','wind_direction','wind_speed','building_id','site_id','year_built'],axis=1)

target=train["meter_reading"]
for col in features.columns:

    if features[col].isnull().any():

       print(col)

del train
from sklearn import ensemble
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
X_train,X_test,y_train,y_test =train_test_split(features,target,test_size=0.2,random_state=0)
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 10, random_state = 42)
rf.fit(X_train,y_train)
predictions1 = rf.predict(X_test)
errors1 = abs(predictions1 - y_test)
#predictions
#y_test
print('Root Mean square Error:', np.sqrt(round(np.mean(errors1), 2)), 'degrees.')

#from sklearn.metrics import mean_squared_error 



#print(np.sqrt(np.mean((y_test-predictions1)**2)))
#from bayes_opt import BayesianOptimization
#function,parameters = rfc_optimization(2)
#regressor = train(X_train, y_train, X_test, y_test, function, parameters)
datamerg_test=pd.merge(data_test,data_bm,on='building_id',how='left')
test=pd.merge(datamerg_test,data_wtest,how='left')
#del data_bm

del data_wtest

del datamerg_test
test["timestamp"] = pd.to_datetime(test["timestamp"])

test["hour"] = test["timestamp"].dt.hour

test["day"]=test["timestamp"].dt.day

#train["weekday_name"]=train["timestamp"].dt.weekday_name

test["weekday"]=test["timestamp"].dt.weekday

test["month"]=test["timestamp"].dt.month
test=test.drop(["timestamp"],axis=1)
test['primary_use']=test.primary_use.map({'Education':1,'Lodging/residential':2,'Entertainment/public assembly':3,

                                           'Public services':4,'Office':5,'Technology/science':6,'Utility':7,

                                           'Parking':8,'Other':9,'Healthcare':10,'Manufacturing/industrial':11})
dataframe, NAlist = reduce_mem_usage(test)

print("_________________")

print("")

print("Warning: the following columns have missing values filled with ''0'': ")

print("_________________")

print("")

print(NAlist)
#test=test[(test.square_feet>0) & (test.year_built>0) & (test.floor_count >0) & (test.day >0) & (test.month >0)]
#test=test.drop(["weekday","precip_depth_1_hr",'wind_direction','wind_speed'],axis=1)

#test=test.drop(['weekday','wind_direction','wind_speed','building_id','site_id'],axis=1)

test=test.drop(['weekday','wind_direction','wind_speed','building_id','site_id','year_built'],axis=1)

test=test.drop(["row_id"],axis=1)
features.info()
test.info()
#test_pred=regressor.predict(test)
test_pred = rf.predict(test)
len(test_pred)
#sample_submission["row_id"]
output = pd.DataFrame({'row_id':sample_submission["row_id"], 

                       'meter_reading': test_pred})
output.to_csv('submission5.csv', index=False)
output