# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



PG1 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv')

PWS1 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
def preprocess_data(gen_data,weather_data,prevDays=3,include_previous_label=True):

  # drop unwanted columns

  #PLANT_ID in gen is 1 value

  gen_data = gen_data.drop(['PLANT_ID'], axis=1)

  #PLANT_ID is  value, SOURCE_KEY is not the same as gen1 cannot join

  weather_data = weather_data.drop(['PLANT_ID', 'SOURCE_KEY'], axis=1)



  gen_data['DATE_TIME'] = pd.to_datetime(gen_data['DATE_TIME'],format='%d-%m-%Y %H:%M')

  weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'],format='%Y-%m-%d %H:%M:%S')

  gen_data = gen_data[(6 <= gen_data['DATE_TIME'].dt.hour) & (gen_data['DATE_TIME'].dt.hour < 19)].set_index('DATE_TIME')

  weather_data = weather_data[(6 <= weather_data['DATE_TIME'].dt.hour) & (weather_data['DATE_TIME'].dt.hour < 19)].set_index('DATE_TIME')



  #Join column with DATE_TIME index

  data1 = gen_data.join(weather_data, on='DATE_TIME',how = 'inner')

  data1['SOURCE_KEY'] = data1['SOURCE_KEY'].astype('category')



  dataset = feature_eng(data1,feat=True,prevDays=prevDays,include_itself_label=include_previous_label)



  for i,j,k in list(zip(dataset.index,dataset.index.hour,dataset.index.minute)):

    dataset.at[i,'Hour_is'] = int(j)

    dataset.at[i,'Minute_is'] = int(k)



  return dataset
#Previous Day at each SOURCE_KEY

from datetime import timedelta

from datetime import datetime



def feature_eng(dataset,feat=True,prevDays=3,include_itself_label=True):

  if feat:

    prevDays = int(prevDays)

    if include_itself_label:

      cols = dataset.columns#.drop(['TOTAL_YIELD'])

    else:

      cols = dataset.columns.drop(['TOTAL_YIELD','DAILY_YIELD'])

    if prevDays>0:

      for prevDay in range(1,prevDays+1):

        delta = timedelta(days=prevDay)

        data_prev = dataset.copy()[cols]



        #Previous prevDays days

        data_prev.index = data_prev.index + delta    

        dataset = pd.merge(dataset,data_prev,how='left',on=['DATE_TIME','SOURCE_KEY'],suffixes=('','_'+str(prevDay)))

        dataset.replace(np.nan, 0,inplace=True)

    else:

      pass

  return dataset
PG1.groupby('SOURCE_KEY').count()
PG1.groupby('SOURCE_KEY').count().count()
PWS1.groupby('SOURCE_KEY').count()
def k_10folds(dataset,folds=10):

  datasets={}

  num_all = len(dataset)



  for i in range(folds):

    datasets[i] = dataset[i:num_all:folds]

  return datasets
# ตั้งค่าโมเดล Regression

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

from sklearn.linear_model import LinearRegression

import time

from sklearn.metrics import mean_squared_error,confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

RF = RandomForestRegressor(max_depth=3, random_state=0)

XGB = xgb.XGBRegressor(n_estimators=50,

                       objective ='reg:squarederror',

                       learning_rate = 0.1,

                       colsample_bytree=0.6,

                       max_depth = 3,

                       min_child_weight = 6)

LR = LinearRegression()



clf = [RF,XGB,LR]
#ฟังก์ชัน สร้าง Modelของแต่ละ fold โดยให้ผลที่แสดงเป็น root-mean-square error (RMSE)

def model(clf,X_train,y_train,X_test,y_test,i):

 

  clf.fit(X_train,y_train)

  y_pred = clf.predict(X_test)

  print('**************************************')

  print('at Fold :{}'.format(str(i)))

  print('**************************************')

  print('Train score : {}'.format(clf.score(X_train,y_train)))

  print('Test score : {}'.format(clf.score(X_test,y_test)))

  print('Root Mean Square Error : {}'.format(mean_squared_error(y_test,y_pred)**0.5))

  return clf,y_pred,clf.score(X_train,y_train),clf.score(X_test,y_test),mean_squared_error(y_test,y_pred)**0.5
#Random Forest

folds = 10

prevDays = 3

include_previous_label = True

dataset_xdays = preprocess_data(PG1,PWS1,prevDays=prevDays,include_previous_label=include_previous_label)

new_dataset_xdays = k_10folds(dataset_xdays,folds=folds)

clfs = [RF,XGB,LR]

train_scores = []

test_scores = []

rmse_scores = []

compare_scores = []



for clf in [clfs[0]]:

  print('**************************************')

  print('Model : {}'.format(type(clf).__name__))

  print('**************************************')



  for fold in range(folds):

    train_data = pd.DataFrame()

    for j in range(folds):

      if j==fold:

        pass

      else:

        train_data = pd.concat([train_data,new_dataset_xdays[j]])

    



    test_data = new_dataset_xdays[fold]

    

    X_train = train_data.drop(['SOURCE_KEY','TOTAL_YIELD','DAILY_YIELD'],axis=1)

    y_train = train_data['DAILY_YIELD']



    print('Train size : {}'.format(X_train.shape))

    X_test = test_data.drop(['SOURCE_KEY','TOTAL_YIELD','DAILY_YIELD'],axis=1)

    X_test_total = test_data['TOTAL_YIELD_1']

    y_test = test_data['DAILY_YIELD']

  

    print('Test size : {}'.format(X_test.shape))

    model_plant1,y_pred,train_score,test_score,rmse_score = model(clf,X_train,y_train,X_test,y_test,fold)

    train_scores.append(train_score)

    test_scores.append(test_score)

    rmse_scores.append(rmse_score)

  print('**************************************')

  print('Average Train score: {}'.format(np.mean(train_scores)))

  print('Average Test score: {}'.format(np.mean(test_scores)))

  print('Average RMSE score: {}'.format(np.mean(rmse_scores)))

  compare_scores.append([type(clf).__name__,np.mean(train_scores),np.mean(test_scores),np.mean(rmse_scores)])
plt.plot(y_test[:800].values)

plt.plot(y_pred[:800])

plt.legend(['Actual Value','Prediction'])

plt.title('Random Forest predictor')
#XGboost

folds = 10

prevDays = 3

include_previous_label = True

dataset_xdays = preprocess_data(PG1,PWS1,prevDays=prevDays,include_previous_label=include_previous_label)

new_dataset_xdays = k_10folds(dataset_xdays,folds=folds)

clfs = [RF,XGB,LR]

train_scores = []

test_scores = []

rmse_scores = []

#compare_scores = []



for clf in [clfs[1]]:

  print('**************************************')

  print('Model : {}'.format(type(clf).__name__))

  print('**************************************')



  for fold in range(folds):

    train_data = pd.DataFrame()

    for j in range(folds):

      if j==fold:

        pass

      else:

        train_data = pd.concat([train_data,new_dataset_xdays[j]])

    



    test_data = new_dataset_xdays[fold]

    

    X_train = train_data.drop(['SOURCE_KEY','TOTAL_YIELD','DAILY_YIELD'],axis=1)

    y_train = train_data['DAILY_YIELD']



    print('Train size : {}'.format(X_train.shape))

    X_test = test_data.drop(['SOURCE_KEY','TOTAL_YIELD','DAILY_YIELD'],axis=1)

    X_test_total = test_data['TOTAL_YIELD_1']

    y_test = test_data['DAILY_YIELD']

  

    print('Test size : {}'.format(X_test.shape))

    model_plant1,y_pred,train_score,test_score,rmse_score = model(clf,X_train,y_train,X_test,y_test,fold)

    train_scores.append(train_score)

    test_scores.append(test_score)

    rmse_scores.append(rmse_score)

  print('**************************************')

  print('Average Train score: {}'.format(np.mean(train_scores)))

  print('Average Test score: {}'.format(np.mean(test_scores)))

  print('Average RMSE score: {}'.format(np.mean(rmse_scores)))

  compare_scores.append([type(clf).__name__,np.mean(train_scores),np.mean(test_scores),np.mean(rmse_scores)])
plt.plot(y_test[:800].values)

plt.plot(y_pred[:800])

plt.legend(['Actual Value','Prediction'])

plt.title('XGB predictor')
#LinearRegression



folds = 10

prevDays = 3

include_previous_label = True

dataset_xdays = preprocess_data(PG1,PWS1,prevDays=prevDays,include_previous_label=include_previous_label)

new_dataset_xdays = k_10folds(dataset_xdays,folds=folds)

clfs = [RF,XGB,LR]

train_scores = []

test_scores = []

rmse_scores = []

#compare_scores = []



for clf in [clfs[2]]:

  print('**************************************')

  print('Model : {}'.format(type(clf).__name__))

  print('**************************************')



  for fold in range(folds):

    train_data = pd.DataFrame()

    for j in range(folds):

      if j==fold:

        pass

      else:

        train_data = pd.concat([train_data,new_dataset_xdays[j]])

    



    test_data = new_dataset_xdays[fold]

    

    X_train = train_data.drop(['SOURCE_KEY','TOTAL_YIELD','DAILY_YIELD'],axis=1)

    y_train = train_data['DAILY_YIELD']



    print('Train size : {}'.format(X_train.shape))

    X_test = test_data.drop(['SOURCE_KEY','TOTAL_YIELD','DAILY_YIELD'],axis=1)

    X_test_total = test_data['TOTAL_YIELD_1']

    y_test = test_data['DAILY_YIELD']

  

    print('Test size : {}'.format(X_test.shape))

    model_plant1,y_pred,train_score,test_score,rmse_score = model(clf,X_train,y_train,X_test,y_test,fold)

    train_scores.append(train_score)

    test_scores.append(test_score)

    rmse_scores.append(rmse_score)

  print('**************************************')

  print('Average Train score: {}'.format(np.mean(train_scores)))

  print('Average Test score: {}'.format(np.mean(test_scores)))

  print('Average RMSE score: {}'.format(np.mean(rmse_scores)))

  compare_scores.append([type(clf).__name__,np.mean(train_scores),np.mean(test_scores),np.mean(rmse_scores)])
plt.plot(y_test[:800].values)

plt.plot(y_pred[:800])

plt.legend(['Actual Value','Prediction'])

plt.title('LinearRegression predictor')
pd.DataFrame(compare_scores[0:3],columns=['Predictor','Training score','Testing score','RMSE score'])
#Random Forest



folds = 10

prevDays = 7

include_previous_label = True

dataset_xdays = preprocess_data(PG1,PWS1,prevDays=prevDays,include_previous_label=include_previous_label)

new_dataset_xdays = k_10folds(dataset_xdays,folds=folds)

clfs = [RF,XGB,LR]

train_scores = []

test_scores = []

rmse_scores = []

compare_scores = []



for clf in [clfs[0]]:

  print('**************************************')

  print('Model : {}'.format(type(clf).__name__))

  print('**************************************')



  for fold in range(folds):

    train_data = pd.DataFrame()

    for j in range(folds):

      if j==fold:

        pass

      else:

        train_data = pd.concat([train_data,new_dataset_xdays[j]])

    



    test_data = new_dataset_xdays[fold]

    

    X_train = train_data.drop(['SOURCE_KEY','TOTAL_YIELD','DAILY_YIELD'],axis=1)

    y_train = train_data['DAILY_YIELD']



    print('Train size : {}'.format(X_train.shape))

    X_test = test_data.drop(['SOURCE_KEY','TOTAL_YIELD','DAILY_YIELD'],axis=1)

    X_test_total = test_data['TOTAL_YIELD_1']

    y_test = test_data['DAILY_YIELD']

  

    print('Test size : {}'.format(X_test.shape))

    model_plant1,y_pred,train_score,test_score,rmse_score = model(clf,X_train,y_train,X_test,y_test,fold)

    train_scores.append(train_score)

    test_scores.append(test_score)

    rmse_scores.append(rmse_score)

  print('**************************************')

  print('Average Train score: {}'.format(np.mean(train_scores)))

  print('Average Test score: {}'.format(np.mean(test_scores)))

  print('Average RMSE score: {}'.format(np.mean(rmse_scores)))

  compare_scores.append([type(clf).__name__,np.mean(train_scores),np.mean(test_scores),np.mean(rmse_scores)])
#XGboost

folds = 10

prevDays = 7

include_previous_label = True

dataset_xdays = preprocess_data(PG1,PWS1,prevDays=prevDays,include_previous_label=include_previous_label)

new_dataset_xdays = k_10folds(dataset_xdays,folds=folds)

clfs = [RF,XGB,LR]

train_scores = []

test_scores = []

rmse_scores = []

compare_scores = []



for clf in [clfs[1]]:

  print('**************************************')

  print('Model : {}'.format(type(clf).__name__))

  print('**************************************')



  for fold in range(folds):

    train_data = pd.DataFrame()

    for j in range(folds):

      if j==fold:

        pass

      else:

        train_data = pd.concat([train_data,new_dataset_xdays[j]])

    



    test_data = new_dataset_xdays[fold]

    

    X_train = train_data.drop(['SOURCE_KEY','TOTAL_YIELD','DAILY_YIELD'],axis=1)

    y_train = train_data['DAILY_YIELD']



    print('Train size : {}'.format(X_train.shape))

    X_test = test_data.drop(['SOURCE_KEY','TOTAL_YIELD','DAILY_YIELD'],axis=1)

    X_test_total = test_data['TOTAL_YIELD_1']

    y_test = test_data['DAILY_YIELD']

  

    print('Test size : {}'.format(X_test.shape))

    model_plant1,y_pred,train_score,test_score,rmse_score = model(clf,X_train,y_train,X_test,y_test,fold)

    train_scores.append(train_score)

    test_scores.append(test_score)

    rmse_scores.append(rmse_score)

  print('**************************************')

  print('Average Train score: {}'.format(np.mean(train_scores)))

  print('Average Test score: {}'.format(np.mean(test_scores)))

  print('Average RMSE score: {}'.format(np.mean(rmse_scores)))

  compare_scores.append([type(clf).__name__,np.mean(train_scores),np.mean(test_scores),np.mean(rmse_scores)])
#LinearRegression

#XGboost

folds = 10

prevDays = 7

include_previous_label = True

dataset_xdays = preprocess_data(PG1,PWS1,prevDays=prevDays,include_previous_label=include_previous_label)

new_dataset_xdays = k_10folds(dataset_xdays,folds=folds)

clfs = [RF,XGB,LR]

train_scores = []

test_scores = []

rmse_scores = []

compare_scores = []



for clf in [clfs[2]]:

  print('**************************************')

  print('Model : {}'.format(type(clf).__name__))

  print('**************************************')



  for fold in range(folds):

    train_data = pd.DataFrame()

    for j in range(folds):

      if j==fold:

        pass

      else:

        train_data = pd.concat([train_data,new_dataset_xdays[j]])

    



    test_data = new_dataset_xdays[fold]

    

    X_train = train_data.drop(['SOURCE_KEY','TOTAL_YIELD','DAILY_YIELD'],axis=1)

    y_train = train_data['DAILY_YIELD']



    print('Train size : {}'.format(X_train.shape))

    X_test = test_data.drop(['SOURCE_KEY','TOTAL_YIELD','DAILY_YIELD'],axis=1)

    X_test_total = test_data['TOTAL_YIELD_1']

    y_test = test_data['DAILY_YIELD']

  

    print('Test size : {}'.format(X_test.shape))

    model_plant1,y_pred,train_score,test_score,rmse_score = model(clf,X_train,y_train,X_test,y_test,fold)

    train_scores.append(train_score)

    test_scores.append(test_score)

    rmse_scores.append(rmse_score)

  print('**************************************')

  print('Average Train score: {}'.format(np.mean(train_scores)))

  print('Average Test score: {}'.format(np.mean(test_scores)))

  print('Average RMSE score: {}'.format(np.mean(rmse_scores)))

  compare_scores.append([type(clf).__name__,np.mean(train_scores),np.mean(test_scores),np.mean(rmse_scores)])
