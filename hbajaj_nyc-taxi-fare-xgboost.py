# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import glob
import time
import datetime
from sklearn.linear_model import LinearRegression 
#import xgboost as xgb #XGBoost classifier
from sklearn.model_selection import train_test_split, cross_val_score
from math import sin, cos, sqrt, atan2, radians
from sklearn import metrics #evaluating models
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from sklearn.externals import joblib
from functools import wraps
import xgboost as xgb
import scipy
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        #print("function type: {}".format(type(function)))
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("***************************Total time running << %s >>: %s seconds" %
               (function.__name__, str(np.round((t1-t0),2)))
               )
        return result
    return function_timer

# Any results you write to the current directory are saved as output.
@fn_timer
def read_file_to_df(fileName,rows):
    # Set columns to most suitable type to optimize for memory usage
    start_time = time.time()
    traintypes = {'fare_amount': 'float32',
                  'pickup_datetime': 'str', 
                  'pickup_longitude': 'float32',
                  'pickup_latitude': 'float32',
                  'dropoff_longitude': 'float32',
                  'dropoff_latitude': 'float32',
                  'passenger_count': 'uint8'}
    
    cols = list(traintypes.keys())
    

#    if (rows> 0):
#        df = pd.read_csv(fileName, usecols=cols, dtype=traintypes,nrows=rows)
#    else:
#        df = pd.read_csv(fileName, usecols=cols, dtype=traintypes)
#    df.head()
#    print(df.shape)
#    return df

    from pathlib import Path
    
    my_file = Path(fileName + ".feather")
    if my_file.is_file():
        df = pd.read_feather(fileName + ".feather")
    else:    
        if (rows> 0):
            df = pd.read_csv(fileName, usecols=cols, dtype=traintypes)
        else:
            df = pd.read_csv(fileName, usecols=cols, dtype=traintypes)
        #df = pd.read_csv("../input/train.csv", usecols=cols, dtype=traintypes)
        df.to_feather(fileName + ".feather")
        #df.head()
    print("--- Read Files: %s secs ---" % np.round((time.time() - start_time),2))
    return df
def read_test_file_to_df(fileName):
    df = pd.read_csv(fileName)
    return df



def drop_rows_with_nan(df):
    print("Before dropna")
    df.dropna(how = 'any', axis = 'rows',inplace=True)
    print(df.shape)

@fn_timer
def data_cleanup(tmp_df):
    print("Before clearing outliers")
    tmp_df = tmp_df[tmp_df['fare_amount'] > 0]
    tmp_df = tmp_df[tmp_df['pickup_longitude'] < -72]
    tmp_df = tmp_df[(tmp_df['pickup_latitude'] > 40) & (tmp_df['pickup_latitude'] < 44)]
    tmp_df = tmp_df[tmp_df['dropoff_longitude'] < -72]
    tmp_df = tmp_df[(tmp_df['dropoff_latitude'] > 40) & (tmp_df['dropoff_latitude'] < 44)]
    tmp_df = tmp_df[(tmp_df['passenger_count'] > 0) & (tmp_df['passenger_count'] < 10)]
    print(tmp_df.shape)
    return tmp_df


def reformat_pickup_datetime(df):
    df['pickup_datetime'] = df['pickup_datetime'].str.slice(0, 13)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'],utc=True,format='%Y-%m-%d %H')
    df['pickup_datetime'] = df['pickup_datetime'].apply(lambda x: x.astimezone('EST'))
    return df

def add_day_of_week_feature(tmp_df):
    tmp_df['dayOfWeek'] = tmp_df['pickup_datetime'].dt.dayofweek.astype('uint8')


def add_time_of_day_feature(tmp_df):
    #val = tmp_df['pickup_datetime'].dt.hour + (1 if tmp_df['pickup_datetime'].dt.minute > 30 else 0)
    #tmp_df['timeOfDay'] = tmp_df['pickup_datetime'].dt.hour.astype('uint8')
    #val = 0 if val > 23 else val
    tmp_df['timeOfDay'] = tmp_df['pickup_datetime'].dt.hour.astype('uint8')

def add_month_feature(tmp_df):
    #tmp_df['month'] = (pd.to_datetime(tmp_df['pickup_datetime'],utc=True,format='%Y-%m-%d %H')).dt.month
    tmp_df['month'] = tmp_df['pickup_datetime'].dt.month.astype('uint8')

def add_week_of_year_feature(tmp_df):
    tmp_df['weekOfYear'] = tmp_df['pickup_datetime'].dt.weekofyear.astype('uint8')

def add_year_feature(tmp_df):
    tmp_df['year'] = tmp_df['pickup_datetime'].dt.year.astype('uint16')


def distance_between_two_points(row):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(row['pickup_latitude'])
    lon1 = radians(row['pickup_longitude'])
    lat2 = radians(row['dropoff_latitude'])
    lon2 = radians(row['dropoff_longitude'])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return "{0:.3f}".format(distance)

#calculate distance between two cordinates 
@fn_timer    
def add_distance_feature(tmp_df):
    start_time = time.time()
    tmp_df['distance'] = tmp_df.apply(distance_between_two_points,axis=1).apply(pd.to_numeric).astype('float32')
    #tmp_df['distance'] = tmp_df.apply(distance_between_two_points,axis=1)
    print("--- Add distance: %s secs ---" % np.round((time.time() - start_time),2))


@fn_timer
def filter_based_on_distance(df,distance):
    df = df[(df['distance'] < 100)]    
    
@fn_timer
def data_one_hot_encoding(tmp_df):
    print("Before one hot encoding")
    start_time = time.time()
    tmp_df = pd.concat([tmp_df,pd.get_dummies(tmp_df['timeOfDay'], prefix='timeOfDay')],axis=1)
    tmp_df.drop(['timeOfDay'],axis=1, inplace=True)
    
    tmp_df = pd.concat([tmp_df,pd.get_dummies(tmp_df['dayOfWeek'], prefix='dayOfWeek')],axis=1)
    tmp_df.drop(['dayOfWeek'],axis=1, inplace=True)
    
    tmp_df = pd.concat([tmp_df,pd.get_dummies(tmp_df['month'], prefix='month')],axis=1)
    tmp_df.drop(['month'],axis=1, inplace=True)
    
    #tmp_df = pd.concat([tmp_df,pd.get_dummies(tmp_df['weekOfYear'], prefix='weekOfYear')],axis=1)
    print(tmp_df.columns)
    print("--- One hot encoding: %s secs ---" % np.round((time.time() - start_time),2))
    return tmp_df

@fn_timer
def prepare_data_split(tmp_df):
    #tmp_X = tmp_df.drop(['pickup_datetime','fare_amount','key'],axis=1)
    tmp_X = tmp_df.drop(['pickup_datetime','fare_amount','key'],axis=1)
    tmp_y = tmp_df['fare_amount']
    print("Before train test split")
    tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test = train_test_split(tmp_X,tmp_y, test_size=0.20)
    print("df.shape:" + str(tmp_df.shape))
    print("tmp_X.shape:" + str(tmp_X.shape))
    print("tmp_y.shape:" + str(tmp_y.shape))
    return tmp_X,tmp_y,tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test






@fn_timer
def get_rmse(model,data,output):
    dtrain = xgb.DMatrix(data, label=output)
    y_pred = model.predict(dtrain)
    rmse = np.sqrt(metrics.mean_squared_error(y_pred, output))
    #print("rmse:" + str(rmse))
    return rmse
@fn_timer

def xgb_evaluate(max_depth, gamma, colsample_bytree):
    params = {'eval_metric': 'rmse',
              'max_depth': int(max_depth),
              'subsample': 0.8,
              'eta': 0.1,
              'gamma': gamma,
              'colsample_bytree': colsample_bytree}
    # Used around 1000 boosting rounds in the full model
    cv_result = xgb.cv(params, dtrain, num_boost_round=100, nfold=3,verbose_eval=None,early_stopping_rounds=10)    
    
    # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]

dtrain =[]
@fn_timer
def fit_xgboost_model(X_train,X_test,y_train,y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    del(X_train)
    dtest = xgb.DMatrix(X_test)
    del(X_test)
    params = {'eval_metric': 'rmse',
              'max_depth': 7,
              'subsample': 0.8,
              'eta': 0.03,
              'gamma': 1,
              'colsample_bytree': 0.9}
    print("fit_xgboost_model")
    model = xgb.train(params, dtrain, num_boost_round=1000,early_stopping_rounds=10)
    
    # Predict on testing and training set
    y_pred = model.predict(dtest)
    y_train_pred = model.predict(dtrain)
    
    # Report testing and training RMSE
    print(np.sqrt(mean_squared_error(y_test, y_pred)))
    print(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    #model.fit(X_train,y_train)
    #print("XGB Score on train:" + str(model.score(X_train,y_train)))
    #print("XGB score on test:" + str(model.score(X_test,y_test)))
    return model

@fn_timer
def bayes_optimization(X_train,X_test,y_train,y_test):
    global dtrain
    dtrain = xgb.DMatrix(X_train, label=y_train)
    del(X_train)
    dtest = xgb.DMatrix(X_test)
    del(X_test)
    print("fit_xgboost_model")
    xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3,6, 7), 
                                             'gamma': (0,0.1,1),
                                             'colsample_bytree': (0.3,0.6, 0.9)})
    # Use the expected improvement acquisition function to handle negative numbers
    # Optimally needs quite a few more initiation points and number of iterations
    xgb_bo.maximize(init_points=2, n_iter=5, acq='ei')


    
@fn_timer
def output_submission(model,df_test,test_X):
    start_time = time.time()
    test_pred = model.predict(test_X)
    print(type(test_pred))
    print("Shape of test_pred:" + str(test_pred.shape))
    test_pred = np.round(test_pred,2)
    print(df_test.shape)
    print(test_pred.shape)
    # Write the predictions to a CSV file which we can submit to the competition.
    submission = pd.DataFrame(
        {'key': df_test.key, 'fare_amount': test_pred},
        columns = ['key', 'fare_amount'])
    submission.to_csv('submission.csv', index = False)
    print("--- Output submission: %s secs ---" % np.round((time.time() - start_time),2))

@fn_timer
def output_submission_stacking(df_test,test_d):
    #files = glob.glob("./model/*.compressed") 
    test_X = xgb.DMatrix(test_d)
    rmse_df = pd.DataFrame()
    for model in models:
    #for f in files:        
        #model = joblib.load(f)
        test_pred = model.predict(test_X)
        print(type(test_pred))
        print("Shape of test_pred:" + str(test_pred.shape))
        test_pred = np.round(test_pred,2)
        rmse_df = pd.concat([rmse_df,pd.DataFrame(test_pred)],axis=1)
        print(df_test.shape)
        print(test_pred.shape)
        del model
        
        # Write the predictions to a CSV file which we can submit to the competition.
        #    submission = pd.DataFrame(
        #        {'key': df_test.key, 'fare_amount': np.round(np.array(rmse_df.mean(axis=1)),2)},
        #        columns = ['key', 'fare_amount'])
    submission = pd.DataFrame(
        {'key': df_test.key, 'fare_amount': np.round(scipy.stats.mstats.gmean(rmse_df,axis=1),2)},
        columns = ['key', 'fare_amount'])
    # scipy.stats.mstats.gmean(rmse_df,axis=1)
    submission.to_csv('submission.csv', index = False)
    

@fn_timer
def preprocess_df(df,ifTest):
    start_time = time.time()
    
    print("Preprocessing data start")
    if(ifTest == False):
        drop_rows_with_nan(df)
        df = data_cleanup(df)
    df = reformat_pickup_datetime(df)
    print("Adding day,time and month feature")
    add_day_of_week_feature(df)
    add_time_of_day_feature(df)
    add_month_feature(df)
    add_year_feature(df)
    #add_week_of_year_feature(df)
    print(df.shape)
    print(df.dtypes)
    df.head()
    print("Adding distance feature")
    add_distance_feature(df)
    #df.describe()
    print("One hot encoding features")
    #print(df.isnull().sum())
    if(ifTest == False):
        filter_based_on_distance(df,100)
    df = data_one_hot_encoding(df)    
    if(ifTest == False):
        drop_rows_with_nan(df)
    print("Preprocessing data end")
    print("--- preprocess_df: %s secs ---" % np.round((time.time() - start_time),2))
    return df    


@fn_timer
def split_and_fit_model(df):

    tmp_df = preprocess_df(df,False)
    print("Shape of tmp_df: {}".format(tmp_df.shape))
    print("Info of tmp_df: {}".format(tmp_df.info()))
    
    X,y,X_train,X_test,y_train,y_test = prepare_data_split(tmp_df)
    del tmp_df
     
    bayes_optimization(X_train,X_test,y_train,y_test)   
    #model =  fit_xgboost_model(X_train,X_test,y_train,y_test)
    #rmse = get_rmse(model,X,y)
    #print("rmse:" + str(rmse))
    del X,y,X_train,X_test,y_train,y_test
    return model,rmse        

def dump_model(model,fileName):
    joblib.dump(model, fileName + ".compressed",compress=True)




#df = read_file_to_df("../input/train.csv",0)
reader = pd.read_csv("../input/train.csv", header=0, iterator=True)
    
#print("Shape of df: {}".format(df.shape))
#print("Info of df: {}".format(df.info()))
models = []
rmses = []
pklFileName = "./model/modelFile"
for i in range(0,5):
    print("##################### Batch start num {} #####################".format(i))
    batchSize = 5_00_000
    df = reader.get_chunk(batchSize)
    startRow = batchSize * i
    endRow = startRow + batchSize
    model,rmse = split_and_fit_model(df)
    #dump_model(model,pklFileName+ str(i) )
    models.append(model)
    del model
    rmses.append(rmse)
    del df
    print("##################### Batch end #####################")
    
reader.close()    
del reader

df_test = read_test_file_to_df("../input/test.csv")
df_test = preprocess_df(df_test,True)    
test_X = df_test.drop(['pickup_datetime','key'],axis=1)
#output_submission_stacking(df_test,test_X)
del df_test
del test_X