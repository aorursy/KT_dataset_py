import numpy as np
import lightgbm as lgb
import pandas as pd
from kaggle.competitions import twosigmanews
import matplotlib.pyplot as plt
import random
from datetime import datetime, date
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
import time
import gc


env = twosigmanews.make_env()
#days = env.get_prediction_days()    
#for (market_obs_df, news_obs_df, predictions_template_df) in days:

#market_obs_df.head()
#news_obs_df.head()
(market_train_df, news_train_df) = env.get_training_data()
print('Done!')
def data_join(market, news):
    # Remove missing values
    #t = time.time()
    market.dropna(inplace= True)
    
    market_2013 = market[market.time.dt.year >= 2014]
    #market_2013 = market_2013[market_2013.universe == 1]
   
    news_2013 = news[news.time.dt.year >= 2014]

    #rket_train_2013 = market_train_2013[['time','assetCode','volume','close','open',
    #                       'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
    #                       'returnsOpenNextMktres10']]
    news_2013 = news_2013[['time','assetCodes','assetName',
                        'relevance', 'sentimentNegative',
                        'sentimentNeutral', 'sentimentPositive']]    
    
    news_2013['time'] = (news_2013['time'] - np.timedelta64(22,'h')).dt.ceil('1D')
    #print("Other stuff Time: ", (time.time() - t))
    #t = time.time()
    # Round time of market_train_df to 0h of curret day
    market_2013['time'] = market_2013['time'].dt.floor('1D')    
     
    #Group to get day & assetName level data 
    news_2013 = news_2013.groupby(['time','assetName']).mean().reset_index()
    
    market_2013 = pd.merge(market_2013,news_2013,
                                 how='left',on = ['assetName','time'])
    #print("Merge Time: ", (time.time() - t))
    gc.collect()
    return market_2013
t = time.time()
data = data_join(market_train_df, news_train_df)

print("Join Time: ", (time.time() - t))
del market_train_df, news_train_df 
gc.collect()
#Fill 0 for NA's in News data
data.fillna(0,inplace=True)

data.head()
def scale_var(x, xmin, xmax):
    scaled = (x - xmin)/(xmax - xmin)
    return scaled

def scale_pred(conf_data):
    pred_min = conf_data['pred'].min()
    pred_max = conf_data['pred'].max()
    conf_data['confidence'] = (2*conf_data['pred'])  - 1
    conf_data.confidence[conf_data.pred > 0] = 1
    conf_data.confidence[conf_data.pred < 0] = -1      
    return conf_data

def data_preprocessing_test(data):
    
    Labels = data[['time','assetCode']]
    #Y = data[['returnsOpenNextMktres10']]#.clip(-1, 1)
    X = data.drop(columns=['time', 'assetCode', 'assetName']).fillna(0)
    return X, Labels
    
def data_preprocessing_train(data): 
    
    #t = time.time()
       
    train = data[data.time.dt.year <= 2016]
    test = data[data.time.dt.year >= 2016]
    #labels = test[['time','assetCode']]
    Ytrain = train['returnsOpenNextMktres10']

    Xtrain = train.drop(columns=['returnsOpenNextMktres10', 'time',
                           'assetCode', 'assetName', 'universe']).fillna(0)    

    Ytest = test[['returnsOpenNextMktres10']]#.clip(-1, 1)
    Xtest = test.drop(columns=['returnsOpenNextMktres10', 'time', 
                           'assetCode', 'assetName', 'universe']).fillna(0)    
    
    #print("Pre-processing Time: ", (time.time() - t))
    return Xtrain, Ytrain, Xtest, Ytest
#data = join_market_news(market_train, news_train)
Xtrain, Ytrain, Xval, Yval = data_preprocessing_train(data) #, volume_min, 
                              #volume_max, close_min, close_max, open_min, open_max)

del data
gc.collect()
import xgboost as xgb
from sklearn.metrics import mean_squared_error

#-------------- XGboost (untuned)
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 6, alpha = 10, n_estimators = 100)

xg_reg.fit(Xtrain,Ytrain)
import matplotlib.pyplot as plt
xgb.plot_importance(xg_reg,max_num_features = 16)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()
from sklearn.metrics import mean_squared_error
from math import sqrt

pred_train = xg_reg.predict(Xtrain)
rms_train = sqrt(mean_squared_error(Ytrain, pred_train))

pred_test = xg_reg.predict(Xval)
rms_test = sqrt(mean_squared_error(Yval, pred_test))

del Xtrain, Ytrain
gc.collect()
print('Train RMSE: {0} Test RMSE: {1}'.format(rms_train,rms_test))
days = env.get_prediction_days()    
n_days = 0
prep_time = prediction_time = packaging_time = 0
#predictions_template_df = []
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days +=1
    print(n_days,end=' ')
    
    looptime = time.time()
    #t = time.time()
    jointime = time.time()
    test_data = data_join(market_obs_df, news_obs_df)
    test_data.fillna(0,inplace=True)
    #print("Join time: ", (time.time() - looptime))
    
    X_test, labels = data_preprocessing_test(test_data) #, volume_min, volume_max, close_min,
                                             #close_max, open_min, open_max)
    prep_time += time.time() - t
    
    t = time.time()
    Y_test = pd.DataFrame() 
    Y_test['pred'] = xg_reg.predict(X_test)
    Y_test = scale_pred(Y_test)
    Y_test = labels.join(Y_test)
    Y_test = Y_test[['assetCode','confidence']]
    prediction_time += time.time() -t
    
    t = time.time()    
    predictions_template_df = predictions_template_df.merge(Y_test,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    #predictions_template_df.append(Y_test) 
    env.predict(predictions_template_df)
    packaging_time += time.time() - t
    #print("Loop time: ", (time.time() - looptime))

total = prep_time + prediction_time + packaging_time
print(f'Preparing Data: {prep_time:.2f}s')
print(f'Making Predictions: {prediction_time:.2f}s')
print(f'Packing: {packaging_time:.2f}s')
print(f'Total: {total:.2f}s')
print('Done!')    

env.write_submission_file()