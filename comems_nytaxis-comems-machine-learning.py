import os

import warnings

warnings.filterwarnings("ignore")



import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

%matplotlib inline
TRAINPATH = '../input/ny-taxis/train_w_zones2.csv'

TESTPATH = '../input/ny-taxis/test_w_zones2.csv'

df_train = pd.read_csv(TRAINPATH)

df_test = pd.read_csv(TESTPATH)

print('df_train:', df_train.shape, '\ndf_test:', df_test.shape)

l=df_train.shape[0]

df_train.head()
# remove trip of less than 10m (#8665)

#print('There is ', df_train[df_train['distances']<=0.01].shape[0], 'travels of less than 100m before filtering')

#df_train = df_train[df_train['distances']>0.01]

#print('There is ', df_train[df_train['distances']<=0.01].shape[0], 'travels of less than 100m after filtering')
#remove trips that lasted less than 1 min (#4933 left after previous filtering)

#print('There is', df_train[df_train['trip_duration']<=1*60].shape[0], 'travels of less than 1min before filtering')

#df_train = df_train[df_train['trip_duration']>1*60]

#print('There is', df_train[df_train['trip_duration']<=1*60].shape[0], 'travels of less than 1min after filtering')
#remove trip with an average speed greater than 200 km/h (distances are in straigth lines, I could probably choose a smaller number) (#22 after the two filters)

#print('There is', df_train[df_train['distances']/(df_train['trip_duration'])>=200/3600].shape[0], 'travels with an average speed faster than 200km/h before filtering')

#df_train = df_train[df_train['distances']/(df_train['trip_duration'])<200/3600]

#print('There is', df_train[df_train['distances']/(df_train['trip_duration'])>=200/3600].shape[0], 'travels with an average speed faster than 200km/h after filtering')
# remove trips that took longer that 3 hours (Who does that ??) (#2101 after filtering)

#print('There is', df_train[df_train['trip_duration']>=3*3600].shape[0], 'travels that took longer than 3 hours before filtering')

#df_train = df_train[df_train['trip_duration']<3*3600]

#print('There is', df_train[df_train['trip_duration']>=3*3600].shape[0], 'travels that took longer than 3 hours after filtering')
#remove trip with an average speed less than 1 km/h (I could probably choose a bigger number) (#3233 after the filters)

#print('There is', df_train[df_train['distances']/(df_train['trip_duration'])<=1/3600].shape[0], 'travels with an average speed slower than 1km/h before filtering (you walk at ~5km/h)')

#df_train = df_train[df_train['distances']/(df_train['trip_duration'])>1/3600]

#print('There is', df_train[df_train['distances']/(df_train['trip_duration'])<=1/3600].shape[0], 'travels with an average speed slower than 1km/h after filtering')
#print('We filtered','{:.3}'.format((l-df_train.shape[0])/l*100), '% of the dataset' )
def featuresSelection(df_in):

    VARS_CAT = [ 'store_and_fwd_flag' ]

    VARS_NUM = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'zone', 'distances', 'pickup_Month', 'pickup_Hour', 'pickup_Weekend', 'passenger_count', 'vendor_id' ]

    vars_cat = VARS_CAT

    vars_num = VARS_NUM



    X=df_in.loc[:, vars_cat + vars_num]



    for cat in vars_cat:

        X[cat] = X[cat].astype('category').cat.codes



    return X
X_train = featuresSelection(df_train)

target = 'trip_duration'

y_train = df_train.loc[:, target]

print(X_train.shape, y_train.shape)

y_train = np.log1p( y_train )

X_train.head()
from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import SGDRegressor

from sklearn.metrics import mean_squared_log_error as MSLE

import xgboost
X_train_sample, X_validation, y_train_sample, y_validation = train_test_split(X_train, y_train, test_size=.2, random_state=42 )

print(X_train_sample.shape, y_train_sample.shape , X_validation.shape, y_validation.shape)

X_train_sample.head(5)
#min_samples_leaf = {  1: 0.14335025261894946,  2: 0.13981831370645642,   3: 0.13852060557356807,  4: 0.1374604137021863, 5: 0.13701190316428685, 6: 0.13719592541154788,  7: 0.13647552678899308,   8: 0.13668619429239404,  9: 0.13678934918189598, 10: 0.13720206662667936, 15: 0.1378838545097919,   20: 0.13858468007164235,  25: 0.1397767624826059, 30: 0.14040835836429333, 35: 0.14162848146663448,  40: 0.14219905657487034,  45: 0.14265841548835242, 50: 0.14374664124817566, 100: 0.14924626267746,   150: 0.15302159678464494, 200: 0.15600849362124466, 250: 0.1578977545855252,  300: 0.16053779676581148, }

#plt.plot(min_samples_leaf.keys(), min_samples_leaf.values());

#plt.title('min_samples_leaf optimization');

#plt.legend(" with hyperparameters: RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features=0.4, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=9, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=-1, oob_score=False, random_state=50, verbose=0, warm_start=False)") #plt.legend(" with features: pickup_latitude	pickup_longitude	dropoff_latitude	dropoff_longitude	zone	distances	pickup_Month	pickup_Hour	pickup_Weekend	passenger_count ")

#plt.xlabel('min_samples_leaf');

#plt.ylabel('MSLE score');

#min(min_samples_leaf, key=min_samples_leaf.get)
#%%time

#modelparams = { 'booster':'gbtree', 'verbosity':1, 'max_depth':15, 'subsample': 1, 'lamda':0, 'max_delta_step':3, 'objective':'reg:linear', 'learning_rate':0.08, 'colsample_bytree':0.9, 'colsample_bylevel':0.9}

#data_train = xgboost.DMatrix(X_train_sample,y_train_sample)

#model = xgboost.train(modelparams, data_train, num_boost_round=200)
#real = list(np.expm1(y_validation))

#predicted = list(np.expm1(model.predict(xgboost.DMatrix(X_validation))))

#print('\nMean Square Log Error score:', MSLE(real, predicted))
%%time

#rf = RandomForestRegressor( n_estimators=100, min_samples_leaf=1, max_depth=None, max_features=.4, oob_score=False, bootstrap=True, n_jobs=-1 )



modelparams = { 'booster':'gbtree', 'verbosity':1, 'max_depth':15, 'subsample': 1, 'lamda':0, 'max_delta_step':3, 'objective':'reg:linear', 'learning_rate':0.08, 'colsample_bytree':0.9, 'colsample_bylevel':0.9}

data_train = xgboost.DMatrix(X_train,y_train)

xg = xgboost.train(modelparams, data_train, num_boost_round=200)
#%%time

#rf.fit( X_train, y_train );

#rf.feature_importances_
#rf1_scores=-cross_val_score( rf, X_train, y_train, cv=5, scoring='neg_mean_squared_log_error' )

#rf1_scores, np.mean(rf1_scores)
X_test = featuresSelection(df_test)

X_test.head()
#y_test_predict = model_final.predict(X_test)

#y_test_predict = np.expm1(y_test_predict)

#y_test_predict[:5]
y_test_predict = np.expm1(xg.predict(xgboost.DMatrix(X_test)))

y_test_predict[:5]
submission = pd.DataFrame(df_test.loc[:, 'id'])

submission['trip_duration']=y_test_predict

print(submission.shape)

submission.head()
submission.to_csv("submit_file.csv", index=False)