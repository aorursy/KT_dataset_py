import gc
import time

# Data Manipulation Packages
import numpy as np
import pandas as pd
import datetime

# Data Visualisation Packages
import matplotlib.pyplot as plt
import seaborn as sns

# Modelling Packages
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
import lightgbm as lgb

# Packages for K-means clustering
import plotly as py
import plotly.graph_objs as go
from sklearn.cluster import KMeans
train = pd.read_csv('../input/talkingdata-adtracking-fraud-detection/train.csv' ,nrows=2000000, parse_dates=['click_time','attributed_time'] )
train.head(10)
train.dtypes
features = ['ip','app','device','os', 'channel']
for i in features:
    print(i + " : " + str(train[i].nunique()))
#Splitting into Training and Validation Data
train_x , val_x = train_test_split(train , random_state = 0 , test_size = 0.2)
train_x.shape
val_x.shape
# Checking for unique values in train data
features = ['ip','app','device','os', 'channel']
for i in features:
    print(i + " : " + str(train_x[i].nunique()))
# Checking for unique values in validation data
features = ['ip','app','device','os', 'channel']
for i in features:
    print(i + " : " + str(val_x[i].nunique()))
# Creating features from click_time
train_x['cl_year'] = train['click_time'].dt.year.astype('int64')
train_x['cl_month'] = train['click_time'].dt.month.astype('int64')
train_x['cl_day'] = train['click_time'].dt.day.astype('int64')
train_x['cl_hour'] = train['click_time'].dt.hour.astype('int64')
train_x['cl_minute'] = train['click_time'].dt.minute.astype('int64')
train_x['cl_second'] = train['click_time'].dt.second.astype('int64')
train_x.head()
train_x.dtypes
# Dropping columns with zero variance
dropcl = ['cl_year','cl_month','cl_day']
train_x.drop( dropcl , axis = 1, inplace = True)
train_x.head()
# Dropping attributed_time column
train_x.drop(['attributed_time'], axis = 1, inplace = True)
train_x.head()
train_x.sort_index(inplace = True)
train_x.head()
train_x[(train_x['ip']== 45745) & (train_x['app']==3) ].channel.value_counts()
train_x['ip_chan_count'] = train_x.groupby(['ip','channel'])['is_attributed'].transform('count')
train_x['app_chan_count'] = train_x.groupby(['app','channel'])['is_attributed'].transform('count')
train_x['device_chan_count'] = train_x.groupby(['device','channel'])['is_attributed'].transform('count')
train_x['os_chan_count'] = train_x.groupby(['os','channel'])['is_attributed'].transform('count')

train_x['ip_app_chan_count'] = train_x.groupby(['ip','app','channel'])['is_attributed'].transform('count')
train_x['ip_device_chan_count'] = train_x.groupby(['ip','device','channel'])['is_attributed'].transform('count')
train_x['ip_os_chan_count'] = train_x.groupby(['ip','os','channel'])['is_attributed'].transform('count')

train_x['app_device_count'] = train_x.groupby(['app','device','channel'])['is_attributed'].transform('count')
train_x['app_os_count'] = train_x.groupby(['app','os','channel'])['is_attributed'].transform('count')
train_x.rename(columns = {'app_device_count':'app_device_chan_count','app_os_count':'app_os_chan_count'}, inplace = True) 
train_x['device_os_chan_count'] = train_x.groupby(['device','os','channel'])['is_attributed'].transform('count')

train_x.drop(['cl_hour'], axis = 1 , inplace = True)
train_y = train_x[['is_attributed']].copy()
train_x.drop(['is_attributed','click_time'], axis = 1, inplace = True)
# K-means Clustering between ip_chan_count and cl_minute
X1 = train_x[['ip_chan_count' , 'cl_minute']].iloc[: , :].values
inertia = []
for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)
algorithm = (KMeans(n_clusters = 4 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(X1)
labels1 = algorithm.labels_
centroids1 = algorithm.cluster_centers_
h = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 

# Plotting the final clustered Data
plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(Z , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

plt.scatter( x = 'ip_chan_count' ,y = 'cl_minute' , data = train_x , c = labels1 , 
            s = 200 )
plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 300 , c = 'red' , alpha = 0.5)
plt.ylabel('cl_minute') , plt.xlabel('ip_chan_count')
plt.show()

# Adding the encoded labels column to the dataframe
train_x['K-Means_encoding'] = labels1
val_x['cl_minute'] = val_x['click_time'].dt.minute.astype('int64')
val_x['cl_second'] = val_x['click_time'].dt.second.astype('int64')

val_x.drop(['attributed_time'], axis = 1, inplace = True)
val_x.sort_index(inplace = True)

val_x['ip_chan_count'] = val_x.groupby(['ip','channel'])['is_attributed'].transform('count')
val_x['app_chan_count'] = val_x.groupby(['app','channel'])['is_attributed'].transform('count')
val_x['device_chan_count'] = val_x.groupby(['device','channel'])['is_attributed'].transform('count')
val_x['os_chan_count'] = val_x.groupby(['os','channel'])['is_attributed'].transform('count')

val_x['ip_app_chan_count'] = val_x.groupby(['ip','app','channel'])['is_attributed'].transform('count')
val_x['ip_device_chan_count'] = val_x.groupby(['ip','device','channel'])['is_attributed'].transform('count')
val_x['ip_os_chan_count'] = val_x.groupby(['ip','os','channel'])['is_attributed'].transform('count')

val_x['app_device_count'] = val_x.groupby(['app','device','channel'])['is_attributed'].transform('count')
val_x['app_os_count'] = val_x.groupby(['app','os','channel'])['is_attributed'].transform('count')
val_x.rename(columns = {'app_device_count':'app_device_chan_count','app_os_count':'app_os_chan_count'}, inplace = True) 

val_x['device_os_chan_count'] = val_x.groupby(['device','os','channel'])['is_attributed'].transform('count')

val_y = val_x[['is_attributed']].copy()
val_x.drop(['is_attributed','click_time'], axis = 1, inplace = True)
# Defining a Function that will return the trained model

def lgb_modelfit_nocv(params, train_x, val_x, predictors ,train_y, val_y, objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.01,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4,
        'verbose': 0,
        'metric':metrics
    }

    lgb_params.update(params)

    print("preparing validation datasets")
    
    xgtrain = lgb.Dataset(train_x[predictors].values, label=train_y['is_attributed'].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(val_x[predictors].values, label=val_y['is_attributed'].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=feval)

    n_estimators = bst1.best_iteration
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])

    return bst1
# def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data, 
#                        model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',
#                        do_probabilities = False):
#     gs = GridSearchCV(
#         estimator=model,
#         param_grid=param_grid, 
#         cv=cv, 
#         n_jobs=-1, 
#         scoring=scoring_fit,
#         verbose=2
#     )
#     fitted_model = gs.fit(X_train_data, y_train_data)
    
#     if do_probabilities:
#       pred = fitted_model.predict_proba(X_test_data)
#     else:
#       pred = fitted_model.predict(X_test_data)
    
#     return fitted_model, pred
# model = lgb.LGBMClassifier()
# param_grid = {
#     'num_leaves': [7,15,31]  
#     'max_depth': [3,4,5]  
#     'min_child_samples': [50,100,150]  
#     'max_bin': [100,150,200]  
#     'subsample': [0.7,0.8,0.9]  
#     'colsample_bytree': [0.7,0.8,0.9] 
#     
# }

# model, pred = algorithm_pipeline(train_x, val_x, train_y, val_y, model, 
#                                  param_grid, cv=5, scoring_fit='accuracy')

# print(model.best_score_)
# print(model.best_params_)
# Calling the function that returns the trained model
predictors = ['ip','app','device','os', 'channel', 'cl_minute', 'cl_second', 
              'ip_chan_count','app_chan_count','device_chan_count','os_chan_count',
              'ip_app_chan_count','ip_device_chan_count','ip_os_chan_count','app_device_chan_count','app_os_chan_count'
             , 'device_os_chan_count']
categorical = ['ip','app', 'device', 'os', 'channel', 'cl_minute', 'cl_second']

params = {
    'learning_rate': 0.15,
    #'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 7,  # 2^max_depth - 1
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight':99 # because training data is extremely unbalanced 
}

model = lgb_modelfit_nocv(params, 
                        train_x, 
                        val_x, 
                        predictors, 
                        train_y,
                        val_y,
                        objective='binary', 
                        metrics='auc',
                        early_stopping_rounds=30, 
                        verbose_eval=True, 
                        num_boost_round=500, 
                        categorical_features=categorical)
# Reading the Test Data
test = pd.read_csv('../input/talkingdata-adtracking-fraud-detection/test.csv' ,parse_dates=['click_time'] )
test.head(10)
#test.set_index('click_id')
test.head(10)
# Feature Engineering of Test data
test['cl_minute'] = test['click_time'].dt.minute.astype('int64')
test['cl_second'] = test['click_time'].dt.second.astype('int64')


test['ip_chan_count'] = test.groupby(['ip','channel'])['click_time'].transform('count')
test['app_chan_count'] = test.groupby(['app','channel'])['click_time'].transform('count')
test['device_chan_count'] = test.groupby(['device','channel'])['click_time'].transform('count')
test['os_chan_count'] = test.groupby(['os','channel'])['click_time'].transform('count')

test['ip_app_chan_count'] = test.groupby(['ip','app','channel'])['click_time'].transform('count')
test['ip_device_chan_count'] = test.groupby(['ip','device','channel'])['click_time'].transform('count')
test['ip_os_chan_count'] = test.groupby(['ip','os','channel'])['click_time'].transform('count')

test['app_device_count'] = test.groupby(['app','device','channel'])['click_time'].transform('count')
test['app_os_count'] = test.groupby(['app','os','channel'])['click_time'].transform('count')
test.rename(columns = {'app_device_count':'app_device_chan_count','app_os_count':'app_os_chan_count'}, inplace = True) 

test['device_os_chan_count'] = test.groupby(['device','os','channel'])['click_time'].transform('count')

test.drop(['click_time'], axis = 1, inplace = True)
test.head()
test.shape
gc.collect()

# Creating the submission file
test_dummy = pd.read_csv('../input/talkingdata-adtracking-fraud-detection/test.csv' ,parse_dates=['click_time'] )
sub = pd.DataFrame()
sub['click_id'] = test_dummy['click_id'].astype('int')
del test_dummy

print("Predicting...")
sub['is_attributed'] = model.predict(test[predictors])
print("writing...")
sub.to_csv('submission1_lgb.csv',index=False)
print("done...")
# Method 2
# # Pipelining all categorical features
# categorical = ['ip','app', 'device', 'os', 'channel', 'cl_minute', 'cl_second']

# # Preparing the LightGBM Data Containers
# lgb_train_data = lgb.Dataset(train_x, label= train_y, categorical_feature=categorical)
# lgb_val_data = lgb.Dataset(val_x, label= val_y)

# #Parameters of the model
# params = {
#     'learning_rate': 0.15,
#     #'is_unbalance': 'true', # replaced with scale_pos_weight argument
#     'num_leaves': 7,  # 2^max_depth - 1
#     'max_depth': 3,  # -1 means no limit
#     'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
#     'max_bin': 100,  # Number of bucketed bin for feature values
#     'subsample': 0.7,  # Subsample ratio of the training instance.
#     'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
#     'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
#     'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
#     'scale_pos_weight':99 # because training data is extremely unbalanced 
# }

# # Training the model
# model = lgb.train(params,
#                        lgb_train_data,
#                        valid_sets=lgb_val_data,
#                        num_boost_round=5000,
#                        early_stopping_rounds=100)
# Test code to view table
#train_x.head()
train_y.head()
#val_x.head()
#val_y.head()
#train_x.shape
#train_y.shape
#val_x.shape
val_y.shape