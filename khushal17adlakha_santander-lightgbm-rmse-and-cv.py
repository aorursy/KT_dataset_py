import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.plotly as py
%matplotlib inline
import plotly
import sklearn
import pymongo
import json
sklearn.__version__
import itertools
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
%matplotlib inline
sample = pd.read_csv('../input/sample_submission.csv')
test_df = pd.read_csv('../input/test.csv')
train_df = pd.read_csv('../input/train.csv')
print (train_df.shape)
print (test_df.shape)
test_df.head()
train_df.head()
cols_and_zeroes = []
for col in train_df.columns:
    no_of_zeroes = []
    cols_dict = {}
    aa = (train_df[col].value_counts())
    for key, value in aa.iteritems():
        if key==0:
            per = int(value * 100 / 4459)
            cols_dict[col] = int(per)
            cols_and_zeroes.append(cols_dict)
l= []
for i in cols_and_zeroes:
    for k,v in i.items():
        if v>=98:
            l.append(k)
print (len(l))
train_df.drop(l, axis=1, inplace=True)
test_df.drop(l, axis=1, inplace=True)

train_df.shape
test_df.shape
cols_to_remove = []
for col in train_df.columns:
    if col != 'ID':
        if train_df[col].std() == 0:
            cols_to_remove.append(col)
train_df.drop(cols_to_remove, axis=1, inplace=True)
test_df.drop(cols_to_remove, axis=1, inplace=True)
train_df.shape
test_df.shape
train_df.head()
corrcoefficient = []
for i in range (2,2123):
    cor = train_df.iloc[:,1].corr(train_df.iloc[:,i])
    corrcoefficient.append(cor)
    
plt.hist(corrcoefficient, normed=True, bins=10)
plt.xlabel('Correlation Coefficeint with target variable');
plt.ylabel('Frequency');
plt.title('Histogram of correlation coefficient of features with target variable');
target = train_df.iloc[:,1]
his = plt.hist(target, normed=True, bins=20)
log_target = np.log(target)
his_log = plt.hist(log_target, normed=True, bins=20)
train=train_df.iloc[:,2:].values
test=test_df.iloc[:,1:].values
print('Shape of train: ',train.shape)
print('Shape of target: ',log_target.shape)
print('Shape of test: ',test.shape)
def transform (dataframe):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataframe)
    return pd.DataFrame(scaled_data)

train = transform(train)
test = transform(test)
def get_PCA(DATAFRAME,NUMBER_OF_COMPONENTS):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=NUMBER_OF_COMPONENTS)
    pca.fit(DATAFRAME)
    DF_RETURN = pca.transform(DATAFRAME)
    print ("data frame shape: %f %f" %DF_RETURN.shape)
    #print pca.explained_variance_ratio_
    zzz = pca.explained_variance_ratio_
    plt.plot(zzz.cumsum())
    plt.xlabel('Number of Components');
    plt.ylabel('Cumulative Variance Ratio');
    plt.title('PCA Variance ratio of first 1500 components');
    return pd.DataFrame(DF_RETURN)
a = get_PCA(train,1500)
b = get_PCA(test,1500)
print (log_target.shape)
from sklearn.model_selection import train_test_split
#X_train, X_val, y_train, y_val = train_test_split(train, test, test_size=0.2, random_state=0)
X_train, X_target, Y_train, Y_target = train_test_split(a, log_target, test_size=0.30, random_state=101)
#Y_train, Y_test = train_test_split(test, test_size=0.20, random_state=101)

print (X_train.shape,Y_train.shape)
print (X_target.shape, Y_target.shape)
'''# coding: utf-8
# pylint: disable = invalid-name, C0111
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
print('Start training...')
import timeit
start = timeit.default_timer()


# other scikit-learn modules
estimator = lgb.LGBMRegressor()

param_grid = {
    'max_depth': [8,10,12,14],
    'learning_rate': [0.001, 0.01, 0.1, 1],
    'n_estimators': [20, 40, 60, 80],
    'bagging_fraction' : [0.00001, 0.0001,0.001, 0.01, 0.1],
    'num_leaves': [60, 90, 120, 150]
}

gbm = GridSearchCV(estimator, param_grid)

gbm.fit(a, log_target)

print('Best parameters found by grid search are:', gbm.best_params_)

stop = timeit.default_timer()
print stop - start'''

from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error,mean_squared_error
import lightgbm
#lgbm = LGBMRegressor()
train_data=lightgbm.Dataset(X_train,Y_train)
valid_data = lightgbm.Dataset(X_target,Y_target)
params={'learning_rate':0.1,
        'boosting_type':'gbdt',
        'objective':'regression',
        'metric':'rmse',
        'sub_feature':0.5,
        'num_leaves':90,
        'feature_fraction': 0.5,
        'bagging_fraction': 1e-05,
        'min_data':50,
        'max_depth':12,
        'reg_alpha': 0.3, 
        'reg_lambda': 0.1, 
        'min_child_weight': 10, 
        'verbose': 1,
        'nthread':5,
        'max_bin':512,
        'subsample_for_bin':200,
        'min_split_gain':0.0001,
        'min_child_samples':5
       }
lgbm = lightgbm.train(params,
                 train_data,
                 25000,
                 valid_sets=valid_data,
                 early_stopping_rounds= 80,
                 verbose_eval= 10
                 )

print( " Best iteration = ", lgbm.best_iteration )
Model_Summary = pd.DataFrame()
model_name='lightgbm_rmse'
RMSLE=np.sqrt(mean_squared_error(Y_target,lgbm.predict(X_target)))
RMSLE
cv_results = lightgbm.cv(params, train_data, num_boost_round=20, nfold=4, 
                    verbose_eval=10, early_stopping_rounds=80, stratified=False)
print (cv_results)
print('Current parameters:\n', params)
print('\nBest num_boost_round:', len(cv_results['rmse-mean']))
print('Best CV score:', cv_results['rmse-mean'][-1])
pred_lgbm=np.expm1(lgbm.predict(test))
histogram = plt.hist(pred_lgbm, normed=True, range = [1000000,3000000])
pred_lgbm.shape 
sub_1 = pd.DataFrame()
sub_1['ID'] = test_df.iloc[:,0]
sub_1['target'] = pred_lgbm
sub_1.to_csv('sub_1.csv',index=False)