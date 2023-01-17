# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import multiprocessing

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline

import seaborn as sns

import scipy as sp

from scipy.stats import norm

from scipy import stats

import lightgbm as lgb

import gc

from time import time

import datetime

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, TimeSeriesSplit, StratifiedKFold, GroupKFold

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_squared_log_error, log_loss

import eli5

from eli5.sklearn import PermutationImportance

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
files = ['/kaggle/input/exam-for-students20200527/sample_submission.csv',

        '/kaggle/input/exam-for-students20200527/station_info.csv',

        '/kaggle/input/exam-for-students20200527/city_info.csv',

        '/kaggle/input/exam-for-students20200527/data_dictionary.csv',

        '/kaggle/input/exam-for-students20200527/test.csv',

        '/kaggle/input/exam-for-students20200527/train.csv']



def load_data(file):

    return pd.read_csv(file)



with multiprocessing.Pool() as pool:

    sub, stationData, cityData, dicData, testData, trainData = pool.map(load_data, files)
#データ確認

print(trainData.columns)

print(trainData.info())

trainData.describe()
trainData
plt.figure(figsize=(15, 5))

sns.distplot(trainData['TradePrice'], bins = 50)
#欠損確認

total = trainData.isnull().sum().sort_values(ascending=False)

percent = (trainData.isnull().sum()/trainData.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
trainData['Remarks'].value_counts()
trainData['Prefecture'].value_counts()
plt.figure(figsize=(15, 5))

sns.distplot(trainData['MinTimeToNearestStation'], bins = 50)

sns.distplot(testData['MinTimeToNearestStation'], bins = 50)
plt.figure(figsize=(20, 5))

sns.distplot(trainData['Area'], bins = 100)

sns.distplot(testData['Area'], bins = 100)
#欠損個数

# trainData['missingCount'] = trainData.isnull().sum(axis=1)

# testData['missingCount'] = testData.isnull().sum(axis=1)
#Null Flg

#Fillna

trainData['Year_Fill_NA'] = trainData['Year']

trainData['Year_Fill_NA'].fillna(trainData['Year'].mode()[0])



testData['Year_Fill_NA'] = testData['Year']

testData['Year_Fill_NA'].fillna(trainData['Year'].mode()[0])
#築年数

trainData['BuildingOld'] = trainData['Year_Fill_NA'] - trainData['BuildingYear']

testData['BuildingOld'] = testData['Year_Fill_NA'] - testData['BuildingYear']
#都心等からの距離?
#交互作用
#最寄駅周辺の物件数

concatStationData_20 = pd.concat([trainData['NearestStation'][trainData['MinTimeToNearestStation'] <= 20], testData['NearestStation'][testData['MinTimeToNearestStation'] <= 20]])

countEstatesAroundStation_20 = concatStationData_20.value_counts()

trainData['countEstatesAroundStation_20min'] = trainData['NearestStation'].map(countEstatesAroundStation_20)

testData['countEstatesAroundStation_20min'] = testData['NearestStation'].map(countEstatesAroundStation_20)



concatStationData_10 = pd.concat([trainData['NearestStation'][trainData['MinTimeToNearestStation'] <= 10], testData['NearestStation'][testData['MinTimeToNearestStation'] <= 10]])

countEstatesAroundStation_10 = concatStationData_10.value_counts()

trainData['countEstatesAroundStation_10min'] = trainData['NearestStation'].map(countEstatesAroundStation_10)

testData['countEstatesAroundStation_10min'] = testData['NearestStation'].map(countEstatesAroundStation_10)



concatStationData_5 = pd.concat([trainData['NearestStation'][trainData['MinTimeToNearestStation'] <= 5], testData['NearestStation'][testData['MinTimeToNearestStation'] <= 5]])

countEstatesAroundStation_5 = concatStationData_5.value_counts()

trainData['countEstatesAroundStation_5min'] = trainData['NearestStation'].map(countEstatesAroundStation_5)

testData['countEstatesAroundStation_5min'] = testData['NearestStation'].map(countEstatesAroundStation_5)
#学習データ作成

useless_col = ['id', 'Municipality', 'DistrictName', 'NearestStation', 'TimeToNearestStation','Prefecture']

train = trainData.drop(useless_col, axis=1)

test = testData.drop(useless_col, axis=1)



#del trainData

#del testData

#gc.collect()



#train = train.sample(n=1000, random_state=0)
train.columns
#CountEncoding

for col in tqdm_notebook(train.columns):

    if train[col].dtype == 'object':

        concatData = pd.concat([train[col],test[col]])

        countValue = concatData.value_counts()

        train[col] = train[col].map(countValue)

        test[col] = test[col].map(countValue)
#LabelEncoder

# for col in tqdm_notebook(train.columns):

#     if train[col].dtype == 'object':

#         le = LabelEncoder()

#         le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))

#         train[col] = le.transform(list(train[col].astype(str).values))

#         test[col] = le.transform(list(test[col].astype(str).values))
train
test
X = train.drop('TradePrice', axis=1)

y = np.log1p(train['TradePrice'])



#del train

#gc.collect()
X
print(y.describe())

plt.figure(figsize=(15, 5))

sns.distplot(y, bins = 50)
#CV etc. 県でGroupする？

num_split = 5

#folds = KFold(n_splits=num_split, random_state=1, shuffle=True)

#folds = TimeSeriesSplit(n_splits=num_split)

#folds = StratifiedKFold(n_splits=num_split, random_state=1, shuffle=True)

folds_group = GroupKFold(n_splits=num_split)

group = trainData.Prefecture.values



scores = list()

list_num_best_iteration = []

feature_importances = pd.DataFrame()

feature_importances['feature'] = X.columns

feature_importances_gain = feature_importances



y_pred_cva = np.zeros(len(test)) #cvaデータ収納用
group
params = {'num_leaves': 30,

          #'min_child_weight': 0.03454472573214212,

          #'feature_fraction': 0.3797454081646243,

          #'bagging_fraction': 0.4181193142567742,

          #'min_data_in_leaf': 106,

          'objective': 'regression', #'regression' / 'binary' / 'multiclass'

          "metric": 'rmse', # 'mae' or 'mse' or 'rmse' / 'binary_logloss' or 'auc'/ 'multi_logloss'

          'max_depth': 8, #-1

          'learning_rate': 0.05, #0.006883242363721497,

          "boosting_type": "gbdt",

          "bagging_seed": 11,

          "verbosity": -1, #途中経過の表示

          #'reg_alpha': 0.3899927210061127,

          #'reg_lambda': 0.6485237330340494,

          'random_state': 0

         }
#Training

training_start_time = time()



for fold, (trn_idx, test_idx) in enumerate(folds_group.split(X,y,group)):

#for fold, (trn_idx, test_idx) in enumerate(folds.split(X,y)):

    start_time=time()

    print('Training on fold {}'.format(fold + 1))

    

    trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])

    val_data = lgb.Dataset(X.iloc[test_idx], label=y.iloc[test_idx])

    model = lgb.train(params, trn_data, num_boost_round=100000, valid_sets = [trn_data, val_data], verbose_eval = 100, early_stopping_rounds=200)

    

    feature_importances['fold_{}'.format(fold + 1)] = model.feature_importance()

    feature_importances_gain['fold_{}'.format(fold + 1)] = model.feature_importance(importance_type='gain')

    scores.append(model.best_score['valid_1'])

    

    list_num_best_iteration.append(model.best_iteration)

    

    y_pred_cva += model.predict(test)

    print(model.predict(test))

    

    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() -start_time))))

print('-' * 30)

print('Training has finished.')

print('Total training time is {}'.format(str(datetime.timedelta(seconds = time() - training_start_time))))

print(scores)

print('-'*30)



y_pred_cva /= (num_split) #(num_split * num_iter)
#FeatureImportancesGain

feature_importances_gain['average'] = feature_importances_gain[['fold_{}'.format(fold + 1) for fold in range(folds.n_splits)]].mean(axis=1)

#feature_importances.to_csv('feature_importances.csv')



plt.figure(figsize=(16,16))

sns.barplot(data=feature_importances_gain.sort_values(by='average', ascending=False).head(20), x = 'average', y='feature');

plt.title('20 Top feature importance(gain) over {} folds average'.format(folds.n_splits));
pred_test = y_pred_cva
#log

pred_test = np.expm1(pred_test) 



#業務知見補正

#pred_test = np.where(pred_test>0, pred_test, 0)
plt.figure(figsize=(15, 5))

sns.distplot(pred_test, bins = 50)
pred_test_pd = pd.DataFrame(pred_test)

pred_test_pd.describe()
sub['TradePrice'] = pred_test

sub.to_csv('sub.csv', index=False)

sub