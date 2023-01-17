import os

from time import time



import matplotlib.pyplot as plt

from matplotlib import rcParams

%matplotlib inline

import seaborn as sns

from seaborn import countplot,lineplot, barplot



import numpy as np 

import pandas as pd 



from sklearn import preprocessing

le = preprocessing.LabelEncoder()

from sklearn.metrics import accuracy_score



from scipy import stats

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from numba import jit

import itertools



from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn.model_selection import StratifiedKFold



import warnings

warnings.filterwarnings('ignore')

import gc

gc.enable()



!ls ../input/
train_raw = pd.read_csv('../input/X_train.csv')

test_raw = pd.read_csv('../input/X_test.csv')

target_raw = pd.read_csv('../input/y_train.csv')
def quaternion_to_euler(x, y, z, w):

    import math

    t0 = +2.0 * (w * x + y * z)

    t1 = +1.0 - 2.0 * (x * x + y * y)

    X = math.atan2(t0, t1)



    t2 = +2.0 * (w * y - z * x)

    t2 = +1.0 if t2 > +1.0 else t2

    t2 = -1.0 if t2 < -1.0 else t2

    Y = math.asin(t2)



    t3 = +2.0 * (w * z + x * y)

    t4 = +1.0 - 2.0 * (y * y + z * z)

    Z = math.atan2(t3, t4)



    return X, Y, Z
def fe_step0 (actual):

    

    # https://www.mathworks.com/help/aeroblks/quaternionnorm.html

    # https://www.mathworks.com/help/aeroblks/quaternionmodulus.html

    # https://www.mathworks.com/help/aeroblks/quaternionnormalize.html

        

    actual['norm_quat'] = (actual['orientation_X']**2 + actual['orientation_Y']**2 + actual['orientation_Z']**2 + actual['orientation_W']**2)

    actual['mod_quat'] = (actual['norm_quat'])**0.5

    actual['norm_X'] = actual['orientation_X'] / actual['mod_quat']

    actual['norm_Y'] = actual['orientation_Y'] / actual['mod_quat']

    actual['norm_Z'] = actual['orientation_Z'] / actual['mod_quat']

    actual['norm_W'] = actual['orientation_W'] / actual['mod_quat']

    

    return actual
train_raw = fe_step0(train_raw)

test_raw = fe_step0(test_raw)
def fe_step1 (actual):

    """Quaternions to Euler Angles"""

    

    x, y, z, w = actual['norm_X'].tolist(), actual['norm_Y'].tolist(), actual['norm_Z'].tolist(), actual['norm_W'].tolist()

    nx, ny, nz = [], [], []

    for i in range(len(x)):

        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])

        nx.append(xx)

        ny.append(yy)

        nz.append(zz)

    

    actual['euler_x'] = nx

    actual['euler_y'] = ny

    actual['euler_z'] = nz

    return actual
train_raw = fe_step1(train_raw)

test_raw = fe_step1(test_raw)
def feat_eng(data):

    

    df = pd.DataFrame()

    data['totl_anglr_vel'] = (data['angular_velocity_X']**2 + data['angular_velocity_Y']**2 + data['angular_velocity_Z']**2)** 0.5

    data['totl_linr_acc'] = (data['linear_acceleration_X']**2 + data['linear_acceleration_Y']**2 + data['linear_acceleration_Z']**2)**0.5

    data['totl_xyz'] = (data['orientation_X']**2 + data['orientation_Y']**2 + data['orientation_Z']**2)**0.5

    data['acc_vs_vel'] = data['totl_linr_acc'] / data['totl_anglr_vel']

    

    def mean_change_of_abs_change(x):

        return np.mean(np.diff(np.abs(np.diff(x))))

    

    for col in data.columns:

        if col in ['row_id','series_id','measurement_number']:

            continue

        df[col + '_mean'] = data.groupby(['series_id'])[col].mean()

        df[col + '_median'] = data.groupby(['series_id'])[col].median()

        df[col + '_max'] = data.groupby(['series_id'])[col].max()

        df[col + '_min'] = data.groupby(['series_id'])[col].min()

        df[col + '_std'] = data.groupby(['series_id'])[col].std()

        df[col + '_range'] = df[col + '_max'] - df[col + '_min']

        df[col + '_maxtoMin'] = df[col + '_max'] / df[col + '_min']

        df[col + '_mean_abs_chg'] = data.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))

        df[col + '_mean_change_of_abs_change'] = data.groupby('series_id')[col].apply(mean_change_of_abs_change)

        df[col + '_abs_max'] = data.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))

        df[col + '_abs_min'] = data.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))

        df[col + '_abs_avg'] = (df[col + '_abs_min'] + df[col + '_abs_max'])/2

    return df
train_raw = feat_eng(train_raw)

test_raw = feat_eng(test_raw)
train_raw.fillna(0,inplace=True)

test_raw.fillna(0,inplace=True)

train_raw.replace(-np.inf,0,inplace=True)

train_raw.replace(np.inf,0,inplace=True)

test_raw.replace(-np.inf,0,inplace=True)

test_raw.replace(np.inf,0,inplace=True)
target_raw['surface'] = le.fit_transform(target_raw['surface'])
train_df = pd.merge(train_raw,target_raw,on='series_id')

train_df.drop('group_id', axis=1, inplace=True)

features = list(train_df.columns.values[1:])

features.remove('surface')



X = train_df[features].values

y = pd.DataFrame(train_df['surface']).values



test = test_raw[features].values



submission = pd.DataFrame()

submission['series_id'] = test_raw.index.values

submission['target'] = ""
def runXGB(train_X, train_y, validation_X, validation_y, test_X):

    param = {}

    param['num_class'] = 9

    param['objective'] = 'multi:softmax'

    param['eta'] = 0.1

    param['max_depth'] = 6

    param['silent'] = 1

    param['gamma'] = 0

    param['eval_metric'] = "merror"

    param['min_child_weight'] = 3

    param['max_delta_step'] = 1

    param['subsample'] = 0.9

    param['colsample_bytree'] = 0.4

    param['colsample_bylevel'] = 0.6

    param['colsample_bynode'] = 0.5

    param['lambda'] = 0

    param['alpha'] = 0

    param['seed'] = 0

    num_rounds = 500



    plst = list(param.items())



    xgtrain = xgb.DMatrix(train_X, label = train_y)

    xgcv = xgb.DMatrix(validation_X, label = validation_y)

    xgtest = xgb.DMatrix(test_X)



    evallist = [(xgcv,'eval')]

    model = xgb.train(plst, xgtrain, num_rounds, evallist, early_stopping_rounds = 100)



    pred_test_y = model.predict(xgtest)

    return pred_test_y, model
kfold = 7

skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)



for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    submission = pd.DataFrame()

    submission['series_id'] = test_raw.index.values

    submission['surface' + str(i+1)] = ""

    

    print('[Fold %d/%d]' % (i + 1, kfold))

    X_train, X_valid = X[train_index], X[test_index]

    y_train, y_valid = y[train_index], y[test_index]

    

    preds, model = runXGB(X_train, y_train, X_valid, y_valid, test)

    

    submission['surface' + str(i+1)] = preds

    

    submission.to_csv('submission_' + str(i+1) + '.csv', index=False)
submission1 = pd.read_csv('submission_1.csv')

submission2 = pd.read_csv('submission_2.csv')

submission3 = pd.read_csv('submission_3.csv')

submission4 = pd.read_csv('submission_4.csv')

submission5 = pd.read_csv('submission_5.csv')

submission6 = pd.read_csv('submission_6.csv')

submission7 = pd.read_csv('submission_7.csv')
from functools import reduce

submissions = [submission1,submission2,submission3,submission4,submission5,submission6,submission7]

submission_final = reduce(lambda left,right: pd.merge(left,right,on='series_id'), submissions)
submission_final = pd.DataFrame(submission_final.mode(axis='columns'))

submission_final = pd.DataFrame(submission_final[submission_final.columns[0]])

submission_final.columns = ['surface']

submission_final['surface'] = submission_final['surface'].astype('int')

submission_final['surface'] = le.inverse_transform(submission_final['surface'])

submission_final['series_id'] = test_raw.index.values

submission_final.to_csv('submission_final.csv', index=False)