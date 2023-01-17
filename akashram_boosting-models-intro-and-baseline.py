import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

import json

import math

import cv2

import PIL

from PIL import Image



import plotly.express as px



import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm

from sklearn.decomposition import PCA

import os

import imagesize

import pydicom



%matplotlib inline
#Loading Train and Test Data



train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")

test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")



print("{} images in train set.".format(train.shape[0]))



print("{} images in test set.".format(test.shape[0]))
train.head()
test.head()
np.mean(train.FVC)
plt.figure(figsize=(12, 5))

plt.hist(train['Percent'].values, bins=200)

plt.title('Histogram Percent counts in train')

plt.xlabel('Value')

plt.ylabel('Count')

plt.show()
plt.figure(figsize=(12, 5))

plt.hist(test['Percent'].values, bins=200)

plt.title('Histogram Percent counts in test')

plt.xlabel('Value')

plt.ylabel('Count')

plt.show()
tt = train['SmokingStatus'].value_counts().reset_index()

tt.columns = ['cat', 'status']
import plotly.graph_objects as go



labels = tt.cat.values

values = tt.status.values



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.show()
fvc = train.FVC

fig = px.histogram(fvc)

fig.show()
age = train.Age

fig = px.histogram(age)

fig.show()
tt = train['Sex'].value_counts().reset_index()

tt.columns = ['sex', 'count']



labels = tt['sex'].values

values = tt['count'].values

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.show()
train.head()
def plot_pixel_array(dataset, figsize=(10,10)):

    plt.figure(figsize=figsize)

    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)

    plt.show()
import glob

import pydicom



def show_info(dataset):

    path = '../input/osic-pulmonary-fibrosis-progression/test/ID00419637202311204720264/'

    #dataset = pydicom.dcmread(path+'/'+filename)

    print("Patient id..........:", dataset.PatientID)

    

    if 'PixelData' in dataset:

        rows = int(dataset.Rows)

        cols = int(dataset.Columns)

        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(

            rows=rows, cols=cols, size=len(dataset.PixelData)))

        if 'PixelSpacing' in dataset:

            print("Pixel spacing....:", dataset.PixelSpacing)

            

for file_path in glob.glob('../input/osic-pulmonary-fibrosis-progression/test/ID00419637202311204720264/*.dcm'):

    #print(file_path)

    filename = file_path.split('/')[-1]

    dataset = pydicom.dcmread(file_path)

    show_info(dataset)

    plot_pixel_array(dataset)

    break
def show_info(dataset):

    path = '../input/osic-pulmonary-fibrosis-progression/test/ID00421637202311550012437/'

    #dataset = pydicom.dcmread(path+'/'+filename)

    print("Patient id..........:", dataset.PatientID)

    

    if 'PixelData' in dataset:

        rows = int(dataset.Rows)

        cols = int(dataset.Columns)

        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(

            rows=rows, cols=cols, size=len(dataset.PixelData)))

        if 'PixelSpacing' in dataset:

            print("Pixel spacing....:", dataset.PixelSpacing)



for file_path in glob.glob('../input/osic-pulmonary-fibrosis-progression/test/ID00421637202311550012437/*.dcm'):

    dataset = pydicom.dcmread(file_path)

    show_info(dataset)

    plot_pixel_array(dataset)

    break
def show_info(dataset):

    path = '../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/'

    #dataset = pydicom.dcmread(path+'/'+filename)

    print("Patient id..........:", dataset.PatientID)

    

    if 'PixelData' in dataset:

        rows = int(dataset.Rows)

        cols = int(dataset.Columns)

        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(

            rows=rows, cols=cols, size=len(dataset.PixelData)))

        if 'PixelSpacing' in dataset:

            print("Pixel spacing....:", dataset.PixelSpacing)



for file_path in glob.glob('../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/*.dcm'):

    dataset = pydicom.dcmread(file_path)

    show_info(dataset)

    plot_pixel_array(dataset)

    break
def show_info(dataset):

    path = '../input/osic-pulmonary-fibrosis-progression/train/ID00012637202177665765362/'

    #dataset = pydicom.dcmread(path+'/'+filename)

    print("Patient id..........:", dataset.PatientID)

    

    if 'PixelData' in dataset:

        rows = int(dataset.Rows)

        cols = int(dataset.Columns)

        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(

            rows=rows, cols=cols, size=len(dataset.PixelData)))

        if 'PixelSpacing' in dataset:

            print("Pixel spacing....:", dataset.PixelSpacing)





for file_path in glob.glob('../input/osic-pulmonary-fibrosis-progression/train/ID00012637202177665765362/*.dcm'):

    dataset = pydicom.dcmread(file_path)

    show_info(dataset)

    plot_pixel_array(dataset)

    break
import xgboost

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier 



import lightgbm as lgb

from numba import jit 
train = pd.read_csv( '../input/osic-pulmonary-fibrosis-progression/train.csv' )

test  = pd.read_csv( '../input/osic-pulmonary-fibrosis-progression/test.csv' )



train['traintest'] = 0

test ['traintest'] = 1

sub   = pd.read_csv( '../input/osic-pulmonary-fibrosis-progression/sample_submission.csv' )

sub['Weeks']   = sub['Patient_Week'].apply( lambda x: int(x.split('_')[-1]) )

sub['Patient'] = sub['Patient_Week'].apply( lambda x: x.split('_')[0] ) 



train['Sex']           = pd.factorize( train['Sex'] )[0]

train['SmokingStatus'] = pd.factorize( train['SmokingStatus'] )[0]
train['Percent']       = (train['Percent'] - train['Percent'].mean()) / train['Percent'].std()

train['Age']           = (train['Age'] - train['Age'].mean()) / train['Age'].std()

train['Sex']           = (train['Sex'] - train['Sex'].mean()) / train['Sex'].std()

train['SmokingStatus'] = (train['SmokingStatus'] - train['SmokingStatus'].mean()) / train['SmokingStatus'].std()
OUTPUT_DICT = './'



ID = 'Patient_Week'

TARGET = 'FVC'

SEED = 42



N_FOLD = 4



train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

train[ID] = train['Patient'].astype(str) + '_' + train['Weeks'].astype(str)

print(train.shape)

train.head()
output = pd.DataFrame()



gb = train.groupby('Patient')

tk0 = tqdm(gb, total=len(gb))



for _, usr_df in tk0:

    usr_output = pd.DataFrame()

    for week, tmp in usr_df.groupby('Weeks'):

        rename_cols = {'Weeks': 'base_Week', 'FVC': 'base_FVC', 'Percent': 'base_Percent', 'Age': 'base_Age'}

        tmp = tmp.drop(columns='Patient_Week').rename(columns=rename_cols)

        drop_cols = ['Age', 'Sex', 'SmokingStatus', 'Percent']

        _usr_output = usr_df.drop(columns=drop_cols).rename(columns={'Weeks': 'predict_Week'}).merge(tmp, on='Patient')

        _usr_output['Week_passed'] = _usr_output['predict_Week'] - _usr_output['base_Week']

        usr_output = pd.concat([usr_output, _usr_output])

    output = pd.concat([output, usr_output])

    

train = output[output['Week_passed']!=0].reset_index(drop=True)
test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv').rename(columns={'Weeks': 'base_Week', 'FVC': 'base_FVC', 'Percent': 'base_Percent', 'Age': 'base_Age'})



submission = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')



submission['Patient'] = submission['Patient_Week'].apply(lambda x: x.split('_')[0])



submission['predict_Week'] = submission['Patient_Week'].apply(lambda x: x.split('_')[1]).astype(int)



test = submission.drop(columns=['FVC', 'Confidence']).merge(test, on='Patient')



test['Week_passed'] = test['predict_Week'] - test['base_Week']



print(test.shape)



test.head()
submission = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from sklearn.metrics import mean_squared_error

import category_encoders as ce



folds = train[[ID, 'Patient', TARGET]].copy()



Fold = GroupKFold(n_splits=N_FOLD)



groups = folds['Patient'].values



for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[TARGET], groups)):

    folds.loc[val_index, 'fold'] = int(n)

    

folds['fold'] = folds['fold'].astype(int)
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor





def adaboost_it(ada_train_df, ada_test_df):

    print("Ada-Boosting...")

    t_splits = 5

    k_scores = []

    

    kf = KFold(n_splits = t_splits)

    features = [i for i in ada_train_df.columns if i not in ['Patient', 'predict_Week', 'FVC']]

    target = 'FVC'

    

    oof_pred = np.zeros((len(ada_train_df), 4))

    y_pred = np.zeros((len(ada_test_df), 4))

    

    for fold, (tr_ind, val_ind) in enumerate(kf.split(ada_train_df)):

        print(f'Fold: {fold+1}')

        x_train, x_val = ada_train_df[features].iloc[tr_ind], ada_train_df[features].iloc[val_ind]

        y_train, y_val = ada_train_df[target][tr_ind], ada_train_df[target][val_ind]

        ada_clf = AdaBoostRegressor(DecisionTreeRegressor(random_state=0 ) ) #max_depth=1), n_estimators=200, algorithm="SAMME.R", learning_rate=0.5)

        ada_clf.fit(x_train, y_train)

        oof_pred[val_ind] = ada_clf.predict(x_val).reshape(-1, 1)

        y_pred += ada_clf.predict(ada_test_df[features]).reshape(-1,1) / t_splits

        

    y_pred = y_pred.mean(axis=1)

    return y_pred
def metric( trueFVC, predFVC):

    deltaFVC = np.clip( np.abs(trueFVC-predFVC), 0 , 1000 )  

    return np.mean( -1*(np.sqrt(2)*deltaFVC/clipSTD) - np.log( np.sqrt(2)*clipSTD ) )
train.head(3)
train.drop(['Sex', 'SmokingStatus', 'Patient', 'Patient_Week'], axis=1, inplace=True)

test.drop(['Sex', 'SmokingStatus', 'Patient_Week', 'Patient'], axis=1, inplace=True)
adab_pred = adaboost_it(train, test)



test['FVC_pred'] = adab_pred
import xgboost



def xgb(xgb_train_df, xgb_test_df):

    

    print("XG-Boosting...")

    t_splits = 5

    k_scores = []

    kf = KFold(n_splits = t_splits)

    

    features = [i for i in xgb_train_df.columns if i not in ['Patient', 'predict_Week', 'FVC']]

    target = 'FVC'

    oof_pred = np.zeros((len(xgb_train_df), 4))

    y_pred = np.zeros((len(xgb_test_df), 4))

    for fold, (tr_ind, val_ind) in enumerate(kf.split(xgb_train_df)):

        

        print(f'Fold: {fold+1}')

        

        x_train, x_val = xgb_train_df[features].iloc[tr_ind], xgb_train_df[features].iloc[val_ind]

        y_train, y_val = xgb_train_df[target][tr_ind], xgb_train_df[target][val_ind]

        

        xgb_clf = xgboost.XGBRegressor()

        

        xgb_clf.fit(x_train, y_train)

        

        oof_pred[val_ind] = xgb_clf.predict(x_val).reshape(-1,1)

      

        y_pred += xgb_clf.predict(xgb_test_df[features]).reshape(-1,1) / t_splits

        

    y_pred = y_pred.mean(axis=1)        

    return y_pred
xgb_pred = xgb(train, test)
import catboost as cb



def cat(cat_train_df, cat_test_df):

    

    print("Meeowwww...")

    t_splits = 5

    k_scores = []

    kf = KFold(n_splits = t_splits)

    features = [i for i in cat_train_df.columns if i not in ['Patient', 'predict_Week', 'FVC']]

    target = 'FVC'

    oof_pred = np.zeros((len(cat_train_df), 4))

    y_pred = np.zeros((len(cat_test_df), 4))

    

    for fold, (tr_ind, val_ind) in enumerate(kf.split(cat_train_df)):

        

        print(f'Fold: {fold+1}')

        x_train, x_val = cat_train_df[features].iloc[tr_ind], cat_train_df[features].iloc[val_ind]

        y_train, y_val = cat_train_df[target][tr_ind], cat_train_df[target][val_ind]

        

        cat_clf = cb.CatBoostRegressor(logging_level='Silent')

        

        cat_clf.fit(x_train, y_train)

        oof_pred[val_ind] = cat_clf.predict(x_val).reshape(-1,1)

      

        y_pred += cat_clf.predict(cat_test_df[features]).reshape(-1,1) / t_splits

        

           

    y_pred = y_pred.mean(axis=1)               

    return y_pred
cat_pred = cat(train, test)
import lightgbm as lgb



def lgbc(lgb_train_df, lgb_test_df):

    

    t_splits = 5

    k_scores = []

    kf = KFold(n_splits = t_splits)

    

    features = [i for i in lgb_train_df.columns if i not in ['Patient', 'predict_Week', 'FVC']]

    target = 'FVC'

    

    oof_pred = np.zeros((len(lgb_train_df), 4))

    y_pred = np.zeros((len(lgb_test_df), 4))

    for fold, (tr_ind, val_ind) in enumerate(kf.split(lgb_train_df)):

        print(f'Fold: {fold+1}')

        x_train, x_val = lgb_train_df[features].iloc[tr_ind], lgb_train_df[features].iloc[val_ind]

        y_train, y_val = lgb_train_df[target][tr_ind], lgb_train_df[target][val_ind]

        

        lg = lgb.LGBMRegressor(silent=False)

        lg.fit(x_train, y_train)

        oof_pred[val_ind] = lg.predict(x_val).reshape(-1,1)

      

        y_pred += lg.predict(lgb_test_df[features]).reshape(-1,1) / t_splits

       

    y_pred = y_pred.mean(axis=1)        

    return y_pred
lg_pred = lgbc(train, test)
sub = lg_pred * 0.25 + xgb_pred * 0.25 + cat_pred * 0.50

submission['FVC'] = sub
train['FVC_pred'] = train.FVC

test['FVC_pred'] = sub



train['Confidence'] = 100

train['sigma_clipped'] = train['Confidence'].apply(lambda x: max(x, 70))

train['diff'] = abs(train['FVC'] - train['FVC_pred'])

train['delta'] = train['diff'].apply(lambda x: min(x, 1000))

train['score'] = -math.sqrt(2)*train['delta']/train['sigma_clipped'] - np.log(math.sqrt(2)*train['sigma_clipped'])

score = train['score'].mean()

print(score)
import scipy as sp

from functools import partial



def loss_func(weight, row):

    confidence = weight

    sigma_clipped = max(confidence, 70)

    diff = abs(row['FVC'] - row['FVC_pred'])

    delta = min(diff, 1000)

    score = -math.sqrt(2)*delta/sigma_clipped - np.log(math.sqrt(2)*sigma_clipped)

    return -score



results = []

tk0 = tqdm(train.iterrows(), total=len(train))

for _, row in tk0:

    loss_partial = partial(loss_func, row=row)

    weight = [100]

    #bounds = [(70, 100)]

    #result = sp.optimize.minimize(loss_partial, weight, method='SLSQP', bounds=bounds)

    result = sp.optimize.minimize(loss_partial, weight, method='SLSQP')

    x = result['x']

    results.append(x[0])
import scipy as sp

from functools import partial



def loss_func(weight, row):

    confidence = weight

    sigma_clipped = max(confidence, 70)

    diff = abs(row['base_FVC'] - row['FVC_pred'])

    delta = min(diff, 1000)

    score = -math.sqrt(2)*delta/sigma_clipped - np.log(math.sqrt(2)*sigma_clipped)

    return -score



tresults = []



tk0 = tqdm(test.iterrows(), total=len(test))

for _, row in tk0:

    loss_partial = partial(loss_func, row=row)

    weight = [100]

    #bounds = [(70, 100)]

    #result = sp.optimize.minimize(loss_partial, weight, method='SLSQP', bounds=bounds)

    result = sp.optimize.minimize(loss_partial, weight, method='SLSQP')

    x = result['x']

    tresults.append(x[0])
# optimized score



train['Confidence'] = results

train['sigma_clipped'] = train['Confidence'].apply(lambda x: max(x, 70))



train['diff'] = abs(train['FVC'] - train['FVC_pred'])

train['delta'] = train['diff'].apply(lambda x: min(x, 1000))

train['score'] = -math.sqrt(2)*train['delta']/train['sigma_clipped'] - np.log(math.sqrt(2)*train['sigma_clipped'])



score = train['score'].mean()



print(score)
# optimized score



test['Confidence'] = tresults



test['sigma_clipped'] = test['Confidence'].apply(lambda x: max(x, 70))



test['diff'] = abs(test['base_FVC'] - test['FVC_pred'])

test['delta'] = test['diff'].apply(lambda x: min(x, 1000))

test['score'] = -math.sqrt(2)*test['delta']/test['sigma_clipped'] - np.log(math.sqrt(2)*test['sigma_clipped'])



score = test['score'].mean()
submission['Confidence'] = tresults



submission.to_csv('submission.csv', index=False)



submission.head()