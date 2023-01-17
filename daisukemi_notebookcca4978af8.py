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
import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.metrics import roc_auc_score



from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm



from sklearn.ensemble import GradientBoostingRegressor

import shap





from datetime import datetime as dt



from tensorflow.keras.layers import Dense ,Dropout, BatchNormalization, Input, Embedding, SpatialDropout1D, Reshape, Concatenate

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.metrics import AUC







from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, train_test_split,GridSearchCV







import seaborn as sns



from sklearn.preprocessing import StandardScaler,MinMaxScaler



import random

from collections import Counter, defaultdict





import xgboost as xgb

import time



from sklearn.preprocessing import quantile_transform



from catboost import CatBoostClassifier, FeaturesData, Pool,CatBoostRegressor



from lightgbm import LGBMRegressor



import lightgbm

import datetime

import random

import glob

import cv2

import os

import gc

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold,RepeatedKFold

import tensorflow as tf

from tensorflow.keras.models import Sequential,Model

from tensorflow.keras.layers import BatchNormalization,Activation,Dropout,Dense, Input,concatenate

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau

from tensorflow.keras.applications import VGG16,VGG19

from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

%matplotlib inline



def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)



# 乱数シード固定

seed = 228

seed_everything(seed)





from sklearn.metrics import mean_squared_error,mean_squared_log_error
# city= pd.read_csv(r"")

# sample_sub=pd.read_csv(r"")

# station_info=pd.read_csv(r"")

# train=pd.read_csv(r"../input/predata/train_income.csv")

# test=pd.read_csv(r"../input/predata/test_income.csv")

train = pd.read_csv('../input/exam-for-students20200923/train.csv')

test  = pd.read_csv('../input/exam-for-students20200923/test.csv')

country =pd.read_csv('../input/exam-for-students20200923/country_info.csv')

# survey =pd.read_csv('../input/exam-for-students20200923/survey_dictionary.csv')



# train_nn = pd.read_csv('')

# test_nn  = pd.read_csv('')



submission = pd.read_csv('../input/exam-for-students20200923/sample_submission.csv', index_col=0)
train_cun=pd.merge(train,country,how = 'left',on="Country")

test_cun=pd.merge(test,country,how = 'left',on="Country")
X_train= train_cun.drop(["ConvertedSalary",'Country'],axis=1)

X_test = test_cun.drop('Country',axis=1)
#ターゲットを指定



target  = "ConvertedSalary"
y_train=train[target]

# y_train=np.log1p(y_train)

train = train.drop(target,axis=1)
# y_target[y_target==0].count()
# #testデータとtrainデータの結合

# def ketsugou(train,test):

#     all_data = pd.concat([train,test]).reset_index(drop = True) 

#     return all_data
#前処理（欠損値を削除、埋め込み、オーディナルエンコーディング、カウントエンコーディング）

def prepare(all_data):

    #新規特徴量作成

    return all_data

# def not_object(X_test,X_train):

#     nums = []

#     for col in X_test.columns:

#         if X_test[col].dtype != 'object':

#             nums.append(col)

#             print(col,X_train[col].nunique())

#     X_train[nums] = X_train[nums].fillna(X_train[nums].median())   

#     X_train[nums] = quantile_transform(X_train[nums],n_quantiles=100,random_state=0,output_distribution='normal')

    

#     X_test[nums] = X_test[nums].fillna(X_test[nums].median())   

#     X_test[nums] = quantile_transform(X_test[nums],n_quantiles=100,random_state=0,output_distribution='normal')

#     return X_test,X_train



def not_object(X_test):

    nums = []

    for col in X_test.columns:

        if X_test[col].dtype != 'object':

            nums.append(col)

            print(col,X_train[col].nunique())



    return X_test
not_object(X_train)
# #カテゴリ変数をマッピングする

# grade_mapping = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':6}

# sub_grade_mapping = {'A1':1,'A2':2,'A3':3,'A4':4,'A5':5,

#                     'B1':6,'B2':7,'B3':8,'B4':9,'B5':10,

#                     'C1':11,'C2':12,'C3':13,'C4':14,'C5':15,

#                     'D1':16,'D2':17,'D3':18,'D4':19,'D5':20,

#                     'E1':21,'E2':22,'E3':23,'E4':24,'E5':25,

#                     'F1':26,'F2':27,'F3':28,'F4':29,'F5':30,

#                     'G1':31,'G2':32,'G3':33,'G4':34,'G5':35,}

# all_data['grade']=all_data['grade'].map(grade_mapping)

# all_data['sub_grade']=all_data['sub_grade'].map(sub_grade_mapping)



# all_data['zip_code']=all_data['zip_code'].str[:3]
###カウントエンコーディング

# all_data['grade_cun']=all_data['grade'].map(all_data['grade'].value_counts())
def object(X_train,X_test):

    cats = []

    for col in X_train.columns:

        if X_train[col].dtype == 'object':

            cats.append(col)

        

            print(col, X_train[col].nunique())



    #ordinal encoding

    encoder=OrdinalEncoder(cols=cats)

    X_train[cats]=encoder.fit_transform(X_train[cats])

    X_test[cats]=encoder.fit_transform(X_test[cats])

    return X_train,X_test,cats
country.head()
# not_object(all_data)

# all_data['missing_sum']=all_data.isnull().sum(axis=1)
cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

        print(col, X_train[col].nunique())
cats
X_train.info()
object(X_train,X_test)
X_train.columns.values
target_enc_col = ['Employment','LastNewJob','YearsCodingProf','Region','SalaryType','Currency','Age','Student','CompanySize',"UndergradMajor","Agriculture","DevType",'MilitaryUS','CareerSatisfaction','NumberMonitors','OperatingSystem','EducationParents']



for i ,enc_col in enumerate(target_enc_col):

    

    all_data = pd.concat([X_train,y_train],axis=1)

    

    sum = all_data.groupby([enc_col])[target].mean()

    test = X_test[enc_col].map(sum)

    

    skf = StratifiedKFold(n_splits=5,random_state=71,shuffle=True)

    

    train = Series(np.zeros(len(X_train)),index=X_train.index)

    

    for i ,(train_ix,val_ix) in enumerate((skf.split(X_train,y_train))):

        X_train_,_=all_data.iloc[train_ix],y_train.iloc[train_ix]

        X_val ,_ = all_data.iloc[val_ix],y_train.iloc[val_ix]

        

        sum = X_train_.groupby([enc_col])[target].mean()

        train.iloc[val_ix] = X_val[enc_col].map(sum)

    X_train[enc_col] = train

    X_test[enc_col] = test
# all_data.fillna(all_data.median(),inplace=True)
y_target = np.log1p(y_train)
drop=["Other (%)","Climate","Infant mortality (per 1000 births)","Service","Phones (per 1000)","HypotheticalTools2","Arable (%)","Birthrate",

      "OpenSource","Hobby","StackOverflowHasAccount","AdsAgreeDisagree2","AIDangerous","StackOverflowJobsRecommend","EthicalImplications",

      "AdBlockerDisable","JobSearchStatus","SurveyTooLong","Industry","StackOverflowJobs","AIInteresting","TimeAfterBootcamp"

     ]
X_train=X_train.drop(drop,axis=1)

X_test=X_test.drop(drop,axis=1)
# imp = [""]
# # all_data = prepare(pd.concat([train,test]).reset_index(drop = True))

# train = all_data.loc[:train.shape[0] -1,:]

# test = all_data.loc[train.shape[0]:,:]
X_train
X_train
X_train["new"]=X_train["LastNewJob"]*X_train["SalaryType"]

X_test["new"]=X_test["LastNewJob"]*X_test["SalaryType"]
#train test y_target



scores = []

# pred_train = np.zeros(X_train.shape[0])

pred_test_lgb = np.zeros(X_test.shape[0])



kf = KFold(n_splits=5, random_state=42, shuffle=True)

# skf=StratifiedKFold(n_splits=5,random_state=71,shuffle=True)



for i, (train_ix, test_ix) in tqdm(enumerate(kf.split(X_train, y_target))):

    X_train_, y_train_ = X_train.values[train_ix], y_target.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_target.values[test_ix]

        

    clf = LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,

                               importance_type='split', learning_rate=0.05, max_depth=-1,

                               min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                               n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

                               random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

                               subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

    

#     clf = LGBMRegressor(boosting_type='gbdt', num_leaves=15, max_depth=- 1, learning_rate=0.1,

#                         n_estimators=1000, subsample_for_bin=200000, objective=None, 

#                         class_weight=None, min_split_gain=0.0, min_child_weight=0.001,

#                         min_child_samples=30, subsample=0.8, subsample_freq=0, colsample_bytree=0.8,

#                         reg_alpha=1, reg_lambda=1, random_state=None, n_jobs=- 1,loss_function = 'RMSE',eval_metric = 'RMSE')

    

#     clf = LGBMRegressor(objective='regression', 

#                        num_leaves=6,

#                        learning_rate=0.01, 

#                        n_estimators=7000,

#                        max_bin=200, 

#                        bagging_fraction=0.8,

#                        bagging_freq=4, 

#                        bagging_seed=8,

#                        feature_fraction=0.2,

#                        feature_fraction_seed=8,

#                        min_sum_hessian_in_leaf = 11,

#                        verbose=-1,

#                        random_state=42)

    

    

    clf.fit(X_train_, y_train_,early_stopping_rounds =25,eval_metric = 'rmse',eval_set=[(X_val,y_val)])          

#     y_pred = np.expm1(clf.predict(X_val)) 

    

    y_pred = clf.predict(X_val)

    

#     lgb_y_pred_train[test_ix] = y_pred

    score = mean_squared_error(y_val, y_pred) 

#     rmse = np.sqrt(score)

    scores.append(score)

    pred_test_lgb += np.expm1(clf.predict(X_test))

    

    print('CV Score of Fold_%d is %f' % (i, score))





pred_test_lgb /= 5

importance = DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance']).sort_values(['importance'], ascending=False)

importance
# fig, ax = plt.subplots(figsize=(5, 8))

# lightgbm.plot_importance(clf, max_num_features=50, ax=ax, importance_type='gain')
# scores = []



# pred_test_xgb = np.zeros(test.shape[0])



# for i, (train_ix, test_ix) in tqdm(enumerate(kf.split(X_train, y_target))):

#     X_train_, y_train_ = X_train.values[train_ix], y_target.values[train_ix]

#     X_val, y_val = X_train.values[test_ix], y_target.values[test_ix]



#     clf = CatBoostRegressor(n_estimators = 500,loss_function = 'RMSE',eval_metric = 'RMSE')



    

#     clf.fit(X_train_, y_train_) 

#     y_pred = clf.predict(X_val) 

    



#     score = mean_squared_error(y_val, y_pred)

#     scores.append(score)

#     pred_test_cat += np.expm1(clf.predict(X_test))

    

    

#     print('CV Score of Fold_%d is %f' % (i, score))





# pred_test_cat /= 5
scores = []



pred_test_cat = np.zeros(test.shape[0])



for i, (train_ix, test_ix) in tqdm(enumerate(kf.split(X_train, y_target))):

    X_train_, y_train_ = X_train.values[train_ix], y_target.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_target.values[test_ix]



    clf = CatBoostRegressor(n_estimators = 500,loss_function = 'RMSE',eval_metric = 'RMSE')



    

    clf.fit(X_train_, y_train_) 

    y_pred = clf.predict(X_val) 

    



    score = mean_squared_error(y_val, y_pred)

    scores.append(score)

    pred_test_cat += np.expm1(clf.predict(X_test))

    

    

    print('CV Score of Fold_%d is %f' % (i, score))





pred_test_cat /= 5


submission.ConvertedSalary = (pred_test_lgb*0.8 + pred_test_cat*0.2)

# submission.ConvertedSalary = pred_test_lgb

submission.to_csv('submission_8.csv')