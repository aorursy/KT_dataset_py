from sklearn.model_selection import train_test_split

from plotly.offline import init_notebook_mode, iplot

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from catboost import CatBoostClassifier,Pool

from sklearn.metrics import roc_auc_score

import matplotlib.gridspec as gridspec

from IPython.display import display

import matplotlib.patches as patch

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from scipy.stats import norm

import plotly.plotly as py

from sklearn import svm

import lightgbm as lgb

import catboost as cb

import xgboost as xgb

import seaborn as sns

import pandas as pd

import numpy as np

import warnings

import pickle

import time

import glob

import sys

import os

import gc

gc.enable()
#fold_n=5

#folds = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=10)

%matplotlib inline

%precision 4

warnings.filterwarnings('ignore')

plt.style.use('ggplot')

np.set_printoptions(suppress=True)

pd.set_option("display.precision", 15)
train= pd.read_csv("../input/train.csv")

test = pd.read_csv('../input/test.csv')
#Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65

def reduce_mem_usage(df):

    start_mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in df.columns:

        if df[col].dtype != object:  # Exclude strings

            

            # Print current column type

            print("******************************")

            print("Column: ",col)

            print("dtype before: ",df[col].dtype)

            

            # make variables for Int, max and min

            IsInt = False

            mx = df[col].max()

            mn = df[col].min()

            

            # Integer does not support NA, therefore, NA needs to be filled

            if not np.isfinite(df[col]).all(): 

                NAlist.append(col)

                df[col].fillna(mn-1,inplace=True)  

                   

            # test if column can be converted to an integer

            asint = df[col].fillna(0).astype(np.int64)

            result = (df[col] - asint)

            result = result.sum()

            if result > -0.01 and result < 0.01:

                IsInt = True



            

            # Make Integer/unsigned Integer datatypes

            if IsInt:

                if mn >= 0:

                    if mx < 255:

                        df[col] = df[col].astype(np.uint8)

                    elif mx < 65535:

                        df[col] = df[col].astype(np.uint16)

                    elif mx < 4294967295:

                        df[col] = df[col].astype(np.uint32)

                    else:

                        df[col] = df[col].astype(np.uint64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        df[col] = df[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        df[col] = df[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        df[col] = df[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        df[col] = df[col].astype(np.int64)    

            

            # Make float datatypes 32 bit

            else:

                df[col] = df[col].astype(np.float32)

            

            # Print new column type

            print("dtype after: ",df[col].dtype)

            print("******************************")

    

    # Print final result

    print("___MEMORY USAGE AFTER COMPLETION:___")

    mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage is: ",mem_usg," MB")

    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")

    return df, NAlist
train, NAlist = reduce_mem_usage(train)

print("_________________")

print("")

print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")

print("_________________")

print("")

print(NAlist)
test, NAlist = reduce_mem_usage(test)

print("_________________")

print("")

print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")

print("_________________")

print("")

print(NAlist)


init_notebook_mode(connected=True)

cnt_trgt = train.target.value_counts()



iplot([go.Bar(x=cnt_trgt.index, y=cnt_trgt.values)])
cols=["target","ID_code"]

X = train.drop(cols,axis=1)

y = train["target"]
#from sklearn.feature_selection import RFECV

#rfr=RandomForestClassifier(random_state=0)

#rfecv = RFECV(estimator=rfr, step=10, min_features_to_select=50, cv=StratifiedKFold(2),

#              scoring='accuracy', verbose =2)

#rfecv.fit(X, y)
#cols = rfecv.get_support(indices=True)

# X= X.iloc[:,cols]
features= ['var_0', 'var_1', 'var_2', 'var_5', 'var_6', 'var_9', 'var_12',

       'var_13', 'var_18', 'var_21', 'var_22', 'var_26', 'var_33', 'var_34',

       'var_40', 'var_44', 'var_53', 'var_56', 'var_67', 'var_75', 'var_76',

       'var_78', 'var_80', 'var_81', 'var_86', 'var_91', 'var_92', 'var_93',

       'var_94', 'var_95', 'var_99', 'var_108', 'var_109', 'var_110',

       'var_115', 'var_118', 'var_121', 'var_122', 'var_123', 'var_133',

       'var_139', 'var_146', 'var_147', 'var_148', 'var_154', 'var_155',

       'var_157', 'var_164', 'var_165', 'var_166', 'var_169', 'var_170',

       'var_174', 'var_177', 'var_179', 'var_184', 'var_188', 'var_190',

       'var_191', 'var_198']

train_df = X.loc[:,features]

test_df = test.loc[:,features]



columns = train_df.columns



pos = train.target == 1

neg = train.target == 0



grid = gridspec.GridSpec(100, 2)

plt.figure(figsize=(15,100*4))



for n, col in enumerate(train_df[columns]):

    ax = plt.subplot(grid[n])

    sns.distplot(train[col][pos], bins = 50, color='c') #Will receive the "semi-salmon" violin

    sns.distplot(train[col][neg], bins = 50, color='y') #Will receive the "ocean" color

    ax.set_ylabel('Density')

    ax.set_title(str(col))

    ax.set_xlabel('')

plt.show()
def fit_lgb(X_fit, y_fit, X_val, y_val, counter, lgb_path, name):

    

    model = lgb.LGBMClassifier(max_depth=-1,

                               n_estimators=999999,

                               learning_rate=0.02,

                               colsample_bytree=0.3,

                               num_leaves=2,

                               metric='auc',

                               objective='binary', 

                               n_jobs=-1)

     

    model.fit(X_fit, y_fit, 

              eval_set=[(X_val, y_val)],

              verbose=0, 

              early_stopping_rounds=1000)

                  

    cv_val = model.predict_proba(X_val)[:,1]

    

    #Save LightGBM Model

    save_to = '{}{}_fold{}.txt'.format(lgb_path, name, counter+1)

    model.booster_.save_model(save_to)

    

    return cv_val

def fit_xgb(X_fit, y_fit, X_val, y_val, counter, xgb_path, name):

    

    model = xgb.XGBClassifier(max_depth=2,

                              n_estimators=999999,

                              colsample_bytree=0.3,

                              learning_rate=0.02,

                              objective='binary:logistic', 

                              n_jobs=-1)

     

    model.fit(X_fit, y_fit, 

              eval_set=[(X_val, y_val)], 

              verbose=0, 

              early_stopping_rounds=1000)

              

    cv_val = model.predict_proba(X_val)[:,1]

    

    #Save XGBoost Model

    save_to = '{}{}_fold{}.dat'.format(xgb_path, name, counter+1)

    pickle.dump(model, open(save_to, "wb"))

    

    return cv_val
def fit_cb(X_fit, y_fit, X_val, y_val, counter, cb_path, name):

    

    model = cb.CatBoostClassifier(iterations=999999,

                                  max_depth=2,

                                  learning_rate=0.02,

                                  colsample_bylevel=0.03,

                                  objective="Logloss")

                                  

    model.fit(X_fit, y_fit, 

              eval_set=[(X_val, y_val)], 

              verbose=0, early_stopping_rounds=1000)

              

    cv_val = model.predict_proba(X_val)[:,1]

    

    #Save Catboost Model          

    save_to = "{}{}_fold{}.mlmodel".format(cb_path, name, counter+1)

    model.save_model(save_to, format="coreml", 

                     export_parameters={'prediction_type': 'probability'})

                     

    return cv_val
def train_stage(df, lgb_path, xgb_path, cb_path):

    

    print('Load Train Data.')

    df = train_df

    print('\nShape of Train Data: {}'.format(df.shape))

    

    y_df = y                        

    df_ids = np.array(df.index)                     

    #df.drop(['ID_code', 'target'], axis=1, inplace=True)

    

    lgb_cv_result = np.zeros(df.shape[0])

    xgb_cv_result = np.zeros(df.shape[0])

    cb_cv_result  = np.zeros(df.shape[0])

    

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    skf.get_n_splits(df_ids, y_df)

    

    print('\nModel Fitting...')

    for counter, ids in enumerate(skf.split(df_ids, y_df)):

        print('\nFold {}'.format(counter+1))

        X_fit, y_fit = df.values[ids[0]], y_df[ids[0]]

        X_val, y_val = df.values[ids[1]], y_df[ids[1]]

    

        print('LigthGBM')

        lgb_cv_result[ids[1]] += fit_lgb(X_fit, y_fit, X_val, y_val, counter, lgb_path, name='lgb')

        print('XGBoost')

        xgb_cv_result[ids[1]] += fit_xgb(X_fit, y_fit, X_val, y_val, counter, xgb_path, name='xgb')

        print('CatBoost')

        cb_cv_result[ids[1]]  += fit_cb(X_fit,  y_fit, X_val, y_val, counter, cb_path,  name='cb')

        

        del X_fit, X_val, y_fit, y_val

        gc.collect()

    

    auc_lgb  = round(roc_auc_score(y_df, lgb_cv_result),4)

    auc_xgb  = round(roc_auc_score(y_df, xgb_cv_result),4)

    auc_cb   = round(roc_auc_score(y_df, cb_cv_result), 4)

    auc_mean = round(roc_auc_score(y_df, (lgb_cv_result+xgb_cv_result+cb_cv_result)/3), 4)

    auc_mean_lgb_cb = round(roc_auc_score(y_df, (lgb_cv_result+cb_cv_result)/2), 4)

    print('\nLightGBM VAL AUC: {}'.format(auc_lgb))

    print('XGBoost  VAL AUC: {}'.format(auc_xgb))

    print('Catboost VAL AUC: {}'.format(auc_cb))

    print('Mean Catboost+LightGBM VAL AUC: {}'.format(auc_mean_lgb_cb))

    print('Mean XGBoost+Catboost+LightGBM, VAL AUC: {}\n'.format(auc_mean))
def prediction_stage(df_path, lgb_path, xgb_path, cb_path):

    

    print('Load Test Data.')

    df = df_path

    print('\nShape of Test Data: {}'.format(df.shape))

    

    #df.drop(['ID_code'], axis=1, inplace=True)

    

    lgb_models = sorted(os.listdir(lgb_path))

    xgb_models = sorted(os.listdir(xgb_path))

    cb_models  = sorted(os.listdir(cb_path))

    

    lgb_result = np.zeros(df.shape[0])

    xgb_result = np.zeros(df.shape[0])

    cb_result  = np.zeros(df.shape[0])

    

    print('\nMake predictions...\n')

    

    print('With LightGBM...')

    for m_name in lgb_models:

        #Load LightGBM Model

        model = lgb.Booster(model_file='{}{}'.format(lgb_path, m_name))

        lgb_result += model.predict(df.values)

     

    print('With XGBoost...')    

    for m_name in xgb_models:

        #Load Xgboost Model

        model = pickle.load(open('{}{}'.format(xgb_path, m_name), "rb"))

        xgb_result += model.predict(df.values)

    

    print('With CatBoost...')        

    for m_name in cb_models:

        #Load Catboost Model

        model = cb.CatBoostClassifier()

        model = model.load_model('{}{}'.format(cb_path, m_name), format = 'coreml')

        cb_result += model.predict(df.values, prediction_type='Probability')[:,1]

    

    lgb_result /= len(lgb_models)

    xgb_result /= len(xgb_models)

    cb_result  /= len(cb_models)

    

    submission = pd.read_csv('../input/sample_submission.csv')

    submission['target'] = (lgb_result+xgb_result+cb_result)/3

    submission.to_csv('xgb_lgb_cb_starter_submission.csv', index=False)

    submission['target'] = (lgb_result+cb_result)/2

    submission.to_csv('lgb_cb_starter_submission.csv', index=False)

    submission['target'] = xgb_result

    submission.to_csv('xgb_starter_submission.csv', index=False)

    submission['target'] = lgb_result

    submission.to_csv('lgb_starter_submission.csv', index=False)

    submission['target'] = cb_result

    submission.to_csv('cb_starter_submission.csv', index=False)
train_path = train_df

test_path  = test_df

    

lgb_path = './lgb_models_stack/'

xgb_path = './xgb_models_stack/'

cb_path  = './cb_models_stack/'



    #Create dir for models

os.mkdir(lgb_path)

os.mkdir(xgb_path)

os.mkdir(cb_path)



print('Train Stage.\n')

train_stage(train_path, lgb_path, xgb_path, cb_path)

    

print('Prediction Stage.\n')

prediction_stage(test_path, lgb_path, xgb_path, cb_path)