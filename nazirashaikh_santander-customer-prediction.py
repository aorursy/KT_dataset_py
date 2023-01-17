# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train=pd.read_csv('../input/train.csv')
df_train.shape
df_train.head()
df_train.info()
df_test=pd.read_csv('../input/test.csv')
df_test.shape
df_test.info()
df_test.head()
def memory_usage(df):

    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    for col in df.columns:

        if str(df[col].dtype) in numerics:

            if str(df[col].dtype)[:3] == 'int':

                if ((df[col].min() > np.iinfo(np.int8).min) and (df[col].max() < np.iinfo(np.int8).max)):

                    df[col] = df[col].astype('int8')

                elif ((df[col].min() > np.iinfo(np.int16).min) and (df[col].max() < np.iinfo(np.int16).max)):

                    df[col] = df[col].astype('int16')

                elif ((df[col].min() > np.iinfo(np.int32).min) and (df[col].max() < np.iinfo(np.int32).max)):

                    df[col] = df[col].astype('int32')

                else:

                    df[col] = df[col].astype('int64')

            else:

                if ((df[col].min() > np.finfo(np.float16).min) and df[col].max() < np.finfo(np.float16).max):

                    df[col] = df[col].astype('float16')

                elif ((df[col].min() > np.finfo(np.float32).min) and df[col].max() < np.finfo(np.float32).max):

                    df[col] = df[col].astype('float32')

                elif ((df[col].min() > np.finfo(np.float64).min) and df[col].max() < np.finfo(np.float64).max):

                    df[col] = df[col].astype('float64')

                else:

                    df[col] = df[col].astype('float128')

    return df
df_train=memory_usage(df_train)

df_train.info()
df_test=memory_usage(df_test)

df_test.info()
X_train=df_train.iloc[:,2:]

Y_train=df_train.iloc[:,1]

ID_train=df_train.iloc[:,0]



X_test=df_test.iloc[:,1:]

ID_test=df_test.iloc[:,0]
X_train.shape, Y_train.shape,ID_train.shape, X_test.shape, ID_test.shape
ID_train.head(),Y_train.head(), X_train.head(), ID_test.head(), X_test.head()
fig, ax= plt.subplots(figsize=(8,8))

sns.countplot(Y_train.values)

ax.set_title('Distribution of target', fontsize=12)

ax.set_xlabel('Index', fontsize=12)

ax.set_ylabel('Target', fontsize=12)

plt.show()
df_null_train=X_train.isna().sum().reset_index()

df_null_train.columns=['Column_name','Null_Count']

df_null_train=df_null_train[df_null_train['Null_Count']>0]
df_null_train
df_null_test=X_test.isna().sum().reset_index()

df_null_test.columns=['Column_name','Null_Count']

df_null_test=df_null_test[df_null_test['Null_Count']>0]
df_null_test
df_Unique_train=X_train.nunique().reset_index()

df_Unique_train.columns=['Column_name','Unique_Count']

df_Unique_train=df_Unique_train[df_Unique_train['Unique_Count']==1]
df_Unique_train
df_Unique_test=X_test.nunique().reset_index()

df_Unique_test.columns=['Column_name','Unique_Count']

df_Unique_test=df_Unique_test[df_Unique_test['Unique_Count']==1]
df_Unique_test
df_corr=X_train.corr().abs().unstack().sort_values().reset_index()

df_corr=df_corr[df_corr['level_0']!= df_corr['level_1']]

df_corr.head()
df_0= df_train[df_train['target']==0]

df_0=df_0.iloc[:, 2:]

df_1= df_train[df_train['target']==1]

df_1=df_1.iloc[:, 2:]

df_0.head(),df_1.head()
# for the first 50 columns, display as 12 rows with 6 columns

fig,ax = plt.subplots(figsize=(25,25))

for i in range(50):

    plt.subplot(12,6,i+1)

    plt.tight_layout()

    sns.kdeplot(df_0.iloc[:,i],bw=0.4,label='0')

    sns.kdeplot(df_1.iloc[:,i],bw=0.4,label='1')

    plt.title('var_'+ str(i), fontsize=9)

plt.show()
# for the next 50 columns, display as 12 rows with 6 columns

fig,ax = plt.subplots(figsize=(25,25))

j=1

for i in range(50,100):

    plt.subplot(12,6,j)

    plt.tight_layout()

    sns.kdeplot(df_0.iloc[:,i],bw=0.4,label='0')

    sns.kdeplot(df_1.iloc[:,i],bw=0.4,label='1')

    plt.title('var_'+str(i), fontsize=9)

    j+=1

plt.show()
# for the next 50 columns, display as 12 rows with 6 columns

fig,ax = plt.subplots(figsize=(25,25))

j=1

for i in range(100,150):

    plt.subplot(12,6,j)

    plt.tight_layout()

    sns.kdeplot(df_0.iloc[:,i],bw=0.4,label='0')

    sns.kdeplot(df_1.iloc[:,i],bw=0.4,label='1')

    plt.title('var_'+str(i), fontsize=9)

    j+=1

plt.show()
# for the next 50 columns, display as 12 rows with 6 columns

fig,ax = plt.subplots(figsize=(25,25))

j=1

for i in range(150,200):

    plt.subplot(12,6,j)

    plt.tight_layout()

    sns.kdeplot(df_0.iloc[:,i],bw=0.4,label='0')

    sns.kdeplot(df_1.iloc[:,i],bw=0.4,label='1')

    plt.title('var_'+str(i), fontsize=9)

    j+=1

plt.show()
# Add more aggregate features to improve the prediction

X_train['sum'] = X_train.sum(axis=1)

X_train['min'] = X_train.min(axis=1)

X_train['max'] = X_train.max(axis=1)

X_train['mean'] = X_train.mean(axis=1)

X_train['median'] = X_train.median(axis=1)

X_train['std'] = X_train.std(axis=1)

X_train['skew'] = X_train.skew(axis=1)

X_train.head()
X_test['sum'] = X_test.sum(axis=1)

X_test['min'] = X_test.min(axis=1)

X_test['max'] = X_test.max(axis=1)

X_test['mean'] = X_test.mean(axis=1)

X_test['median'] = X_test.median(axis=1)

X_test['std'] = X_test.std(axis=1)

X_test['skew'] = X_test.skew(axis=1)

X_test.head()
X_train.shape, X_test.shape
skewed_columns=['var_0', 'var_1', 'var_2', 'var_5', 'var_13', 'var_18','var_19','var_20','var_21', 'var_22',

                'var_24','var_26', 'var_35', 'var_36','var_40','var_44','var_45','var_47','var_48','var_49',

               

                'var_51','var_52','var_54','var_56','var_61', 'var_66','var_67','var_70','var_74','var_75',

                'var_76','var_80','var_81','var_82','var_83','var_86','var_87','var_90', 'var_92','var_94',

                'var_97', 'var_99',

               

                'var_100', 'var_101', 'var_102', 'var_107', 'var_109','var_110', 'var_115', 'var_117', 

                'var_118', 'var_122','var_123', 'var_127', 'var_128', 'var_136', 'var_137', 'var_139',

                'var_147', 'var_149', 

                

                

                'var_150','var_151','var_154','var_155','var_157','var_160','var_163','var_164','var_165',

                'var_167','var_170','var_172','var_173','var_174','var_177','var_178','var_179','var_180',

                'var_184','var_187','var_188','var_190','var_191','var_196','var_198','var_199']
def bin_feature(col_name,X):

    interval=pd.qcut(X[col_name],5).value_counts().index.sort_values()

    X.loc[(X[col_name] <= interval[0].right),col_name] = 0

    X.loc[((X[col_name] > interval[1].left) & (X[col_name] <= interval[1].right)),col_name] = 1

    X.loc[((X[col_name] > interval[2].left) & (X[col_name] <= interval[2].right)),col_name] = 2

    X.loc[((X[col_name] > interval[3].left) & (X[col_name] <= interval[3].right)),col_name] = 3

    X.loc[(X[col_name] > interval[4].left) ,col_name] = 4

    return X
for col in skewed_columns:

    binned_col = col + '_bin'

    X_train[binned_col] = X_train[col]

    X_train[binned_col] = bin_feature(binned_col, X_train)
for col in skewed_columns:

    binned_col = col + '_bin'

    X_test[binned_col] = X_test[col]

    X_test[binned_col] = bin_feature(binned_col, X_test)
X_train.shape, X_test.shape
X_train.head()
from sklearn.model_selection import train_test_split

X_dev, X_val, Y_dev, Y_val=train_test_split(X_train, Y_train, test_size=0.4, random_state=0)

X_dev.shape, X_val.shape, Y_dev.shape, Y_val.shape
from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

import xgboost as xgb

from sklearn.ensemble import GradientBoostingClassifier
def xgb_train(X_train, Y_train, X_test):

    xgb_param = {

                'objective' : "binary:logistic", 

                'eval_metric' : "auc",

                'max_depth' : 6,

                'eta' : 0.1,

                'gamma' : 5,

                'subsample' : 0.7,   

                'colsample_bytree' : 0.7,

                'min_child_weight' : 50,  

                'colsample_bylevel' : 0.7,

                'lambda' : 1, 

                'alpha' : 0,

                'booster' : "gbtree",

                'silent' : 1,

                "random_state": 4950

                }

    n_folds = 10

    oof_train = np.zeros(X_train.shape[0])

    oof_test = np.zeros(X_test.shape[0])

    oof_test_kf = np.empty((n_folds, X_test.shape[0]))

    oof_train.shape, oof_test.shape, oof_test_kf.shape   

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=2019)

    for fold_, (dev_idx, val_idx) in enumerate(kf.split(X_train)):

        print('Fold: ' + str(fold_))

        X_dev, Y_dev = X_train.iloc[dev_idx, :], Y_train[dev_idx]

        X_val, Y_val = X_train.iloc[val_idx, :], Y_train[val_idx]

        xgb_dev = xgb.DMatrix(data=X_dev, label=Y_dev)

        xgb_val = xgb.DMatrix(data=X_val, label=Y_val)

        xgb_test = xgb.DMatrix(X_test)

        watchlist = [(xgb_dev, 'train'), (xgb_val, 'valid')]

        xgb_model = xgb.train(xgb_param, xgb_dev, num_boost_round=1000, evals=watchlist,maximize=True,verbose_eval=50,early_stopping_rounds = 50)

        Y_pred = xgb_model.predict(xgb_val, ntree_limit=xgb_model.best_ntree_limit+50)

        Y_test = xgb_model.predict(xgb_test, ntree_limit=xgb_model.best_ntree_limit+50)

        oof_train[val_idx] = Y_pred

        oof_test_kf[fold_, :] = Y_test

        val_acc = np.round(roc_auc_score(Y_val, Y_pred), 4)

    val_acc = np.round(roc_auc_score(Y_train, oof_train), 4)

    print('Overall Training Accuracy: {0}'.format(val_acc))

    oof_test = oof_test_kf.mean(axis=0)

    return oof_train, oof_test
oof_xgb_train, oof_xgb_test = xgb_train(X_train, Y_train, X_test)
df_sample_submission= pd.read_csv('../input/sample_submission.csv')

df_sample_submission.head()
output=pd.DataFrame({

    "ID_code" : df_sample_submission['ID_code'],

    "target" : oof_xgb_test

})

output.to_csv('SCP_xgb_test.csv', index=False)

output.head()
id_train=df_train['ID_code']

df_out = pd.DataFrame({'id_train':id_train.values, 'target':oof_xgb_train})

df_out.to_csv('SCP_xgb_train.csv', index=False)
roc_auc_score(Y_train, oof_xgb_train)