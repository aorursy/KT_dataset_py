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
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import OneHotEncoder

oh = OneHotEncoder()

import gc
train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')

train
test_id = test.id

train_target = train.target
train.info()
df = pd.concat((train.loc[:,'bin_0':'month'], test.loc[:,'bin_0':'month']))

# before tuning



def basic_details(df):

    b = pd.DataFrame()

    b['Missing value, %'] = round(df.isnull().sum()/df.shape[0]*100)

    b['N unique value'] = df.nunique()

    b['dtype'] = df.dtypes

    return b

basic_details(df)
num_cols = df.select_dtypes(exclude=['object']).columns

fig, ax = plt.subplots(2,3,figsize=(22,7))

for i, col in enumerate(num_cols):

    plt.subplot(2,3,i+1)

    plt.xlabel(col, fontsize=9)

    sns.kdeplot(df[col].values, bw=0.5)  

plt.show() 
# simplest NaN imputation



for col in df:

    if df[col].dtype == 'object':        

        df[col].fillna('N', inplace=True)

    else: df[col].fillna(-10000, inplace=True)
def basic_details(df):

    b = pd.DataFrame()

    b['Missing value, %'] = round(df.isnull().sum()/df.shape[0]*100)

    b['N unique value'] = df.nunique()

    b['dtype'] = df.dtypes

    return b

basic_details(df)
# #label encoding for high shape object columns    

# high_shape_features = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_5']

# le = LabelEncoder()

# for col in high_shape_features:

#     df[col] = le.fit_transform(df[col].astype(str))



# # one hot encoding of low shape features

    

# for col in df:

#     if df[col].nunique()<=27:

#         df[col] = df[col].astype(str)

# df = pd.get_dummies(df)
columns = [i for i in df.columns]

dummies = pd.get_dummies(df,columns=columns, drop_first=True,sparse=True)
train = dummies.iloc[:train.shape[0], :]

test = dummies.iloc[train.shape[0]:, :]
train = train.sparse.to_coo().tocsr()

test = test.sparse.to_coo().tocsr()
#creating matrices for feature selection:

X = train

y = train_target

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score





x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=10)



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)

d_test = xgb.DMatrix(test)



params = {

        'objective':'binary:logistic',

        'eta': 0.3,

        'tree_method':'gpu_hist',

        'max_depth':2,

        'learning_rate':0.03,

        'eval_metric':'auc',

        'min_child_weight':3,

        'subsample':0.9,

        'colsample_bytree':0.59,

        'seed':29,

        'reg_lambda':0.8,

        'reg_alpha':0.000001,

        'gamma':0.1,

        'scale_pos_weight':2.5,

        'nthread':-1

}



watchlist = [(d_train, 'train'), (d_valid, 'valid')]

nrounds=100000 

model = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=500, maximize=True, verbose_eval=10)

p_test = model.predict(d_test)


plt.style.use('fivethirtyeight')

fig,ax = plt.subplots(figsize=(30,20))

xgb.plot_importance(model,ax=ax,height=0.8,color='r')

plt.tight_layout()

plt.show()
sub = pd.DataFrame({'id':test_id,'target':p_test})



sub.to_csv('submission.csv',index=False)