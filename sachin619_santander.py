# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train=reduce_mem_usage(pd.read_csv("../input/train.csv"))

test=reduce_mem_usage(pd.read_csv("../input/test.csv"))
train.head()
train.describe()
train.isnull().sum() 
test.isnull().sum() 
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC, LinearSVC



#from catboost import CatBoostClassifier

#from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from lightgbm import LGBMRegressor

# Modelling Helpers

from sklearn.preprocessing import Imputer , Normalizer , scale

from sklearn.feature_selection import RFECV

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split



# Visualisation

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

#import seaborn as sns



# Configure visualisations

%matplotlib inline

mpl.style.use( 'ggplot' )

#sns.set_style( 'white' )

pylab.rcParams[ 'figure.figsize' ] = 8 , 6



#import networkx as nx
train['target'].value_counts().plot.bar();
import seaborn as sns



f,ax=plt.subplots(1,2,figsize=(18,8))

train['target'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('target')

ax[0].set_ylabel('')

sns.countplot('target',data=train,ax=ax[1])

ax[1].set_title('target')

plt.show()
y=train["target"]

train=train.drop("target",axis=1)
train=train.drop("ID_code",axis=1)

test=test.drop("ID_code",axis=1)
train_X, val_X, train_y, val_y = train_test_split(train, y, random_state=1)

rfc_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(rfc_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist(), top=150)
import lightgbm as lgb



#d_train=lgb.Dataset(chunk,label=target)

params = {'num_leaves': 32,

         'min_data_in_leaf': 30, 

         'objective':'binary',

         'max_depth': -1,

         'metric': 'auc',

         'learning_rate': 0.012,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.92,

         "bagging_seed": 11,

         

         "lambda_l1": 0.1,

         "verbosity": -1,

         "nthread": 4,

         }
X_train,X_test,y_train,y_test=train_test_split(train,y,test_size=0.1,random_state=0)

d_train=lgb.Dataset(X_train,label=y_train)

test_data=lgb.Dataset(X_test,label=y_test)

clf=lgb.train(params,d_train,valid_sets=test_data,num_boost_round=3000,early_stopping_rounds=100)
y_pred=clf.predict(test,num_iteration=clf.best_iteration)
submission=pd.read_csv("../input/sample_submission.csv")
submission["target"]=y_pred
submission.to_csv("santa1.csv",index=False)