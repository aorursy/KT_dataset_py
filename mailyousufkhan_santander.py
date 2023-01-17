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
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

import lightgbm as lgb

import seaborn as sns

import warnings

import time
fold_n=5

folds = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=10)

%matplotlib inline

%precision 4

warnings.filterwarnings('ignore')

plt.style.use('ggplot')

np.set_printoptions(suppress=True)

pd.set_option("display.precision", 15)
train= pd.read_csv("../input/train.csv")

test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')

sample_submission.head()
train.shape, test.shape, sample_submission.shape
train.columns
train.describe()
train.isnull().values.any()
test.isnull().values.any()
train['target'].value_counts().plot.barh()
f,ax=plt.subplots(1,2,figsize=(18,8))

train['target'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('target')

ax[0].set_ylabel('')

sns.countplot('target',data=train,ax=ax[1])

ax[1].set_title('target')

plt.show()
cols=["target","ID_code"]

X = train.drop(cols,axis=1)

y = train["target"]
X.head()
y.head()
X_test  = test.drop("ID_code",axis=1)
X_test.head()
params = {'objective' : "binary", 

               'boost':"gbdt",

               'metric':"auc",

               'boost_from_average':"false",

               'num_threads':8,

               'learning_rate' : 0.01,

               'num_leaves' : 13,

               'max_depth':-1,

               'tree_learner' : "serial",

               'feature_fraction' : 0.05,

               'bagging_freq' : 5,

               'bagging_fraction' : 0.4,

               'min_data_in_leaf' : 80,

               'min_sum_hessian_in_leaf' : 10.0,

               'verbosity' : 1}
%%time

y_pred_lgb = np.zeros(len(X_test))

num_round = 1000000

for fold_n, (train_index, valid_index) in enumerate(folds.split(X,y)):

    print('Fold', fold_n, 'started at', time.ctime())

    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

    train_data = lgb.Dataset(X_train, label=y_train)

    valid_data = lgb.Dataset(X_valid, label=y_valid)

        

    lgb_model = lgb.train(params,train_data,num_round,#change 20 to 2000

                    valid_sets = [train_data, valid_data],verbose_eval=1000,early_stopping_rounds = 3500)##change 10 to 200

            

    y_pred_lgb += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)/5
submission_lgb = pd.DataFrame({

        "ID_code": test["ID_code"],

        "target": y_pred_lgb

    })

submission_lgb.to_csv('submission_lgb.csv', index=False)