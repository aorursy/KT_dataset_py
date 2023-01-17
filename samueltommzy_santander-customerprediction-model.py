# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

import sklearn.decomposition as skde

import sklearn.model_selection as ms

from sklearn import linear_model

import sklearn.metrics as sklm

import seaborn as sns

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#import required datasets

test_data = pd.read_csv("../input/test.csv")

train_data= pd.read_csv("../input/train.csv")

train_data.head()
#view Data shape

print(train_data.shape,test_data.shape)

#check for missing values

print(train_data.isnull().values.any(),test_data.isnull().values.any())

#check data correlation 

print(train_data.corr())

#Visualize class distribution

sns.countplot(train_data['target'])
# check class distribution in percentage

count_0 = len(train_data[train_data["target"] == 0])

count_1 = len(train_data[train_data["target"] == 1])

percentage_count_0 = ((count_0)/(count_0+count_1)) * 100

percentage_count_1 = 100-percentage_count_0

print("{}{}{}{}{}".format("Percentage of 0 class is ",percentage_count_0,"\n","Percentage of 1 class is ",percentage_count_1))

labels = train_data["target"]

new_train_data = train_data.drop(["target","ID_code"],axis =1)

new_train_data.head()

x_train, x_test, y_train, y_test = train_test_split(new_train_data, labels, test_size = 0.25, random_state = 0)

print(x_train.shape,x_test.shape)

print(y_train.shape,y_test.shape)

x_train.head()
folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=2319)

param = {

    'bagging_freq': 5, 

    'bagging_fraction': 0.33,

    'boost_from_average':'false',   

    'boost': 'gbdt',

    'feature_fraction': 0.0405,

    'learning_rate': 0.083,

    'max_depth': -1,

    'metric':'auc',

    'min_data_in_leaf': 80,     

    'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 13,

    'num_threads': 4,            

    'tree_learner': 'serial',

    'objective': 'binary',

    'verbosity': 1

}

oof = np.zeros(len(train_data))

predictions = np.zeros(len(test_data))

features = [c for c in train_data.columns if c not in ['ID_code', 'target']]

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_data.values, labels.values)):

    trn_data = lgb.Dataset(train_data.iloc[trn_idx][features], label=labels.iloc[trn_idx])

    val_data = lgb.Dataset(train_data.iloc[val_idx][features], label=labels.iloc[val_idx])

    clf = lgb.train(param, trn_data, 1000, valid_sets = [trn_data, val_data], verbose_eval=5000, early_stopping_rounds = 4000)

    oof[val_idx] = clf.predict(train_data.iloc[val_idx][features], num_iteration=clf.best_iteration)

    predictions += clf.predict(test_data[features], num_iteration=clf.best_iteration) / folds.n_splits

target = train_data.iloc[val_idx]['target']

print("\n >> CV score: {:<8.5f}".format(roc_auc_score(target, oof[val_idx])))

ID_code = test_data["ID_code"]

submission = pd.DataFrame({'ID_code' : ID_code,

                            'target' : predictions})

submission.to_csv('./version1.csv', index=False)

sub = pd.read_csv('./version1.csv')

sub.head()