# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import gc, math



import matplotlib.gridspec as gridspec # to do the grid of plots



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sns.set(rc={'figure.figsize':(11,8)})

sns.set(style="whitegrid")
train_df = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')

test_df = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')

sample_sub_df = pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv')
print (f'Shape of training data: {train_df.shape}')

print (f'Shape of testing data: {test_df.shape}')
train_df.head()
test_df.head()
train_df.columns
train_df.dtypes
not train_df.isna().sum().values.any()
not test_df.isna().sum().values.any()
## Function to reduce the memory usage

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

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train_df = reduce_mem_usage(train_df)

test_df = reduce_mem_usage(test_df)
total = len(train_df)



ax = sns.barplot(pd.unique(train_df['target']), train_df['target'].value_counts())

ax.set(xlabel='Target Type', ylabel='# of records', title='Tsrget Distribution')

plt.show()
def bin_feature_transform(df):

    feature_map = {

        'T': 1,

        'Y': 1,

        'F': 0,

        'N': 0

    }

    df['bin_3'] = df['bin_3'].map(feature_map)

    df['bin_4'] = df['bin_4'].map(feature_map)

    return df
train_df = bin_feature_transform(train_df)

test_df = bin_feature_transform(test_df)
train_df = reduce_mem_usage(train_df)

test_df = reduce_mem_usage(test_df)
grid = gridspec.GridSpec(3, 2) # The grid of chart

plt.figure(figsize=(16,20)) # size of figure



bin_cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']

val_num_map = {0: 0, 1: 1}





for n, col in enumerate(train_df[bin_cols]):

    ax = plt.subplot(grid[n]) # feeding the figure of grid

    ax = sns.barplot(np.vectorize(val_num_map.get)(pd.unique(train_df[col])), train_df[col].value_counts())

    ax.set(xlabel=f'Feature: {col}', ylabel='# of records', title=f'Binary feature {n} vs. # of records')

    sizes = []

    for p in ax.patches: # loop to all objects

        height = p.get_height()

        sizes.append(height)

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/total*100),

                ha="center", fontsize=14) 



    

plt.show()
train_df.groupby('nom_9')['id'].nunique().shape
grid = gridspec.GridSpec(5, 5) # The grid of chart

plt.figure(figsize=(30,30)) # size of figure



nom_cols = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']

val_num_map = {0: 0, 1: 1}



for n, col in enumerate(train_df[nom_cols]):

    ax = plt.subplot(grid[n]) # feeding the figure of grid

    ax = sns.barplot(train_df.groupby(col)['id'].nunique().keys(), train_df.groupby(col)['id'].nunique())

    ax.set(xlabel = f'Feature: {col}', ylabel='# of records', title=f'Nominal feature {n} vs. # of records')

    sizes = []

    for p in ax.patches: # loop to all objects

        height = p.get_height()

        sizes.append(height)

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/total*100),

                ha="center", fontsize=14) 



    

plt.show()
low_card_nom_cols=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']

train_df = pd.get_dummies(train_df, columns=low_card_nom_cols)

test_df = pd.get_dummies(test_df, columns=low_card_nom_cols)
train_df.head()
train_df.columns
test_df.columns
print (train_df.shape)

print (test_df.shape)
high_card_nom_cols = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

for col in high_card_nom_cols:

    train_df[f'hash_{col}'] = train_df[col].apply(lambda x: hash(str(x))%5000)

    test_df[f'hash_{col}'] = test_df[col].apply(lambda x: hash(str(x))%5000)
train_df = train_df.drop(high_card_nom_cols, axis=1)

test_df = test_df.drop(high_card_nom_cols, axis=1)
train_df.head()
ord_features = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']

train_df[ord_features].head()
train_df[ord_features].nunique()
grid = gridspec.GridSpec(2, 3) # The grid of chart

plt.figure(figsize=(40,40)) # size of figure



for n, col in enumerate(train_df[ord_features[:-3]]):

    ax = plt.subplot(grid[n]) # feeding the figure of grid

    ax = sns.barplot(train_df.groupby(col)['id'].nunique().keys(), train_df.groupby(col)['id'].nunique())

    ax.set(xlabel = f'Feature: {col}', ylabel='# of records', title=f'Ordinal feature {n} vs. # of records')

    sizes = []

    for p in ax.patches: # loop to all objects

        height = p.get_height()

        sizes.append(height)

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/total*100),

                ha="center", fontsize=30) 



    

plt.show()
# Importing categorical options of pandas

from pandas.api.types import CategoricalDtype 



# seting the orders of our ordinal features

ord_1 = CategoricalDtype(categories=['Novice', 'Contributor','Expert', 

                                     'Master', 'Grandmaster'], ordered=True)

ord_2 = CategoricalDtype(categories=['Freezing', 'Cold', 'Warm', 'Hot',

                                     'Boiling Hot', 'Lava Hot'], ordered=True)

ord_3 = CategoricalDtype(categories=['a', 'b', 'c', 'd', 'e', 'f', 'g',

                                     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'], ordered=True)

ord_4 = CategoricalDtype(categories=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',

                                     'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',

                                     'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], ordered=True)
# Transforming ordinal Features for train dataset

train_df.ord_1 = train_df.ord_1.astype(ord_1)

train_df.ord_2 = train_df.ord_2.astype(ord_2)

train_df.ord_3 = train_df.ord_3.astype(ord_3)

train_df.ord_4 = train_df.ord_4.astype(ord_4)



# Same test dataset

test_df.ord_1 = test_df.ord_1.astype(ord_1)

test_df.ord_2 = test_df.ord_2.astype(ord_2)

test_df.ord_3 = test_df.ord_3.astype(ord_3)

test_df.ord_4 = test_df.ord_4.astype(ord_4)
train_df['ord_1'].head()
# Geting the codes of ordinal categoy's - train

train_df.ord_1 = train_df.ord_1.cat.codes

train_df.ord_2 = train_df.ord_2.cat.codes

train_df.ord_3 = train_df.ord_3.cat.codes

train_df.ord_4 = train_df.ord_4.cat.codes



# Geting the codes of ordinal categoy's - test

test_df.ord_1 = test_df.ord_1.cat.codes

test_df.ord_2 = test_df.ord_2.cat.codes

test_df.ord_3 = test_df.ord_3.cat.codes

test_df.ord_4 = test_df.ord_4.cat.codes
train_df.head()
from sklearn.preprocessing import OrdinalEncoder



oe = OrdinalEncoder()

oe.fit(train_df['ord_5'].values.reshape(-1, 1))

oe.categories_
encoded_train = oe.transform(train_df['ord_5'].values.reshape(-1, 1))

encoded_test = oe.transform(test_df['ord_5'].values.reshape(-1, 1))
train_df
train_df['ord_5'] = encoded_train

test_df['ord_5'] = encoded_test
def encode_cyclic_feature(df, col, max_vals):

    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_vals)

    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_vals)

    df = df.drop(col, axis=1)

    return df



train_df = encode_cyclic_feature(train_df, 'day', 7)

test_df = encode_cyclic_feature(test_df, 'day', 7) 



train_df = encode_cyclic_feature(train_df, 'month', 12)

test_df = encode_cyclic_feature(test_df, 'month', 12)
# Drop ID columns from both train and test dataset as it's not a feature

train_df = train_df.drop(['id'], axis=1)

test_df = test_df.drop(['id'], axis=1)
Y_train = train_df['target']

X_train = train_df.drop(['target'], axis=1)
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.svm import SVC

from sklearn.model_selection import KFold
ntrain = train_df.shape[0]

ntest = test_df.shape[0]

SEED = 666

NFOLDS = 5

kf = KFold(n_splits = NFOLDS, shuffle=True, random_state=SEED)



class StackingHelper(object):

    def __init__(self, clf, seed=0, params=None):

        params['random_state'] = seed

        self.clf = clf(**params)

    

    def train(self, X_train, Y_train):

        self.clf.fit(X_train, Y_train)

        

    def predict(self, X):

        return self.clf.predict_proba(X)
# Get out of fold predictions

def get_oof(clf, X_train, Y_train, X_test):

    oof_train = np.zeros((ntrain, ))

    oof_test = np.zeros((ntest, ))

    oof_test_skf = np.empty((NFOLDS, ntest))

    

    for i, (train_index, test_index) in enumerate(kf.split(X_train, Y_train)):

        x_train = X_train.iloc[train_index]

        y_train = Y_train.iloc[train_index]

        x_test = X_train.iloc[test_index]

        

        clf.train(x_train, y_train)

        

        oof_train[test_index] = clf.predict(x_test)[:, 1]

        oof_test_skf[i: ] = clf.predict(X_test)[:, 1]

        

    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
# Put in our parameters for said classifiers

# Random Forest parameters

rf_params = {

    'n_jobs': -1,

    'n_estimators': 500,

     'warm_start': True, 

    'max_depth': 6,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt'

}



# Extra Trees Parameters

et_params = {

    'n_jobs': -1,

    'n_estimators':500,

    'max_depth': 8,

    'min_samples_leaf': 2

}



# AdaBoost parameters

ada_params = {

    'n_estimators': 500,

    'learning_rate' : 0.75

}



# Gradient Boosting parameters

gb_params = {

    'n_estimators': 500,

    'max_depth': 5,

    'min_samples_leaf': 2

}



# Support Vector Classifier parameters 

svc_params = {

    'kernel' : 'linear',

    'C' : 0.025

}
# Create 5 objects that represent our 4 models

rf = StackingHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

et = StackingHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)

ada = StackingHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)

gb = StackingHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)

svc = StackingHelper(clf=SVC, seed=SEED, params=svc_params)
# Create our OOF train and test predictions. These base results will be used as new features

et_oof_train, et_oof_test = get_oof(et, X_train, Y_train, test_df) # Extra Trees

rf_oof_train, rf_oof_test = get_oof(rf,X_train, Y_train, test_df) # Random Forest

ada_oof_train, ada_oof_test = get_oof(ada, X_train, Y_train, test_df) # AdaBoost 

gb_oof_train, gb_oof_test = get_oof(gb,X_train, Y_train, test_df) # Gradient Boost

svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier



print("Training is complete")
base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),

     'ExtraTrees': et_oof_train.ravel(),

     'AdaBoost': ada_oof_train.ravel(),

      'GradientBoost': gb_oof_train.ravel()

    })

base_predictions_train.head()
from sklearn.metrics import roc_auc_score

print (roc_auc_score(Y_train, et_oof_train))

print (roc_auc_score(Y_train, rf_oof_train))

print (roc_auc_score(Y_train, ada_oof_train))

print (roc_auc_score(Y_train, gb_oof_train))
print (gb.predict(test_df)[:, 1])

print (ada.predict(test_df)[:, 1])

print (rf.predict(test_df)[:, 1])

print (et.predict(test_df)[:, 1])

x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train), axis=1)

x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test), axis=1)
import xgboost as xgb

gbm = xgb.XGBClassifier(

    #learning_rate = 0.02,

 n_estimators= 2000,

 max_depth= 4,

 min_child_weight= 2,

 #gamma=1,

 gamma=0.9,                        

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread= -1,

 scale_pos_weight=1).fit(x_train, Y_train)

predictions = gbm.predict_proba(x_test)
predictions
sample_sub_df['target'] = predictions[:, 1]
sample_sub_df.to_csv('submission.csv', index=False)

sample_sub_df
from IPython.display import FileLink, FileLinks

FileLink('submission.csv')