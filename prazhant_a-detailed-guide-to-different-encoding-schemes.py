import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy as sp

from scipy import stats

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Load data

train = pd.read_csv('../input/cat-in-the-dat/train.csv')

test = pd.read_csv('../input/cat-in-the-dat/test.csv')

sub = pd.read_csv('../input/cat-in-the-dat/sample_submission.csv')

print(train.shape)

print(test.shape)
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

train.head()
bin_cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']

# loop to get column and the count of plots

for n, col in enumerate(train[bin_cols]): 

    plt.figure(n)

    sns.countplot(x=col, data=train, hue='target', palette='husl')
train['bin_3'] = train['bin_3'].replace(to_replace=['F', 'T'], value=['0', '1']).astype(int)

train['bin_4'] = train['bin_4'].replace(to_replace=['Y', 'N'], value=['1', '0']).astype(int)

test['bin_3'] = test['bin_3'].replace(to_replace=['F', 'T'], value=['0', '1']).astype(int)

test['bin_4'] = test['bin_4'].replace(to_replace=['Y', 'N'], value=['1', '0']).astype(int)

# train['bin_3'].astype(int)

# train['bin_4'].astype(int)
train.head(3)
#Drop ID and seperate target variable

target = train['target']

train_id = train['id']

test_id = test['id']

train.drop(['target', 'id'], axis=1, inplace=True)

test.drop('id', axis=1, inplace=True)



print(train.shape)

print(test.shape)
ord_cols = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']



for i in ord_cols:

    print("The number of unique values in {} column is : {}".format(i, train[i].nunique()))

    print("The unique values in {} column is : \n {}".format(i, train[i].value_counts()[:7]))

    print('\n')
# credits for mapper code : https://www.kaggle.com/gogo827jz/catboost-baseline-with-feature-importance



mapper_ord_1 = {'Novice': 1, 'Contributor': 2, 'Expert': 3, 'Master': 4, 'Grandmaster': 5}



mapper_ord_2 = {'Freezing': 1, 'Cold': 2, 'Warm': 3, 'Hot': 4,'Boiling Hot': 5, 'Lava Hot': 6}



mapper_ord_3 = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 

                'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15}



mapper_ord_4 = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 

                'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15,

                'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 

                'W': 23, 'X': 24, 'Y': 25, 'Z': 26}



for col, mapper in zip(['ord_1', 'ord_2', 'ord_3', 'ord_4'], [mapper_ord_1, mapper_ord_2, mapper_ord_3, mapper_ord_4]):

    train[col+'_oe'] = train[col].replace(mapper)

    test[col+'_oe'] = test[col].replace(mapper)

    train.drop(col, axis=1, inplace=True)

    test.drop(col, axis=1, inplace=True)
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(categories='auto')

encoder.fit(train.ord_5.values.reshape(-1, 1))

train.ord_5 = encoder.transform(train.ord_5.values.reshape(-1, 1))

test.ord_5 = encoder.transform(test.ord_5.values.reshape(-1, 1))
train.ord_5[:5]
train[['ord_1_oe','ord_2_oe','ord_3_oe','ord_4_oe','ord_5','ord_0']].info()
nom_cols = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']



for i in nom_cols:

    print("The number of unique values in {} column is : {}".format(i, train[i].nunique()) )

        
%%time

from sklearn.preprocessing import OneHotEncoder

one=OneHotEncoder()

train_ohe1 = one.fit_transform(train)

test_ohe1 = one.fit_transform(test)

# ohe_obj_train = one.fit(train)

# ohe_obj_test = one.fit(test)



# train_ohe1 = ohe_obj_train.transform(train)

# test_ohe1 = ohe_obj_test.transform(test)

print(train_ohe1.shape)

print(train_ohe1.dtype)

print(test_ohe1.shape)

print(test_ohe1.dtype)
# %%time

# nom_col = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8','nom_9']



# traintest = pd.concat([train, test])

# traintest_ohe = pd.get_dummies(traintest, columns=nom_col, drop_first=True, sparse=True)

# train_ohe = traintest_ohe.iloc[:train.shape[0], :]

# test_ohe = traintest_ohe.iloc[train.shape[0]:, :]



# print(train_ohe.shape)

# print(test_ohe.shape)
def logistic(X,y):

    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

    lr=LogisticRegression()

    lr.fit(X_train,y_train)

    y_pre=lr.predict(X_test)

    print('Accuracy : ',accuracy_score(y_test,y_pre))
logistic(train_ohe1,target)
%%time

nom_col = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8','nom_9']

from sklearn import model_selection, preprocessing, metrics

le = preprocessing.LabelEncoder()

traintest = pd.concat([train, test])



for col in nom_col:

    traintest[col] = le.fit_transform(traintest[col])



train_le = traintest.iloc[:train.shape[0], :]

test_le = traintest.iloc[train.shape[0]:, :]



print(train_le.shape)

print(test_le.shape)
train_le.head()
logistic(train_le,target)
# LGBM on LabelEncoded data

import lightgbm as lgb

num_round = 50000



param = {'num_leaves': 64,

         'min_data_in_leaf': 30, 

         'objective':'binary',

         'max_depth': 5,

         'learning_rate': 0.001,

         "min_child_samples": 20,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9 ,

         "bagging_seed": 44,

         "metric": 'auc',

         "verbosity": -1}



X_train,X_test,y_train,y_test=train_test_split(train_le,target,random_state=42,test_size=0.2)



train = lgb.Dataset(X_train, label=y_train)

test = lgb.Dataset(X_test, label=y_test)



clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=50, 

                early_stopping_rounds = 500)

X_train_ohe,X_test_ohe,y_train_ohe,y_test_ohe=train_test_split(train_ohe1,target,random_state=42,test_size=0.2)



train = lgb.Dataset(X_train_ohe, label=y_train_ohe)

test = lgb.Dataset(X_test_ohe, label=y_test_ohe)



clf_ohe = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=50, 

                early_stopping_rounds = 500)

y_preds = clf.predict(test_le)
sub['target'] = y_preds

sub.to_csv('lgb_model.csv', index=False)