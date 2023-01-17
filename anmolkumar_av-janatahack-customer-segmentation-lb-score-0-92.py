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
from numpy import mean, std

import seaborn as sns

from matplotlib import *

from matplotlib import pyplot as plt

from catboost import CatBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, RepeatedStratifiedKFold

from sklearn.metrics import accuracy_score

from lightgbm import LGBMClassifier
train_data = pd.read_csv('/kaggle/input/automobile-customer-segmentation/train.csv')

test_data = pd.read_csv('/kaggle/input/automobile-customer-segmentation/test.csv')

sample_submission = pd.read_csv('/kaggle/input/automobile-customer-segmentation/sample_submission.csv')

train_data.columns = train_data.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

test_data.columns = test_data.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
print('Train Data Shape: ', train_data.shape)

print('Test Data Shape: ', test_data.shape)

train_data.head()
train_data.dtypes
train_data.isnull().sum()
# Unique values for all the columns

for col in ['gender', 'ever_married', 'graduated', 'profession', 'spending_score', 'family_size', 'var_1']:

    print(" Unique Values --> " + col, ':', len(train_data[col].unique()), ': ', train_data[col].unique())
# Value counts for the columns

for col in ['segmentation', 'gender', 'ever_married', 'graduated', 'profession', 'spending_score', 'family_size', 'var_1']:

    print(col + ": \n", train_data[col].value_counts(), '\n')
i = 1

for column in train_data.columns[~(train_data.columns.isin(['age', 'id', 'work_experience', 'family_size']))].tolist():

    plt.figure(figsize = (40, 10))

    plt.subplot(3, 3, i)

    sns.barplot(x = train_data[column].value_counts().index, y = train_data[column].value_counts())

    i += 1

    plt.show()
sns.boxplot(x = 'age', data = train_data)

sns.despine()
plt.figure(figsize = (20, 6))

sns.barplot(x = train_data.groupby(['profession'])['spending_score'].value_counts().index, y = train_data.groupby(['profession'])['spending_score'].value_counts())

plt.xticks(rotation = 90)

sns.despine()
train_data['type'] = 'train'

test_data['type'] = 'test'

train_data = train_data.drop('age', axis = 1)

test_data = test_data.drop('age', axis = 1)

master_data = pd.concat([train_data, test_data])

train_seg = train_data[['id', 'segmentation']]

train_seg.columns = ['id', 'assumed_seg']

master_data = master_data.merge(train_seg, on = 'id', how = 'left')

#master_data = master_data.sort_values(['id', 'type'], ascending = [True, False])

master_data.head(20)
# master_data['age_bckt'] = pd.cut(x = master_data['age'], bins = [0, 22, 32, 42, 52, 62, 90], labels = ['1', '2', '3', '4', '5', '6'])



le = LabelEncoder()

cat_cols = ['gender', 'ever_married', 'graduated', 'profession', 'spending_score', 'var_1', 'assumed_seg']



for col in cat_cols:

    master_data[col] = master_data[col].astype(str)

    LE = le.fit(master_data[col])

    master_data[col] = LE.transform(master_data[col])

    

train_data = master_data.loc[master_data['type'] == 'train']

test_data = master_data.loc[master_data['type'] == 'test']



testIDs = test_data.id.values



train_data = train_data.drop(['id', 'type', 'family_size', 'work_experience'], axis = 1)

test_data = test_data.drop(['id', 'segmentation', 'type', 'family_size', 'work_experience'], axis = 1)



train_data = train_data.fillna('NaN')

test_data = test_data.fillna('NaN')



# Partitioning the features and the target



X = train_data[train_data.columns[~(train_data.columns.isin(['segmentation']))].tolist()].values

y = train_data['segmentation'].values



train_data.head()
model = LGBMClassifier()

cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 22)

n_scores = cross_val_score(model, X, y, scoring = 'accuracy', cv = cv, n_jobs = -1, error_score = 'raise')

print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

# fit the model on the whole dataset

model = LGBMClassifier()

model.fit(X, y)

# make a single prediction

yhat = (model.predict(test_data)).ravel()

print('Prediction: ', yhat)
pred = pd.DataFrame()

#pred['ID'] = test_data['id'].values

pred['ID'] = testIDs

pred['Segmentation'] = pd.Series((model.predict(test_data)).ravel())

pred.to_csv('lgbm_v1.csv', index = None)
kfold, scores = KFold(n_splits = 6, shuffle = True, random_state = 22), list()

for train, test in kfold.split(X):

    X_train, X_test = X[train], X[test]

    y_train, y_test = y[train], y[test]

    

    model = CatBoostClassifier(random_state = 22, max_depth = 6, n_estimators = 200, verbose = 100)

    model.fit(X_train, y_train, cat_features = [0,1,2,3,4,5,6])

    preds = model.predict(X_test)

    score = accuracy_score(y_test, preds)

    scores.append(score)

    print('Validation Accuracy:', score)

print("Average Validation Accuracy: ", sum(scores)/len(scores))
pred = pd.DataFrame()

#pred['ID'] = test_data['id'].values

pred['ID'] = testIDs

pred['Segmentation'] = pd.Series((model.predict(test_data)).ravel())

pred.to_csv('catboost_v1.csv', index = None)