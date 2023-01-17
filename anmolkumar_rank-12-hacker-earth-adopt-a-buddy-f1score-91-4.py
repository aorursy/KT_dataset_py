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
# Import useful libraries

import time
import re
import string
from numpy import mean
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest, f_classif

import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, KFold, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.multiclass import type_of_target

from catboost import CatBoostClassifier
from collections import Counter

import warnings
warnings.filterwarnings('ignore')
# Read dataset

train_data = pd.read_csv('/kaggle/input/hackerearth-ml-challenge-adopt-a-buddy/train.csv')
test_data = pd.read_csv('/kaggle/input/hackerearth-ml-challenge-adopt-a-buddy/test.csv')
train_data.columns = train_data.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
test_data.columns = test_data.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
print('Train Data Shape: ', train_data.shape)
print('Test Data Shape: ', test_data.shape)
train_data.head()
train_data.isnull().sum()
# See the distribution of outcome 1: breed_category

sns.countplot(x = 'breed_category',data = train_data)
sns.despine()
print(train_data.breed_category.value_counts())
print(train_data.pet_category.value_counts())
# See the distribution of outcome 2: pet_category

sns.countplot(x = 'pet_category',data = train_data)
sns.despine()
train_data['type'] = 'train'
test_data['type'] = 'test'

master_data = pd.concat([train_data, test_data])
master_data['issue_date'] = pd.to_datetime(master_data['issue_date'], dayfirst = True)
master_data['listing_date'] = pd.to_datetime(master_data['listing_date'].apply(lambda x: x.split(' ')[0]), dayfirst = True)
# Relation between length and breed category

plt.figure(figsize = (8, 6))
sns.boxplot(x = 'breed_category',y = 'lengthm',data = master_data)
plt.show()
# Relation between length and pet category

plt.figure(figsize = (8, 6))
sns.boxplot(x = 'pet_category',y = 'lengthm',data = master_data)
plt.show()
# See the distribution of outcome 2: pet_category

plt.figure(figsize = (8, 6))
sns.countplot(x = 'condition',data = master_data)
sns.despine()
# See the distribution of outcome 1: breed_category

plt.figure(figsize = (22, 5))
sns.countplot(x = 'color_type',data = master_data)
plt.xticks(rotation = 90)
plt.show()
plt.figure(figsize = (8, 6))
sns.scatterplot(x = master_data['lengthm'], y = master_data['heightcm']/100)
plt.show()
plt.figure(figsize = (8, 5))
sns.distplot(master_data['lengthm'])
plt.show()
plt.figure(figsize = (8, 5))
df = master_data[['lengthm','heightcm']]
df['lengthcm'] = df['lengthm']*100
df[['lengthcm','heightcm']].boxplot()
plt.show()
# Correlation matrix

plt.figure(figsize = (11, 10))
#plt.subplots(figsize=(10,8))
sns.heatmap(master_data.corr(), annot = True)
master_data['days_to_reach'] = master_data['listing_date'] - master_data['issue_date']
master_data['days_to_reach'] = master_data['days_to_reach'].apply(lambda x: int(str(x).split(' ')[0]))

master_data['age'] = master_data['days_to_reach'] / 365
master_data['age'] = master_data['age'].abs()

# Mapping for condition of pets

condition = {0.0: 'A', 1.0: 'B', 2.0: 'C'}
master_data['condition'] = master_data['condition'].map(condition)
master_data['condition'] = master_data['condition'].astype(str)

# Convert height to cms

master_data['heightm'] = master_data['heightcm'] / 100
master_data = master_data.drop(['heightcm'], axis = 1)

#length_mean = master_data['lengthm'].mean()

master_data.loc[(master_data['lengthm'] == 0), 'lengthm'] = 0.005

#master_data['len_to_height'] = master_data['lengthm']/master_data['heightm']

master_data['color_type'] = master_data['color_type'].apply(lambda x: x.lower())



master_data.head()
# 2 records exist where the listing date is less than the issue date, convert them to positive

master_data.loc[(master_data['days_to_reach'] <= 0), 'days_to_reach'] = master_data.loc[(master_data['days_to_reach'] < 0), 'days_to_reach'] * -1
# Generate master color feature from the available color_type

master_data['master_color'] = master_data['color_type'].apply(lambda x: x.split(' ')[0])
master_data['species'] = master_data['color_type'].apply(lambda x: x.split(' ')[1] if len(x.split(' ')) == 2 else x.split(' ')[0])
# Generate time features - e.g. Quarter

master_data['issue_qtr'] = master_data['issue_date'].dt.quarter
master_data['list_qtr'] = master_data['listing_date'].dt.quarter

master_data['issue_yr'] = master_data['issue_date'].dt.year
master_data['list_yr'] = master_data['listing_date'].dt.year

master_data['issue_mth'] = master_data['issue_date'].dt.month
master_data['list_mth'] = master_data['listing_date'].dt.month

master_data['issue_weekend'] = master_data['issue_date'].apply(lambda x: 1 if x.dayofweek in [5, 6] else 0)
master_data['list_weekend'] = master_data['listing_date'].apply(lambda x: 1 if x.dayofweek in [5, 6] else 0)

master_data.head()
# Get numerical columns

cat_cols = ['condition', 'color_type', 'master_color', 'species']
numerical_cols = master_data.columns[~master_data.columns.isin(cat_cols + ['pet_id', 'issue_date', 'listing_date', 'type', 'breed_category', 'pet_category'])].tolist()
numerical_cols
ss = StandardScaler()
master_data[numerical_cols] = ss.fit_transform(master_data[numerical_cols])
#le = LabelEncoder()
#for col in cat_cols:
#    master_data[col] = le.fit_transform(master_data[col])
    
#master_data[numerical_cols + cat_cols] = ss.fit_transform(master_data[numerical_cols + cat_cols])
train_data.columns
# Separate train and test data

train_data = master_data.loc[master_data['type'] == 'train']
test_data = master_data.loc[master_data['type'] == 'test']

train_data['breed_category'] =train_data['breed_category'].astype(str)
train_data['pet_category'] =train_data['pet_category'].astype(str)

testIDs = test_data['pet_id']

train_data = train_data.drop(['pet_id', 'issue_date', 'listing_date', 'type'], axis = 1)

for col in ['breed_category', 'pet_category']:
    train_data[col] = train_data[col].apply(lambda x: np.float16(x))
    train_data[col] = train_data[col].apply(lambda x: np.int8(x))

testData = test_data.drop(['issue_date', 'listing_date', 'type', 'x2', 'breed_category', 'pet_category'], axis = 1)
test_data = test_data.drop(['pet_id', 'issue_date', 'listing_date', 'type', 'x2', 'breed_category', 'pet_category'], axis = 1)
train_data = train_data[['condition', 'color_type', 'lengthm', 'x1', 'days_to_reach', 'age', 'heightm', 'master_color', 'species', 'issue_qtr', 'list_qtr',
                         'breed_category', 'issue_yr', 'list_yr', 'issue_mth', 'list_mth', 'issue_weekend', 'list_weekend', 'pet_category']]
train_data.head()
test_data.head()
train_data_1 = train_data.copy()
"""
X1 = train_data_1.drop(['breed_category', 'pet_category'],axis = 1).values
y1 = train_data_1['pet_category'].values

for num_feats in range(1, 9):
    print('Using {} features:'.format(num_feats))
    test = SelectKBest(score_func = f_classif, k = num_feats)
    fit = test.fit(X1, y1)
    # summarize scores
    set_printoptions(precision = 0)
    for i in fit.scores_:
        print(i)
    print(fit.scores_)
    features = fit.transform(X1)
    # summarize selected features
    print(features[0:num_feats + 1, :])
"""
"""
X2 = train_data_1.drop(['breed_category'],axis = 1).values
y2 = train_data_1['breed_category'].values

for num_feats in range(1, 10):
    print('Using {} features:'.format(num_feats))
    test = SelectKBest(score_func = f_classif, k = num_feats)
    fit = test.fit(X2, y2)
    # summarize scores
    set_printoptions(precision = 0)
    for i in fit.scores_:
        print(i)
    print(fit.scores_)
    features = fit.transform(X2)
    # summarize selected features
    print(features[0:num_feats + 1, :])
"""
train_data_1 = train_data.drop(['lengthm', 'heightm'], axis = 1)

test_data_1 = test_data.copy()
test_data_1 = test_data.drop(['lengthm', 'heightm'], axis = 1)
X = train_data_1.drop(['breed_category', 'pet_category'],axis = 1).values
y_1 = train_data_1['pet_category'].values

_cat_indices_ = [0, 1, 5, 6]
#_cat_indices_ = [0, 4, 5]


type_of_target(y_1)
# Catboost for pet_category

kfold, scores = KFold(n_splits = 6, shuffle = True, random_state = 22), list()
for train, test in kfold.split(X):
    X_train, X_test = X[train], X[test]
    y_train, y_test = y_1[train], y_1[test]

    model = CatBoostClassifier(random_state = 22, max_depth = 6, n_estimators = 1000, verbose = 1000, l2_leaf_reg = 1)
    model.fit(X_train, y_train, cat_features = _cat_indices_)
    preds = model.predict(X_test)
    score = f1_score(y_test, preds, average = 'weighted')
    scores.append(score)
    print('Validation f1_score:', score)
print("Average Validation f1_score: ", sum(scores)/len(scores))
y_Preds_1 = model.predict(test_data_1.values)
pet_cat = pd.DataFrame(data = {'pet_id': testIDs, 'pet_category': y_Preds_1.ravel()})
pet_cat.head()
test_data_1 = testData.merge(pet_cat, on = 'pet_id', how = 'left')
test_data_1 = test_data_1.drop(['pet_id', 'heightm', 'lengthm'], axis = 1)
X = train_data_1.drop(['breed_category', 'issue_yr', 'list_yr', 'issue_mth', 'list_mth', 'issue_weekend', 'list_weekend'],axis = 1).values
y_2 = train_data_1['breed_category'].values

_cat_indices_ = [0, 1, 5, 6, 9]
#_cat_indices_ = [0, 4, 5, 8]

type_of_target(y_2)
# Catboost for breed_category

kfold, scores = KFold(n_splits = 6, shuffle = True, random_state = 22), list()
for train, test in kfold.split(X):
    X_train, X_test = X[train], X[test]
    y_train, y_test = y_2[train], y_2[test]

    model_1 = CatBoostClassifier(random_state = 22, max_depth = 8, n_estimators = 300, verbose = 1000, l2_leaf_reg = 3.5)
    model_1.fit(X_train, y_train, cat_features = _cat_indices_)
    preds = model_1.predict(X_test)
    score = f1_score(y_test, preds, average = 'weighted')
    scores.append(score)
    print('Validation f1_score:', score)
print("Average Validation f1_score: ", sum(scores)/len(scores))
y_Preds_2 = model_1.predict(test_data_1.drop(['issue_yr', 'list_yr', 'issue_mth', 'list_mth', 'issue_weekend', 'list_weekend'], axis = 1))

submission = pd.DataFrame(data = {'pet_id': testIDs, 'breed_category': y_Preds_2.ravel(), 'pet_category': y_Preds_1.ravel()})
submission.to_csv('HE_adopt_a_buddy_final_v1.csv', index = False)
submission.head()