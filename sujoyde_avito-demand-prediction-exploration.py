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
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import nltk
import gensim
from gensim import *
import re
from nltk.stem import WordNetLemmatizer
import os
wnl = WordNetLemmatizer()
from collections import defaultdict
import operator
import datetime
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from sklearn import model_selection
import xgboost as xgb
train = pd.read_csv('../input/train.csv', parse_dates=["activation_date"])
test = pd.read_csv('../input/test.csv', parse_dates=["activation_date"])
train.shape, test.shape
train.head()
test.head()
train.info()
test.info()
#how many deal probability is over 0.5
train['deal'] = 0
train.loc[train['deal_probability']>0.5, 'deal'] = 1
train['deal'].value_counts(normalize = True)
train['deal_probability'].describe()
#histogram of the deal probability
sns.distplot(train['deal_probability'])
#skewness and kurtosis
print("Skewness: %f" % train['deal_probability'].skew())
print("Kurtosis: %f" % train['deal_probability'].kurt())
# how many unique item id and user ids are there?
train['item_id'].nunique(), train['user_id'].nunique()
# avergae no. of items posted by a user
train['item_id'].nunique()/train['user_id'].nunique()
# categorical columns
cat_cols = train.select_dtypes(include = ['O']).columns.values
cat_cols
# numerical columns
num_cols = train.select_dtypes(exclude = ['O']).columns.values
num_cols
type(train['activation_date'][0])
# range of activation date of train dataset
train['activation_date'].min(), train['activation_date'].max()
# how many days of train dataset
train['activation_date'].max() - train['activation_date'].min()
# range of activation date of test dataset
test['activation_date'].min(), test['activation_date'].max()
# how many days of test dataset
test['activation_date'].max() - test['activation_date'].min()
train['activation_wd'] = train['activation_date'].dt.weekday
test['activation_wd'] = test['activation_date'].dt.weekday                                                             
# lets see which weekday had most ads
grouped = train.groupby(['activation_wd'])['deal_probability'].mean().reset_index()
grouped.columns = ['activation_wd', 'deal_probability']
grouped
# how many regions and cities are there in train dataset
train['region'].nunique(), train['city'].nunique()
# how many regions and cities are there in test dataset
test['region'].nunique(), test['city'].nunique()
from sklearn.preprocessing import LabelEncoder 
lbl = LabelEncoder()
# since all the information regarding region and city is categorical and is in Russian, we will label encode them
cols = ['region', 'city']
for col in cols:
    lbl.fit(list(train[col].values) + list(test[col].values))
    train[col] = lbl.transform(train[col])
    test[col] = lbl.transform(test[col])
#lets see the relationship of a region with the probability
grouped = train.groupby(['region'])['deal_probability'].mean().reset_index()
grouped.columns = ['region', 'region_probability']
plt.rcParams['figure.figsize'] = [15, 4]
plt.xticks(rotation=90)
sns.barplot(x = 'region', y = 'region_probability', data = grouped)
# lets see the relationship of a city with the probability
# Since there are so many levels of a city we wont be able to plot its barplot
# we will have to deal with it in a different way
grouped_1 = train.groupby(['city'])['deal_probability'].mean().reset_index()
grouped_2 = train.groupby(['city']).size().reset_index()
grouped_2.columns = ['city', 'count']
grouped = pd.merge(grouped_1, grouped_2, on = 'city', how = 'inner')
grouped = grouped.sort_values(by = "deal_probability", ascending = False)
grouped.head()
# info about the city feature
grouped.describe()
plt.xlim(0, 500)
sns.regplot(x = 'count', y = 'deal_probability', data = grouped)
# city and date together
grouped_1 = train.groupby(['activation_date', 'city']).size().reset_index()
grouped_1.columns = ['activation_date', 'city', 'city_daily_ads']
grouped_2 = train.groupby(['activation_date', 'city'])['deal_probability'].mean().reset_index()
grouped_2.columns = ['activation_date', 'city', 'deal_probability']
grouped = pd.merge(grouped_1, grouped_2, on = ['activation_date', 'city'], how = 'inner')
grouped = grouped.sort_values(by = "deal_probability", ascending = False)

del grouped_1, grouped_2
grouped.head()
grouped['deal'] = 0
grouped.loc[grouped['deal_probability']>0.5, 'deal'] = 1
grouped.head()
sns.barplot(x = 'deal', y = 'city_daily_ads', data = grouped)
train = pd.merge(train, grouped[['activation_date', 'city', 'city_daily_ads']], on = ['activation_date', 'city'], how = 'inner')
features_to_use = ['city_daily_ads']
grouped = test.groupby(['activation_date', 'city']).size().reset_index()
grouped.columns = ['activation_date', 'city', 'city_daily_ads']
test = pd.merge(test, grouped[['activation_date', 'city', 'city_daily_ads']], on = ['activation_date', 'city'], how = 'inner')
# lets label encode parent category name and category name
cols = ['parent_category_name', 'category_name']
for col in cols:
    lbl.fit(list(train[col].values) + list(test[col].values))
    train[col] = lbl.transform(train[col])
    test[col] = lbl.transform(test[col])
# number of unique values in parent category name and category name
train['parent_category_name'].nunique(), train['category_name'].nunique()
sns.barplot(x = 'parent_category_name', y = 'deal_probability', data = train)
# lets see the distribution of each parent category name
sns.countplot(x = 'parent_category_name', data = train)
train['parent_category_4'] = 0
train.loc[train['parent_category_name']==4, 'parent_category_4'] = 1
plt.rcParams['figure.figsize'] = [12, 4]
sns.barplot(x = 'parent_category_4', y = 'deal_probability', data = train)
test['parent_category_4'] = 0
test.loc[train['parent_category_name']==4, 'parent_category_4'] = 1
features_to_use.append('parent_category_4')
param = {}
param['objective'] = 'reg:logistic'
param['eta'] = 0.1
param['max_depth'] = 7
param['silent'] = 0
param['eval_metric'] = "rmse"
param['min_child_weight'] = 6
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['seed'] = 0
num_rounds = 500
train_X = np.array(train[features_to_use])
train_y = np.array(train['deal_probability'])

X_tr, X_va, y_tr, y_va = model_selection.train_test_split(train_X, train_y, test_size=0.2, random_state=2018)

tr_data = xgb.DMatrix(X_tr, y_tr)
va_data = xgb.DMatrix(X_va, y_va)

watchlist = [(tr_data, 'train'), (va_data, 'valid')]

model = xgb.train(param, tr_data, 1000, watchlist, maximize=False, early_stopping_rounds = 25, verbose_eval=25)

train['title'].head()
def get_number_of_chars(text):
    return (len(text))
get_number_of_chars(train['title'][0])
train['title_chars_count'] = train['title'].apply(lambda x: get_number_of_chars(x))
test['title_chars_count'] = test['title'].apply(lambda x: get_number_of_chars(x))
train['title_chars_count'].min(), train['title_chars_count'].max()
plt.rcParams['figure.figsize'] = [22, 4]
sns.barplot(x = 'title_chars_count', y = 'deal_probability', data = train)
sns.countplot(x = 'title_chars_count', data = train)
train['title_chars_count_bucket'] = train['title_chars_count'].apply(lambda x:np.floor(x/10)+1)
test['title_chars_count_bucket'] = test['title_chars_count'].apply(lambda x:np.floor(x/10)+1)
sns.barplot(x = 'title_chars_count_bucket', y = 'deal_probability', data = train)
features_to_use.append('title_chars_count')
train_X = np.array(train[features_to_use])
train_y = np.array(train['deal_probability'])

X_tr, X_va, y_tr, y_va = model_selection.train_test_split(train_X, train_y, test_size=0.2, random_state=2018)

tr_data = xgb.DMatrix(X_tr, y_tr)
va_data = xgb.DMatrix(X_va, y_va)

watchlist = [(tr_data, 'train'), (va_data, 'valid')]

model = xgb.train(param, tr_data, 1000, watchlist, maximize=False, early_stopping_rounds = 25, verbose_eval=25)

train['description'].head()
def get_number_of_words(text):
    return len(str(text).split())

train['description_words_count'] = train['description'].apply(lambda x: get_number_of_words(x))
test['description_words_count'] = test['description'].apply(lambda x: get_number_of_words(x))
train['description_words_count'].min(), train['description_words_count'].max()
train['description_words_count_bucket'] = train['description_words_count'].apply(lambda x:np.floor(x/100)+1)
test['description_words_count_bucket'] = test['description_words_count'].apply(lambda x:np.floor(x/100)+1)
sns.barplot(x = 'description_words_count_bucket', y = 'deal_probability', data = train)
sns.countplot(train['description_words_count_bucket'])
features_to_use.append('description_words_count')
len(features_to_use)
features_to_use
train_X = np.array(train[features_to_use])
train_y = np.array(train['deal_probability'])

X_tr, X_va, y_tr, y_va = model_selection.train_test_split(train_X, train_y, test_size=0.2, random_state=2018)

tr_data = xgb.DMatrix(X_tr, y_tr)
va_data = xgb.DMatrix(X_va, y_va)

watchlist = [(tr_data, 'train'), (va_data, 'valid')]

model = xgb.train(param, tr_data, 1000, watchlist, maximize=False, early_stopping_rounds = 25, verbose_eval=25)

train['price_present'] = 0
train.loc[train['price'].isnull(), 'price_present'] = 1
test['price_present'] = 0
test.loc[train['price'].isnull(), 'price_present'] = 1
sns.barplot(x = 'price_present', y = 'deal_probability', data = train)
# histogram of price
sns.distplot(train[train['price'].notnull()]['price'])
train['price'].describe()
# lets break the price feature into buckets in steps of 25%

train['price_bucket'] = 0
train.loc[(train['price']>=500)&(train['price']<1300), 'price_bucket'] = 1
train.loc[(train['price']>=1300)&(train['price']<7000), 'price_bucket'] = 2
train.loc[train['price']>=7000, 'price_bucket'] = 3
train.loc[train['price'].isnull(), 'price_bucket'] = 4

test['price_bucket'] = 0
test.loc[(test['price']>=500)&(test['price']<1300), 'price_bucket'] = 1
test.loc[(test['price']>=1300)&(test['price']<7000), 'price_bucket'] = 2
test.loc[test['price']>=7000, 'price_bucket'] = 3
test.loc[test['price'].isnull(), 'price_bucket'] = 4
sns.barplot(x = 'price_bucket', y = 'deal_probability', data = train)
features_to_use.append('price')
features_to_use
train.loc[train['price'].isnull(), 'price'] = -999
test.loc[test['price'].isnull(), 'price'] = -999
train_X = np.array(train[features_to_use])
train_y = np.array(train['deal_probability'])

X_tr, X_va, y_tr, y_va = model_selection.train_test_split(train_X, train_y, test_size=0.2, random_state=2018)

tr_data = xgb.DMatrix(X_tr, y_tr)
va_data = xgb.DMatrix(X_va, y_va)

watchlist = [(tr_data, 'train'), (va_data, 'valid')]

model = xgb.train(param, tr_data, 1000, watchlist, maximize=False, early_stopping_rounds = 25, verbose_eval=25)

train['item_seq_number'].min(), train['item_seq_number'].max()
train['item_seq_number'].describe()
sns.barplot(x = 'deal', y = 'item_seq_number', data = train)
features_to_use.append('item_seq_number')
# histogram of item seq number
sns.distplot(train['item_seq_number'])
features_to_use
train_X = np.array(train[features_to_use])
train_y = np.array(train['deal_probability'])

X_tr, X_va, y_tr, y_va = model_selection.train_test_split(train_X, train_y, test_size=0.2, random_state=2018)

tr_data = xgb.DMatrix(X_tr, y_tr)
va_data = xgb.DMatrix(X_va, y_va)

watchlist = [(tr_data, 'train'), (va_data, 'valid')]

model = xgb.train(param, tr_data, 1000, watchlist, maximize=False, early_stopping_rounds = 25, verbose_eval=25)
test_X = np.array(test[features_to_use])
X_te = xgb.DMatrix(test_X)
y_pred = model.predict(X_te)
sub = pd.read_csv('../input/sample_submission.csv')
sub['deal_probability'] = y_pred
sub['deal_probability'].clip(0.0, 1.0, inplace=True)
sub.to_csv('xgb_with_basic_features.csv', index=False)
sub.head()
