import numpy as np

import pandas as pd

import csv

from sklearn.model_selection import train_test_split

import lightgbm as lgb

import gc

import datetime

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import LabelBinarizer

from time import time

import scipy

from xgboost import XGBRegressor

import xgboost

from xgboost import plot_importance

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error
NUM_BRANDS = 500

NAME_MIN_DF = 10

MAX_FEAT_DESCP = 50000

initial = time()
train = pd.read_csv('../input/train.tsv', sep = '\t')

test = pd.read_csv('../input/test.tsv', sep = '\t')
train.head()

train_id = train['train_id']

test_id = test['test_id']
train.drop('train_id',axis = 1, inplace = True)

test.drop('test_id',axis=1,  inplace = True)
train_num = len(train)

y_train = np.log1p(train['price'])

all_data = pd.concat([train, test], axis = 0)

all_data.drop('price', axis = 1,inplace = True)

all_data.head()
start = time()

all_data['cat1'] = all_data['category_name'].str.extract('([A-Z]\w{0,})',expand = True)

all_data['cat2'] = all_data['category_name'].str.extract('/(.*)/', expand = True)

all_data['cat3'] = all_data['category_name'].str.extract('/.+/(.*)', expand = True)

all_data.head()

end = time()

print (end - start )
all_data['category_name'].fillna('Other', inplace = True)

all_data['cat1'].fillna('Other', inplace = True)

all_data['cat2'].fillna('Other', inplace = True)

all_data['cat3'].fillna('Other', inplace = True)
all_data['brand_name'].fillna('unknown', inplace = True)

pop_brands = train['brand_name'].value_counts().index[:NUM_BRANDS]

all_data.loc[~all_data['brand_name'].isin(pop_brands), 'brand_name'] = 'Other'

all_data['item_description'].fillna('None', inplace = True)

all_data['item_condition_id'] = all_data['item_condition_id'].astype('category')

all_data['brand_name'] = all_data['brand_name'].astype('category')
#encoding of the name

start = time()

count = CountVectorizer(min_df = NAME_MIN_DF)

X_name = count.fit_transform(all_data['name'])

end = time()

print (end - start)
#category encoders

start = time()

count_cat1 = CountVectorizer()

X_cat1 = count_cat1.fit_transform(all_data['cat1'])

count_cat2 = CountVectorizer()

X_cat2 = count_cat2.fit_transform(all_data['cat2'])

count_cat3 = CountVectorizer()

X_cat3 = count_cat3.fit_transform(all_data['cat3'])

end = time()

print (end - start)
#description encoder

start = time()

count_descp = TfidfVectorizer(max_features = MAX_FEAT_DESCP, ngram_range = (1,3),

                             stop_words = 'english')

X_descp = count_descp.fit_transform(all_data['item_description'])

end = time()

print (end - start)
#brand encoders

start = time()

vect_brand = LabelBinarizer(sparse_output = True)

X_brand = vect_brand.fit_transform(all_data['brand_name'])

end = time()

print (end - start)
#row compressor, dummy encoders

start = time()

data = pd.get_dummies(all_data[['item_condition_id', 'shipping']], sparse = True).values

X_dummies = scipy.sparse.csr_matrix(data)

end = time()

print (end - start)
X = scipy.sparse.hstack((X_brand, X_cat1, X_cat2, X_cat3, X_descp, X_dummies, X_name)).tocsr()

print (X_dummies.shape, X_cat1.shape, X_cat2.shape, X_cat3.shape, X_descp.shape,X_name.shape)
X_train = X[:train_num]

X_test = X[train_num:]
#train_X, valid_X, train_y, valid_y = train_test_split(X_train, y_train, test_size = 0.99, random_state = 42)
OPTIMIZE_ROUNDS = False

EARLY_STOPPING_ROUNDS = 10

LEARNING_RATE = 0.1

MAX_ROUNDS = 100

model_xgb = XGBRegressor(n_estimators = MAX_ROUNDS,

                    max_depth = 16, 

                    learning_rate = LEARNING_RATE,

                    subsample = 0.88,

                    gamma = 10,

                    reg_alpha = 8,

                    reg_lambda=1.3,

                    min_child_weight = 20, 

                    colsample_bytree = 0.45,

                    silent = 1)
K = 2

kf = KFold(n_splits = K, shuffle = True, random_state = 42)
start = time()

y_test_pred = 0

total_valid_score = 0

for i, (train_index, valid_index) in enumerate(kf.split(X_train)):

    train_y, valid_y = y_train[train_index], y_train[valid_index]

    train_X, valid_X = X_train[train_index], X_train[valid_index]

    X_test_copy = X_test.copy()

    if OPTIMIZE_ROUNDS:

        eval_set = [(valid_X, valid_y)]

        fit_model = model_xgb.fit(train_X, train_y, eval_set = eval_set, eval_metric = 'rmse',

                             early_stopping_rounds = EARLY_STOPPING_ROUNDS, verbose = False)

        print ('Best Ntrees = ', fit_model.best_ntree_limit)

        print ('Best RMSE = ', model_xgb.best_score)

    else: 

        fit_model = model_xgb.fit(train_X, train_y)

    pred = fit_model.predict(valid_X)

    valid_score = np.sqrt(mean_squared_error(pred,valid_y))

    print ('{} fold'.format(i))

    print ('score = ',valid_score )

    total_valid_score += valid_score

    #y_valid_pred[valid_index] = pred

    y_test_pred += fit_model.predict(X_test)

y_test_pred /= K

print ('average valid dataset score = ', total_valid_score/K )

#print ('rmsle for the full training set = ', rmsle(np.expm1(y_valid_pred), np.expm1(train_y)))

end = time()

print (end - start)
sub = pd.DataFrame()

sub['test_id'] = test_id

sub['price'] = np.expm1(y_test_pred)

print (sub.head())

sub.to_csv('submission_xgb_cv.csv.csv', index = False)

final = time()

print ('the total running time is {}'.format(final-initial))