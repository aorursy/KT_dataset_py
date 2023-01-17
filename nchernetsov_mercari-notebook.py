import math

import time

import re

from __future__ import print_function

from collections import defaultdict



import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import seaborn as sns



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import make_union, make_pipeline

from sklearn.preprocessing import FunctionTransformer, StandardScaler, LabelEncoder, MinMaxScaler,  Imputer, LabelBinarizer, OneHotEncoder

from sklearn.feature_extraction import DictVectorizer

from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, SGDRegressor, ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import accuracy_score, r2_score

from sklearn.model_selection import RandomizedSearchCV



# Ансамбли



from sklearn.ensemble import RandomForestRegressor, BaggingRegressor

import xgboost as xgb

import lightgbm as lgb



%matplotlib inline

plt.rcParams["figure.figsize"] = (15, 8)

pd.options.display.float_format = '{:.2f}'.format
# Функция для вычисления квадратного корня среднеквадратической ошибки логарифма 

# (Root Mean Squared Logarithmic Error (RMSLE))

def rmsle(y, y_pred):

    y_pred[y_pred < 0.0] = 0.0

    log_sqr = np.square(np.log(np.add(y_pred, 1.0)) - np.log(np.add(y, 1.0)))

    return math.sqrt(np.sum(log_sqr) / y.shape[0])
df = pd.read_csv('../input/train.tsv', sep='\t')

df.info()
df.head(3)
df_test = pd.read_csv('../input/test.tsv', sep='\t')

df_test.info()
df_test.head(3)
def split_category(df):

    df['category_name'] = df['category_name'].astype(str).apply(lambda x: x.split("/"))

    df['main_category'] = df['category_name'].apply(lambda x: x[0])

    df['second_category'] = df['category_name'].apply(lambda x: x[1] if len(x) > 1 else '')

    df['third_category'] = df['category_name'].apply(lambda x: x[2] if len(x) > 2 else '')
split_category(df)

split_category(df_test)
top_25_brands = df.groupby(['brand_name'])['price'].max().sort_values(ascending=False).head(25).reset_index()['brand_name'].as_matrix()
def is_brand_in_top_25(brand_name):

    if brand_name in top_25_brands:

        return 1

    else:

        return 0
min_price = df['price'].min()

max_price  = df['price'].max()

print(min_price)

print(max_price)
n_intervals = 50  # кол-во категорий, на которые будем делить бренды в зависимости от максимальной стоимости



delta_price = (max_price - min_price) / n_intervals



intervals = []  # список интервалов стоимости



interval_left = min_price

for i in range(n_intervals):

    interval_right = interval_left + delta_price

    intervals.append((interval_left, interval_right))

    interval_left = interval_right

    

def find_interval_number(price):

    return int(math.floor((price - min_price) / delta_price))
df_brand_max_price = df.groupby(['brand_name'])['price'].max().reset_index()

df_brand_max_price.head()
df_brand_max_price['brand_category'] = df_brand_max_price['price'].apply(lambda x: find_interval_number(x))

df_brand_max_price.sort_values(by=['brand_name'])

df_brand_max_price.head()
# Общее количество брендов

all_brands_count = df_brand_max_price.count()['price']

all_brands_count
brands_sort_by_max_price = df.groupby(['brand_name'])['price'].max().sort_values(ascending=False).reset_index()

brands_sort_by_max_price.head()
# Список "массивов брендов"

brands_split = []

# Делим все бренды на n_intervals частей по убыванию максимальной цены товара



brands_count_on_interval = (int) (all_brands_count / n_intervals)

last_interval_brands_count = all_brands_count - (n_intervals - 1)*brands_count_on_interval



interval_left = 0

for i in range(n_intervals):

    if i == n_intervals - 1:

        break

    interval_right = interval_left + brands_count_on_interval

    brands_split.append(brands_sort_by_max_price['brand_name'].iloc[interval_left:interval_right].as_matrix())

    interval_left = interval_right

# последние элементы

brands_split.append(brands_sort_by_max_price['brand_name'].iloc[-last_interval_brands_count:].as_matrix())
# Ищем позицию бренда в списке брендов

def get_brand_category(brand_name):

    for i in range(n_intervals):

        brands_i = brands_split[i]

        if brand_name in brands_i:

            return i

        else:

            continue

    return -1
df['brand_category'] = df['brand_name'].astype('str').apply(lambda x: get_brand_category(x))

df_test['brand_category'] = df_test['brand_name'].astype('str').apply(lambda x: get_brand_category(x))
df['top_brand'] = df['brand_name'].astype('str').apply(lambda x: is_brand_in_top_25(x))

df_test['top_brand'] = df_test['brand_name'].astype('str').apply(lambda x: is_brand_in_top_25(x))
class LabelEncoderPipelineFriendly(LabelEncoder):

    

    def fit(self, X, y=None):

        """this would allow us to fit the model based on the X input."""

        super(LabelEncoderPipelineFriendly, self).fit(X)

        

    def transform(self, X, y=None):

        return super(LabelEncoderPipelineFriendly, self).transform(X).reshape(-1, 1)



    def fit_transform(self, X, y=None):

        return super(LabelEncoderPipelineFriendly, self).fit(X).transform(X).reshape(-1, 1)
def get_num_cols(df):

    return df[['item_condition_id', 'shipping']]



def get_main_category(df):

    return df[['main_category']]



def get_second_category(df):

    return df[['second_category']]



def get_third_category(df):

    return df[['third_category']]



def get_top_brand(df):

    return df[['top_brand']]



def get_brand_category(df):

    return df[['brand_category']]



vec = make_union(*[

    make_pipeline(FunctionTransformer(get_num_cols, validate=False), Imputer(strategy='mean'), MinMaxScaler()),

    make_pipeline(FunctionTransformer(get_main_category, validate=False), LabelEncoderPipelineFriendly(),

                  OneHotEncoder(sparse=False)),

    make_pipeline(FunctionTransformer(get_second_category, validate=False), LabelEncoderPipelineFriendly(), MinMaxScaler()),

    make_pipeline(FunctionTransformer(get_third_category, validate=False), LabelEncoderPipelineFriendly(), MinMaxScaler()),

    make_pipeline(FunctionTransformer(get_brand_category, validate=False), MinMaxScaler())

])
x_train = vec.fit_transform(df)

y_train = np.log(df['price'].as_matrix() + 1.0)

print(x_train.shape)

print(y_train.shape)
x_test = vec.fit_transform(df_test)

print('shape of x_test is {}'.format(x_test.shape))
forest = RandomForestRegressor(n_estimators=100, criterion='mse', random_state=1, n_jobs=-1)

forest.fit(x_train, y_train)

y_forest_predict = forest.predict(x_train)

score_forest = rmsle(y_train, y_forest_predict)

print(score_forest)
y_test_log = forest.predict(x_test)

y_test = np.exp(y_test_log) - 1.0

print('shape of y_test is {}'.format(y_test.shape))
df_predicted = pd.DataFrame({'test_id': df_test['test_id'], 'price': y_test})

df_predicted.to_csv('mercari_submission_05.csv', sep=',', index=False)