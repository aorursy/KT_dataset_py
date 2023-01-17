import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', None)
import os

# Reaading directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Importing useful libraries

import math
import gc
import warnings
warnings.filterwarnings("ignore")
import itertools
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import statsmodels.api as sm
from pylab import rcParams
# rcParams['figure.figsize'] = 18, 6
from tqdm import tqdm_notebook as tqdm
import datetime
from datetime import datetime
import json

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from fbprophet import Prophet

import matplotlib
matplotlib.rcParams['axes.labelsize'] = 24
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 

from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score, RepeatedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures

import xgboost as xgb
import lightgbm as lgb
import sklearn.ensemble as ensemble
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge, LogisticRegressionCV, RidgeCV, LassoCV, ElasticNetCV, OrthogonalMatchingPursuit, ElasticNet, LassoLarsCV, BayesianRidge
from scipy import stats
# Read public holidays for India (such as Republic day, Independence day, New year eve etc.)

holidays = pd.read_csv('/kaggle/input/gift-cart-sales-forecast/holidays.csv')
holidays['holidays'] = holidays['holidays'].apply(lambda x: datetime.strptime(x, '%d-%b-%y'))
holidays['year'] = holidays['holidays'].dt.year
holidays['month'] = holidays['holidays'].dt.month
holidays['day'] = holidays['holidays'].dt.day
holidays['is_public_holiday'] = 1
holidays['is_public_holiday'] = holidays['is_public_holiday'].astype(np.int8)
holidays = holidays.drop('holidays', axis = 1)
holidays.head()
# read city mapping

# Opening JSON file 
with open('/kaggle/input/gift-cart-sales-forecast/sales_data/Dataset/city_dict.json') as cities:
    city_map = json.load(cities)
    city_map = dict([(value, key) for key, value in city_map.items()])
for key in city_map:
    city_map[key] = int(city_map[key])
city_map
# read expected discount file
exp_discount = pd.read_csv('/kaggle/input/gift-cart-sales-forecast/sales_data/Dataset/expected_discount.csv', low_memory = False, dtype = {'product': np.int16})
exp_discount['date'] = pd.to_datetime(exp_discount['date'])

# format column names
exp_discount.columns = exp_discount.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('Discount_', '')

# pivot down columns to rows
exp_discount = exp_discount.melt(id_vars = ["date", "product"], var_name = "city", value_name = "exp_discount")

# Add city-number mapping
exp_discount = exp_discount.replace({"city": city_map})

# Creating additional calendar features

exp_discount['year'] = exp_discount['date'].dt.year
exp_discount['month'] = exp_discount['date'].dt.month
exp_discount['day'] = exp_discount['date'].dt.day
exp_discount = exp_discount.drop('date', axis = 1)

################################

# read historical discount file

his_discount = pd.read_csv('/kaggle/input/gift-cart-sales-forecast/sales_data/Dataset/historical_discount.csv', low_memory = False, dtype = {'product': np.int16})
his_discount['date'] = pd.to_datetime(his_discount['date'])

# format column names
his_discount.columns = his_discount.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('Discount_', '')

# pivot down columns to rows
his_discount = his_discount.melt(id_vars = ["date", "product"], var_name = "city", value_name = "his_discount")

# Add city-number mapping
his_discount = his_discount.replace({"city": city_map})

# Creating additional calendar features

his_discount['year'] = his_discount['date'].dt.year
his_discount['month'] = his_discount['date'].dt.month
his_discount['day'] = his_discount['date'].dt.day
his_discount = his_discount.drop('date', axis = 1)

his_discount.head()
his_discount.dtypes
# Parsing dates for foot-fall data

def parser(x):
    if '-' in x:
        x = x.replace('-', '/')
    return datetime.strptime(x, '%m/%d/%Y')
# read foot fall info

foot_fall = pd.read_csv('/kaggle/input/gift-cart-sales-forecast/sales_data/Dataset/foot_fall.csv', low_memory = False)

# pivot down columns to rows, apply date formats, Add city-number mapping

foot_fall = foot_fall.melt(id_vars = ["city"], var_name = "date", value_name = "footfall")
foot_fall['date'] = foot_fall['date'].astype(str).apply(lambda x: parser(x))
foot_fall = foot_fall.replace({"city": city_map})

# Creating additional calendar features

foot_fall['year'] = foot_fall['date'].dt.year
foot_fall['month'] = foot_fall['date'].dt.month
foot_fall['day'] = foot_fall['date'].dt.day

foot_fall['city'] = foot_fall['city'].astype(np.int8)
foot_fall = foot_fall.drop('date', axis = 1)

# Mean footfalls in a city

# Fill NaNs in footfall with avg. footfall on Jan 2, 2018

mean_ff = pd.DataFrame({'footfall' : foot_fall.groupby(['city'])['footfall'].apply(lambda x: np.round(x.mean()))}).reset_index()
mean_ff.columns = ['city', 'footfall']
mean_ff = mean_ff.set_index('city')['footfall']
foot_fall['footfall'] = foot_fall['footfall'].fillna(foot_fall['city'].map(mean_ff))

foot_fall.head()
# read product information

prod_info = pd.read_csv('/kaggle/input/gift-cart-sales-forecast/sales_data/Dataset/product_information.csv', low_memory = False)
prod_info['product'] = prod_info['product'].astype(np.int16)
prod_info['product_category'] = prod_info['product_category'].astype(str).apply(lambda x: x.split('_')[1])
prod_info['product_subcategory'] = prod_info['product_subcategory'].astype(str).apply(lambda x: x.split('_')[1])

for column in ['product_category', 'product_subcategory']:
    prod_info[column] = prod_info[column].astype(np.int16)

for i in range(1, 11):
    column = 'var_' + str(i)
    prod_info[column] = prod_info[column].apply(lambda x: round(x, 3))
prod_info.head()
print('Unique values in each columns:\n\n', prod_info.nunique())

# Correlation matrix

sns.heatmap(prod_info.corr())
plt.show()
# As var_4 and var_7 are constant with no predictive power, dropping the same

prod_info = prod_info.drop(['var_4', 'var_7'], axis = 1)
# read sales data

sales_15 = pd.read_csv('/kaggle/input/gift-cart-sales-forecast/sales_data/Dataset/2015_sales_data.csv', low_memory = False)
sales_16 = pd.read_csv('/kaggle/input/gift-cart-sales-forecast/sales_data/Dataset/2016_sales_data.csv', low_memory = False)
sales_17 = pd.read_csv('/kaggle/input/gift-cart-sales-forecast/sales_data/Dataset/2017_sales_data.csv', low_memory = False)
sales_18 = pd.read_csv('/kaggle/input/gift-cart-sales-forecast/sales_data/Dataset/2018_sales_data.csv', low_memory = False)
#, parse_dates = [0], date_parser = dateparser)

# append all sales data into one dataframe

train_data = sales_15.copy()
train_data = train_data.append(sales_16)
train_data = train_data.append(sales_17)
train_data = train_data.append(sales_18)

# Collecting Memory dump

del sales_15, sales_16, sales_17, sales_18
gc.collect()

# Creating additional calendar features

train_data['date'] = pd.to_datetime(train_data['date'])
train_data['quarter'] = train_data['date'].dt.quarter
train_data['year'] = train_data['date'].dt.year
train_data['month'] = train_data['date'].dt.month
train_data['day'] = train_data['date'].dt.day
train_data['week_day'] = train_data['date'].dt.dayofweek
train_data['is_weekend'] = np.where(train_data['date'].isin([5, 6]), 1, 0)
train_data['is_weekday'] = np.where(train_data['date'].isin([0, 1, 2, 3, 4]), 1, 0)


cols = ['city', 'product']
train_data['city_prod'] = train_data[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis = 1)

train_data.drop('date', axis = 1)

for column in ['city', 'quarter', 'month', 'day', 'week_day', 'is_weekend', 'is_weekday']:
    train_data[column] = train_data[column].astype(np.int8)

for column in ['year', 'product']:
    train_data[column] = train_data[column].astype(np.int16)

# Replacing negative sales with 0 (~1550 records)
train_data.loc[(train_data['sales'] < 0), 'sales'] = 0

print('Sales data size: ', train_data.shape)
train_data.head()
train_data.dtypes
# Merging all created datasets to train_data

train_data = train_data.merge(his_discount, on = ['year', 'month', 'day', 'product', 'city'], how = 'left')
train_data = train_data.merge(foot_fall, on = ['year', 'month', 'day', 'city'], how = 'left')
train_data = train_data.merge(prod_info, on = ['product'], how = 'left')
train_data = train_data.merge(holidays, on = ['year', 'month', 'day'], how = 'left')
train_data['is_public_holiday'] = train_data['is_public_holiday'].fillna(0)
train_data['is_public_holiday'] = train_data['is_public_holiday'].astype(np.int8)
train_data.head()
gc.collect()
train_data.isnull().sum()
# Fill NaNs in his_discount with 0
train_data.loc[(train_data['his_discount'].isnull()), 'his_discount'] = 0

# Dropping date column
train_data = train_data.drop(['date'], axis = 1)

# For Jan 3,4 2018 the footfall data doesn't exist - filling with avg. footfalls in these days
train_data['footfall'] = train_data['footfall'].fillna(train_data['city'].map(mean_ff))
test_data = pd.read_csv('/kaggle/input/gift-cart-sales-forecast/sales_data/Dataset/test_data.csv', low_memory = False)
test_data['date'] = pd.to_datetime(test_data['date'])
gc.collect()
train_prod = train_data[['product']].drop_duplicates(subset = None, keep = 'first', inplace = False)
print('Unique products in train data: ', len(train_prod))
test_prod = test_data[['product']].drop_duplicates(subset = None, keep = 'first', inplace = False)
print('Unique products in test data: ', len(test_prod))

train_prod_city = train_data[['product', 'city']].drop_duplicates(subset = None, keep = 'first', inplace = False)
#test_prod_city.to_csv('product_city_comb_train.csv', index = False)
print('Unique city-product pairs in train data: ', len(train_prod_city))

test_prod_city = test_data[['product', 'city']].drop_duplicates(subset = None, keep = 'first', inplace = False)
#product_city.to_csv('product_city_comb_test.csv', index = False)
print('Unique city-product pairs in test data: ', len(test_prod_city))

# City product pairs in test data NOT in train data

city_product_train = list((train_prod_city['city'].astype(str) + "_" + train_prod_city['product'].astype(str)).unique())
city_product_test = list((test_prod_city['city'].astype(str) + "_" + test_prod_city['product'].astype(str)).unique())
new_city_prod = [item for item in city_product_test if item not in city_product_train]
new_city_prod = pd.DataFrame({'city_prod': new_city_prod})
new_city_prod[['city','product']] = new_city_prod.city_prod.str.split("_", expand = True,)
new_city_prod = new_city_prod.drop('city_prod', axis = 1)
print('There are ', len(new_city_prod), ' unique combinations of city-product in test dataset not present in training dataset')
# As there are 2 crore+ records in training data, we can remove all products' data not present in test set

test_unique_prods = test_data['product'].unique().tolist()
train_unique_prods = train_data['product'].unique().tolist()

# There are new products in test dataset which are not a part of train dataset

new_prod = [item for item in test_unique_prods if item not in train_unique_prods]
new_prod = pd.DataFrame({'prod': new_prod})
print('There are ', len(new_prod), ' unique products left in test dataset not present in train dataset!')
# City products combination not present in test data and that can be removed from the training data

dump_city_prod = [item for item in city_product_train if item not in city_product_test]
dump_city_prod = pd.DataFrame({'city_prod': dump_city_prod})
dump_city_prod['remove'] = 'Y'
#dump_city_prod[['dump_city','dump_product']] = dump_city_prod.city_prod.str.split("_", expand = True,)
#dump_city_prod = dump_city_prod.drop('city_prod', axis = 1)
print('There are ', len(dump_city_prod), ' unique combinations of city-product in train dataset not present in test dataset that can be dropped!')

print('Training data size before removing irrelevant product records (not a part of test dataset): ', train_data.shape)
train_data = train_data[train_data['product'].isin(test_unique_prods)]
print('Training data size after removing irrelevant product records (not a part of test dataset): ', train_data.shape)

gc.collect()
train_data = train_data.merge(dump_city_prod, on = ['city_prod'], how = 'left')
train_data = train_data.loc[~(train_data['remove'] == 'Y')]
train_data = train_data.drop(['remove', 'city_prod'], axis = 1)
print('Training data size after removing irrelevant city-product combination records (not a part of test dataset): ', train_data.shape)
# train_data.to_csv('train.csv', index = False)
train_data.head()
# Categorical features
cat_feats = ['city', 'product','product_category', 'product_subcategory', 'is_public_holiday']

# Excluding target column - sales
useless_cols = ['sales']
train_cols = train_data.columns[~train_data.columns.isin(useless_cols)]

X_train = train_data[train_cols]
y_train = train_data["sales"]
%%time

np.random.seed(22)

# Generating random training and validation indices

fake_valid_inds = np.random.choice(X_train.index.values, 2000000, replace = False)
train_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)
trainData = lgb.Dataset(X_train.loc[train_inds] , label = y_train.loc[train_inds], categorical_feature = cat_feats, free_raw_data = False)
fake_valid_data = lgb.Dataset(X_train.loc[fake_valid_inds], label = y_train.loc[fake_valid_inds], categorical_feature = cat_feats, free_raw_data = False)
# Garbage collection

del train_data, X_train, y_train, fake_valid_inds, train_inds
gc.collect()
# Setting parameters for LightGBM model

params = {
        'objective' : 'poisson', 'metric' :'rmse', 'learning_rate' : 0.075,
#         'sub_feature' : 0.8,
        'sub_row' : 0.75, 'bagging_freq' : 2, 'lambda_l2' : 0.1,
#         'nthread' : 4
        'metric': ['rmse'], 'verbosity': 1, 'num_iterations' : 1200, 'num_leaves': 128, 'min_data_in_leaf': 100,
        }
m_lgb = lgb.train(params, trainData, valid_sets = [fake_valid_data], verbose_eval = 20)
m_lgb.save_model('model_2.lgb')
foot_fall_pred = pd.DataFrame(pd.date_range(start='5/1/2018', periods = 92))
foot_fall_pred.columns = ['date']
foot_fall_pred['year'] = foot_fall_pred['date'].dt.year
foot_fall_pred['month'] = foot_fall_pred['date'].dt.month
foot_fall_pred['day'] = foot_fall_pred['date'].dt.day
foot_fall_pred = foot_fall_pred.drop('date', axis = 1)
foot_fall_pred['key'] = 0

Cities = pd.DataFrame([i for i in range(1, 11)])
Cities.columns = ['city']
Cities['key'] = 0

foot_fall_pred = pd.merge(foot_fall_pred, Cities, on = 'key')
foot_fall_pred = foot_fall_pred.drop('key', axis = 1)
from numpy import mean
from numpy import std
from math import sqrt

X = foot_fall[['year', 'month', 'day', 'city']].values
y = foot_fall['footfall'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 22)

# Predict the footfall for future months

gb = GradientBoostingRegressor(random_state = 22)

# Random grid for best parameters evaluation

random_grid = {'max_depth': [2, 4, 6, 8, 10, 12],
               'max_features': ['auto', 'sqrt'], 
               'min_samples_leaf': [1, 2, 4, 6, 8, 10], 
               'min_samples_split': [2, 4, 6, 8, 10, 12], 
               'n_estimators': [100, 200, 400]
              }

# RandomizedSearchCV for best parameters (folds = 5)

grid_gb = RandomizedSearchCV(estimator = gb, param_distributions = random_grid, cv = 5, n_iter = 100,
                             verbose = 1, n_jobs = -1, scoring = "neg_mean_squared_error").fit(X_train, y_train)
print("Best parameter: {}".format(grid_gb.best_params_))
print("Best score: {:.2f}".format(100*(max(0, 1 - (-1*grid_gb.best_score_)**0.5))))
"""
Best parameter: {'n_estimators': 400, 'min_samples_split': 6, 'min_samples_leaf': 10, 'max_features': 'auto', 'max_depth': 10}
Best score: 81.20
"""
GBR = GradientBoostingRegressor(random_state = 22, n_estimators = 400, min_samples_split = 6, 
                                min_samples_leaf = 10, max_features = 'auto', max_depth = 10)

GBR.fit(X_train, y_train)
y_pred = GBR.predict(X_test)
print('Gradient Boost regressor validation score: ', 100*(max(0, 1 - sqrt(mean_squared_error(y_test, y_pred)))))

foot_fall_eval = foot_fall_pred[['year', 'month', 'day', 'city']].values

# make predictions

Y_Pred = GBR.predict(foot_fall_eval)
foot_fall_pred['footfall'] = pd.Series(Y_Pred, index = foot_fall_pred.index)
test_data['quarter'] = test_data['date'].dt.quarter
test_data['year'] = test_data['date'].dt.year
test_data['month'] = test_data['date'].dt.month
test_data['day'] = test_data['date'].dt.day
test_data['week_day'] = test_data['date'].dt.dayofweek
test_data['is_weekend'] = np.where(test_data['date'].isin([5, 6]), 1, 0)
test_data['is_weekday'] = np.where(test_data['date'].isin([0, 1, 2, 3, 4]), 1, 0)

test_data = test_data.drop('date', axis = 1)

for column in ['city', 'quarter', 'month', 'day', 'week_day', 'is_weekend', 'is_weekday']:
    test_data[column] = test_data[column].astype(np.int8)

for column in ['year', 'product']:
    test_data[column] = test_data[column].astype(np.int16)

test_data = test_data.merge(exp_discount, on = ['year', 'month', 'day', 'product', 'city'], how = 'left')
test_data = test_data.merge(foot_fall_pred, on = ['year', 'month', 'day', 'city'], how = 'left')
test_data = test_data.merge(prod_info, on = ['product'], how = 'left')

# Add public holidays if any

test_data = test_data.merge(holidays, on = ['year', 'month', 'day'], how = 'left')
test_data['is_public_holiday'] = test_data['is_public_holiday'].fillna(0)
test_data['is_public_holiday'] = test_data['is_public_holiday'].astype(np.int8)

print('Test data size: ', test_data.shape)
test_data.head()
test_data.head()
# Fill NaNs in his_discount with 0

test_data.loc[(test_data['exp_discount'].isnull()), 'exp_discount'] = 0
test_data.isnull().sum()

# Separating the unseen data present in test dataset

test_data_unseen = test_data[(test_data['product_category'].isnull())]
test_data = test_data[~(test_data['product_category'].isnull())]
# Downcast datatypes to conserve memory

for column in ['product_category', 'product_subcategory']:
    test_data[column] = test_data[column].astype(np.int16)
# Preserve the test-IDs for later mapping the sales

predictionIDs = test_data['id']
test_set = test_data.drop('id', axis = 1).values

# Predict sales using the same model
predictions = m_lgb.predict(test_set)
print('Unseen dataset shape: ', test_data_unseen.shape, 'Remaining test dataset shape: ', test_data.shape)
submission_df_1 = pd.DataFrame({
                                "id": predictionIDs.values,
                                 "city": test_data['city'].values,
                                "sales": pd.Series(predictions)})
# submission_df.to_csv('submission_1.csv', index = False)
submission_df_1.head()

unseenData = test_data_unseen[['id', 'city', 'product']]

print('Unseen Product IDs: ', unseenData['product'].nunique())

# For products with no data available, replacing sales with average sales for respective cities on a day

NewProds = unseenData['product'].unique().tolist()
city_avg_sales = pd.DataFrame(submission_df_1.groupby('city')['sales'].mean()).reset_index()
city_avg_sales.columns = ['city', 'sales']
unseenData = unseenData.merge(city_avg_sales, on = 'city', how = 'left')
unseenData = unseenData.drop(['city', 'product'], axis = 1)

submission_df = submission_df_1[['id', 'sales']].append(unseenData)
submission_df = submission_df.sort_values('id')
submission_df['sales'] = submission_df['sales'].apply(lambda x: int(np.round(x)))
submission_df.to_csv('submission_4.csv', index = False)
print('Total sales volume predicted for ', len(submission_df), 'records')
submission_df.head()
gc.collect()