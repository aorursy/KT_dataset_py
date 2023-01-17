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



from sklearn.tree import DecisionTreeRegressor

from scipy import stats

import sklearn as sk

import itertools

import matplotlib.pyplot as plt

import seaborn as sns

import warnings 

warnings.filterwarnings('ignore')

%matplotlib inline

sns.set(style='white', context='notebook', palette='deep') 

import matplotlib.style as style

style.use('fivethirtyeight')



from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split 

from sklearn.model_selection import GridSearchCV



# Data Scaler

from sklearn.preprocessing import StandardScaler



# Regression

from sklearn.linear_model import LinearRegression 

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import ElasticNet



# Metrics

from sklearn import metrics

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score
# Reading dataset and seting on variables

example = pd.read_csv('../input/exemplo.csv')

test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')

valid = pd.read_csv('../input/valid.csv')
# Transforming neighborhoods on numbers position

test['borough'][test['borough'] == 1] = 'Manhattan'

test['borough'][test['borough'] == 2] = 'Bronx'

test['borough'][test['borough'] == 3] = 'Brooklyn'

test['borough'][test['borough'] == 4] = 'Queens'

test['borough'][test['borough'] == 5] = 'Staten Island'



train['borough'][train['borough'] == 1] = 'Manhattan'

train['borough'][train['borough'] == 2] = 'Bronx'

train['borough'][train['borough'] == 3] = 'Brooklyn'

train['borough'][train['borough'] == 4] = 'Queens'

train['borough'][train['borough'] == 5] = 'Staten Island'



valid['borough'][valid['borough'] == 1] = 'Manhattan'

valid['borough'][valid['borough'] == 2] = 'Bronx'

valid['borough'][valid['borough'] == 3] = 'Brooklyn'

valid['borough'][valid['borough'] == 4] = 'Queens'

valid['borough'][valid['borough'] == 5] = 'Staten Island'
# Removing empty columns

del test['ease-ment']

del train['ease-ment']

del valid['ease-ment']
# Testing existence of duplicated values

sum(test.duplicated(test.columns))

sum(train.duplicated(train.columns))

sum(valid.duplicated(valid.columns))
# Removing data i'm not going to use

test = test.drop(['sale_date', 'neighborhood', 'building_class_category', 'building_class_category', 

                  'tax_class_at_present', 'block', 'lot', 'building_class_at_present','address', 

                  'apartment_number', 'total_units','tax_class_at_time_of_sale', 

                  'building_class_at_time_of_sale'],axis=1)

train = train.drop(['sale_date', 'neighborhood', 'building_class_category', 'building_class_category',

                    'tax_class_at_present', 'block', 'lot', 'building_class_at_present', 

                    'address', 'apartment_number', 'total_units','tax_class_at_time_of_sale', 

                    'building_class_at_time_of_sale'],axis=1)

valid = valid.drop(['sale_date', 'neighborhood', 'building_class_category', 'building_class_category', 

                    'tax_class_at_present', 'block', 'lot', 'building_class_at_present', 

                    'address', 'apartment_number', 'total_units', 'tax_class_at_time_of_sale', 

                    'building_class_at_time_of_sale'],axis=1)
test.dtypes
train.dtypes
# Cheking data types

valid.dtypes
# Transforming land square and gross square into numbers

test['land_square_feet'] = pd.to_numeric(test['land_square_feet'], errors='coerce')

test['gross_square_feet'] = pd.to_numeric(test['gross_square_feet'], errors='coerce')

train['land_square_feet'] = pd.to_numeric(train['land_square_feet'], errors='coerce')

train['gross_square_feet'] = pd.to_numeric(train['gross_square_feet'], errors='coerce')

valid['land_square_feet'] = pd.to_numeric(valid['land_square_feet'], errors='coerce')

valid['gross_square_feet'] = pd.to_numeric(valid['gross_square_feet'], errors='coerce')
# Cheking data types

test.dtypes

train.dtypes

valid.dtypes
# Cheking number of columns and rows on data

test.shape

train.shape

valid.shape
# Checking my data, saw that Train is the only .csv who contains the variable sale_price

train.describe()
#Transforming NaN values on most seeing numbers 

test['land_square_feet'].fillna(test['land_square_feet'].mean(), inplace=True)

test['gross_square_feet'].fillna(test['gross_square_feet'].mean(), inplace=True)

test['year_built'].fillna(test['year_built'].mean, inplace=True)



train['land_square_feet'].fillna(train['land_square_feet'].mean(), inplace=True)

train['gross_square_feet'].fillna(train['gross_square_feet'].mean(), inplace=True)

train['year_built'].fillna(train['year_built'].mean(), inplace=True)



valid['land_square_feet'].fillna(valid['land_square_feet'].mean(), inplace=True)

valid['gross_square_feet'].fillna(valid['gross_square_feet'].mean(), inplace=True)

valid['year_built'].fillna(valid['year_built'].mean(), inplace=True)
# Enconding data into indicator variable

enconding_test = pd.get_dummies(test['borough'])

enconding_train = pd.get_dummies(train['borough'])

enconding_valid = pd.get_dummies(valid['borough'])



# Concatenate on a axis

test = pd.concat([test, enconding_test], axis = 1)

train = pd.concat([train, enconding_train], axis = 1)

valid = pd.concat([valid, enconding_valid], axis = 1)



# Dropping labels

train = train.drop(['borough'], axis = 1)

test = test.drop(['borough'], axis = 1)

valid = valid.drop(['borough'], axis = 1)
y = train['sale_price'] 

x = train.drop('sale_price', axis=1)



tree = DecisionTreeRegressor()

tree = tree.fit(x, y)
# Watch the result

train.head()
linreg = LinearRegression()

lingreg = linreg.fit(x, y)

y_pred_t = linreg.predict(test)

y_pred_v = linreg.predict(valid)



dropping_valid = pd.DataFrame()

dropping_valid['sale_id'] = valid['sale_id']

dropping_valid['sale_price'] = y_pred_v



dropping_test = pd.DataFrame()

dropping_test['sale_id'] = test['sale_id']

dropping_test['sale_price'] = y_pred_t



concat_data = pd.DataFrame()

concat_data = pd.concat([dropping_valid, dropping_test])



concat_data.to_csv('Helena_160124034.csv', index=False)
print(tree.score(x, y))
test.shape
valid.shape
train.shape
concat_data.shape