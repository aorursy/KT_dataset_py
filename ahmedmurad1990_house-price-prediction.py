# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Imports

import math





import scipy.stats



%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns
# Read the data

X = pd.read_csv('/kaggle/input/neolen-house-price-prediction/train.csv', index_col='Id')

X_test_full = pd.read_csv('/kaggle/input/neolen-house-price-prediction/test.csv', index_col='Id')

z = pd.read_csv('/kaggle/input/neolen-house-price-prediction/sample_submission.csv')



# How is our data shaping up?

print('Our training dataset has {} rows and {} columns.'.format(X.shape[0], X.shape[1]))

print('Our test dataset has {} rows and {} columns.'.format(X_test_full.shape[0], X_test_full.shape[1]))
X.head()
X.describe()
# How expensive are houses?

print('The cheapest house sold for ${:,.0f} and the most expensive for ${:,.0f}'.format(

    X.SalePrice.min(), X.SalePrice.max()))

print('The average sales price is ${:,.0f}, while median is ${:,.0f}'.format(

    X.SalePrice.mean(), X.SalePrice.median()))

# When were the houses built?

print('Oldest house built in {}. Newest house built in {}.'.format(

    X.YearBuilt.min(), X.YearBuilt.max()))

X.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X.SalePrice              

X.drop(['SalePrice'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)
# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype == "object"]



# Select numeric columns

numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()



# One-hot encode the data (to shorten the code, we use pandas)

X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_test = pd.get_dummies(X_test)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

X_train, X_test = X_train.align(X_test, join='left', axis=1)
len(numeric_cols)
# Shape of training data (num_rows, num_columns)

print(X_train.shape)



# Number of missing values in each column of training data

missing_val_count_by_column = (X_train.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
X_train.isnull().sum() #to get all columns have or not missing values
from sklearn.impute import SimpleImputer





my_imputer = SimpleImputer()

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))

imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))



# Fill in the lines below: imputation removed column names; put them back

imputed_X_train.columns = X_train.columns

imputed_X_valid.columns = X_valid.columns
# Preprocessed training and validation features

final_imputer = SimpleImputer(strategy='median')

final_X_train = pd.DataFrame(final_imputer.fit_transform(X_train))

final_X_valid = pd.DataFrame(final_imputer.transform(X_valid))



# Imputation removed column names; put them back

final_X_train.columns = X_train.columns

final_X_valid.columns = X_valid.columns
from xgboost import XGBRegressor

from ml_metrics import rmse

# Define the model

model = XGBRegressor(n_estimators=350,max_depth=15,learning_rate=.1,random_state=1,

                     n_jobs=10, subsample=1,min_child_weight=0.6) # Your code here



# Fit the model

model.fit(final_X_train, y_train)



# Get validation predictions and MAE

preds_valid = model.predict(final_X_valid)





print("RMSE (Your appraoch):")

print(rmse(y_valid, preds_valid))
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath',

                   'BedroomAbvGr', 'TotRmsAbvGrd']

X_up = X[feature_columns]



from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X_up, y, random_state=1)

train_X.columns
X_up.columns
my_imputer_up = SimpleImputer()

imputed_X_up_train = pd.DataFrame(my_imputer_up.fit_transform(train_X))

imputed_X_up_valid = pd.DataFrame(my_imputer_up.transform(val_X))



# Fill in the lines below: imputation removed column names; put them back

imputed_X_up_train.columns = train_X.columns

imputed_X_up_valid.columns = val_X.columns
# Preprocessed training and validation features

final_imputer_up = SimpleImputer(strategy='median')

final_X_train_up = pd.DataFrame(final_imputer_up.fit_transform(train_X))

final_X_valid_up = pd.DataFrame(final_imputer_up.transform(val_X))



# Imputation removed column names; put them back

final_X_train_up.columns = train_X.columns

final_X_valid_up.columns = val_X.columns
model_up = XGBRegressor(n_estimators=350,max_depth=15,random_state=5,learning_rate=.1,

                        n_jobs=10, subsample=1,min_child_weight=0.6) # Your code here



# Fit the model

model_up.fit(final_X_train_up, train_y)



# Get validation predictions and MAE

preds_valid = model_up.predict(final_X_valid_up)





print("RMSE (Your appraoch):")

print(rmse(val_y, preds_valid))
from sklearn.model_selection import GridSearchCV
Select_parameters = {'min_samples_leaf':[1,2,3,5,6,4,8,9], 'random_state':[1,2,3,4,5,6,9,8],

                     'n_estimators':[100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]}
gsearch = GridSearchCV(estimator=XGBRegressor(),

                       param_grid = Select_parameters, 

                       scoring='neg_mean_absolute_error',

                       n_jobs=4,cv=3)



gsearch.fit(X_train,y_train)

gsearch.best_params_, gsearch.best_score_




gsearch.best_params_.get('min_samples_leaf')



gsearch.best_params_.get('random_state')

gsearch.best_params_.get('n_estimators')
final_model_up = XGBRegressor(n_estimators=gsearch.best_params_.get('n_estimators'),  learning_rate=0.1,

                           max_depth=16,random_state=gsearch.best_params_.get('random_state'),

                           subsample=1.0,min_samples_leaf=gsearch.best_params_.get('min_samples_leaf'),

                           n_jobs=4)
final_model_up.fit(final_X_train, y_train)



#  preds_test = final_model_up.predict(final_X_valid_up)

preds_test = final_model_up.predict(X_test)
y.head()
# print the top few validation predictions

print(final_model_up.predict(X_valid.head(10)))





# print the top few actual prices from validation data

print(y.head(10))
len(final_X_train_up)
# Save test predictions to file

output = pd.DataFrame({'Id': X_test_full.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)

print('done')