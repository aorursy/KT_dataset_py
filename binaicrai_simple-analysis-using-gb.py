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



import warnings

warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output.
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

sample_submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
pd.set_option('display.max_columns', 100)

train.head()
f = open('../input/house-prices-advanced-regression-techniques/data_description.txt','r')

message = f.read()

print(message)
# On train data

NA_col = pd.DataFrame(train.isna().sum(), columns = ['NA_Count'])

NA_col['% of NA'] = (NA_col.NA_Count/len(train))*100

NA_col.sort_values(by = ['% of NA'], ascending = False, na_position = 'first')



# Dropping 'PoolQC', 'MiscFeature', 'Alley', 'Fence'
pd.set_option('display.max_rows', 500)

NA_row = pd.DataFrame(train.isnull().sum(axis = 1), columns = ['NA_Row_Count'])

NA_row['% of NA'] = (NA_row.NA_Row_Count/len(train.columns))*100

NA_row.sort_values(by = ['% of NA'], ascending = False, na_position = 'first')



# We are good row-wise
# On test data

NA_col = pd.DataFrame(test.isna().sum(), columns = ['NA_Count'])

NA_col['% of NA'] = (NA_col.NA_Count/len(test))*100

NA_col.sort_values(by = ['% of NA'], ascending = False, na_position = 'first')



# Dropping 'PoolQC', 'MiscFeature', 'Alley', 'Fence'
pd.set_option('display.max_rows', 500)

NA_row = pd.DataFrame(test.isnull().sum(axis = 1), columns = ['NA_Row_Count'])

NA_row['% of NA'] = (NA_row.NA_Row_Count/len(test.columns))*100

NA_row.sort_values(by = ['% of NA'], ascending = False, na_position = 'first')



# We are good row-wise
train_data = train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1) 

test_data = test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1) 
train_wo_target = train_data.drop(['Id', 'SalePrice'], axis=1)
cols = train_wo_target.columns

num_cols = train_wo_target._get_numeric_data().columns

cat_cols = list(set(cols) - set(num_cols))
num_cols
cat_cols
X_train = train_data.copy().drop('SalePrice', axis = 1)

y_train = train_data[['Id','SalePrice']]
from sklearn.preprocessing import Imputer

from sklearn.impute import SimpleImputer
# Imputing Numerical Columns



mean_imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)

mean_imputer.fit(X_train[num_cols])

X_train[num_cols] = pd.DataFrame(mean_imputer.fit_transform(X_train[num_cols]))

test_data[num_cols] = pd.DataFrame(mean_imputer.fit_transform(test_data[num_cols]))
# Imputing Categorical Columns on train data



def impute_with_mode(x):

    max_x = x.value_counts()

    mode = max_x[max_x == max_x.max()].index[0]

    x[x.isna()] = mode

    return x



X_train[cat_cols] = X_train[cat_cols].apply(lambda x: impute_with_mode(x))
# Imputing Categorical Columns on test data

test_data[cat_cols] = test_data[cat_cols].apply(lambda x: impute_with_mode(x))
train_objs_num = len(X_train)

dataset = pd.concat(objs=[X_train, test_data], axis=0)

dataset_preprocessed = pd.get_dummies(dataset)

X_train = dataset_preprocessed[:train_objs_num]

test_data = dataset_preprocessed[train_objs_num:]
# Checking NAs

X_train.isna().sum()
test_data.isna().sum()
# Setting Index



X_train.set_index('Id', inplace = True)

y_train.set_index('Id', inplace = True)

test_data.set_index('Id', inplace = True)
# Checking shape

print(X_train.shape)

print(test_data.shape)
from sklearn import metrics

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error

from sklearn.ensemble import GradientBoostingRegressor
# Grid Search 



from sklearn.model_selection import GridSearchCV



# Model in use

GBM = GradientBoostingRegressor()

param_grid = { 

           "n_estimators" : [80, 100, 150, 200, 230, 250, 270, 300],

           "max_depth" : [1,2,3,4,5,8,10],

           "learning_rate" : [0.01, 0.05, 0.08, 0.1]}

 

CV_GBM = GridSearchCV(estimator=GBM, param_grid=param_grid, cv= 10)
%time CV_GBM.fit(X=X_train, y=y_train.values.ravel())
best_gbm_model = CV_GBM.best_estimator_

print (CV_GBM.best_score_, CV_GBM.best_params_)
# Predictions



pred_train_gbm = best_gbm_model.predict(X_train)

pred_test_gbm = best_gbm_model.predict(test_data)
print('Mean Squared Log Error: ', metrics.mean_squared_log_error(y_train, pred_train_gbm).round(5))

print('Mean Squared Error: ', metrics.mean_squared_error(y_train, pred_train_gbm).round(5))

print('R-squared: ', metrics.r2_score(y_train, pred_train_gbm).round(5))
y_pred_test_gbm = pd.DataFrame(pred_test_gbm, columns = ['SalePrice'])

y_pred_test_gbm['Id'] = test['Id']
columnsTitles = ['Id', 'SalePrice']



submission = y_pred_test_gbm.reindex(columns=columnsTitles)

submission .head()
filename = 'House_Pricing_GB.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)