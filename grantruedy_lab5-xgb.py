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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import Imputer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.metrics import mean_squared_log_error

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))
train_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")



def get_cat_cols(df):

    return  [col for col in df.columns if df[col].dtype == 'object']



y = np.log1p(train_data.SalePrice)

# test is meant for predictions and doesn't contain any price data. I need to provide it.

cand_train_predictors = train_data.drop(['Id', 'SalePrice'], axis=1)

cand_test_predictors = test_data.drop(['Id'], axis=1)



cat_cols = get_cat_cols(cand_train_predictors)



cand_train_predictors[cat_cols] = cand_train_predictors[cat_cols].fillna('NotAvailable')

cand_test_predictors[cat_cols] = cand_test_predictors[cat_cols].fillna('NotAvailable')



encoders = {}



for col in cat_cols:

    encoders[col] = LabelEncoder()

    val = cand_train_predictors[col].tolist()

    val.extend(cand_test_predictors[col].tolist())

    encoders[col].fit(val)

    cand_train_predictors[col] = encoders[col].transform(cand_train_predictors[col])+1

    cand_test_predictors[col] = encoders[col].transform(cand_test_predictors[col])+1

        

    

corr_matrix = cand_train_predictors.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

cols_to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

print('Highly correlated features(will be droped):',cols_to_drop)



cand_train_predictors = cand_train_predictors.drop(cols_to_drop, axis=1)

cand_test_predictors = cand_test_predictors.drop(cols_to_drop, axis=1)



print(cand_train_predictors.shape)

print(cand_test_predictors.shape)



train_set, test_set = cand_train_predictors.align(cand_test_predictors,join='left', axis=1)

train_set = np.log1p(train_set)

test_set = np.log1p(test_set)
from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LassoCV

from sklearn.model_selection import KFold



regressor = XGBRegressor(n_estimators=1000, 

                             random_state=1,

                         learning_rate=0.07300000000000001)



my_model = regressor

kf = KFold(5, shuffle=True, random_state=1).get_n_splits(train_set)

rmse = np.sqrt(-cross_val_score(my_model, train_set,y, scoring="neg_mean_squared_error", cv=kf, verbose=0))



# my_model = make_pipeline(imputer, scaler,select,regressor)

# train_x, test_x, train_y, test_y = train_test_split(train_set.values, y)

my_model.fit(train_set.values, y)

print('-'*80)

print(my_model.score(train_set.values,y))

print('rmse:',rmse.mean())

train_pred = my_model.predict(train_set.values)

print('rmsle: ',np.sqrt(mean_squared_log_error(y, train_pred)))

print('rmse: ',np.sqrt(mean_squared_error(train_data.SalePrice, np.expm1(train_pred))))

print('mae: ',mean_absolute_error(train_data.SalePrice, np.expm1(train_pred)))
predicted_prices = np.expm1(my_model.predict(test_set.values))

print(predicted_prices[:5])



my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_prices})

my_submission.Id = my_submission.Id.astype(int)

my_submission.to_csv('XGBsubmission.csv', index=False)