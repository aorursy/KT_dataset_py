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
import numpy as np

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

from sklearn.metrics import precision_score, recall_score

from sklearn import preprocessing

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import skew 

%matplotlib inline



import csv

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train.head()
test.head()
test.describe()
train.describe()
test.dtypes
train.dtypes
#summary of testing dataset

print(test.info())
#summary of testing dataset

print(train.info())
X = train.iloc[:,0:-1] 

Y = train.SalePrice 
X
Y.shape
train.dtypes.sample(10)

target = train.SalePrice
cols_with_missing = [col for col in train.columns

if train[col].isnull().any()]
cols_with_missing1 = [col for col in test.columns

if test[col].isnull().any()]
train_predictors = train.drop(['Id', 'SalePrice'] + cols_with_missing, axis=1) 
test_predictors = test.drop(['Id'] + cols_with_missing1, axis=1) 
train_predictors.dtypes.sample(10)

one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor



def get_mae(X, y):

    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention

    return -1 * cross_val_score(RandomForestRegressor(50), 

                                X, y, 

                                scoring = 'neg_mean_absolute_error').mean()



predictors_without_categoricals = train_predictors.select_dtypes(exclude=['object'])



mae_without_categoricals = get_mae(predictors_without_categoricals, target)



mae_one_hot_encoded = get_mae(one_hot_encoded_training_predictors, target)



print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals)))

print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)

one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)

final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,

                                                                    join='left', 

                                                                    axis=1)
one_hot_encoded_test_predictors
X
test_predictors
X = one_hot_encoded_test_predictors.iloc[:,0:-1] 

Y = train.SalePrice 

X.shape
Y.shape
#summary of SalesPrice

train['SalePrice'].describe()
corr = train.corr()

corr.sort_values(["SalePrice"], ascending = False, inplace = True)

print(corr.SalePrice)
# We will see if there's a relationship between SalesPrice and OverallQual: overall material and finish of the house.

# 1-10 where 1=Very Poor and 10=Very Excellent

sns.barplot(train.OverallQual,train.SalePrice)

# pulling data into  the target (y) which is the SalePrice and predictors (X)

train_y = train.SalePrice

pred_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
#prediction data

train_x = train[pred_cols]



model =  LogisticRegression()



model.fit(train_x, train_y)
# pulling same columns from the test data

test_x = test[pred_cols]

pred_prices = model.predict(test_x)

print(pred_prices)
#save file

ayesha_submission = pd.DataFrame({'Id': test.Id, 'SalePrice' : pred_prices})

ayesha_submission.to_csv('submission.csv', index=False)