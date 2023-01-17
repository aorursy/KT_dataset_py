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
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.metrics import r2_score

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor 

from sklearn.ensemble import AdaBoostRegressor





df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train = pd.DataFrame(df_train)

test = pd.DataFrame(df_test)
train.info()
test.info()
train.dtypes
train = train.drop(['Alley', 'PoolQC','FireplaceQu','MiscFeature', 'Fence', 'LotFrontage', 'Id' ], axis = 1)
test = test.drop(['Alley', 'PoolQC','FireplaceQu','MiscFeature', 'Fence', 'LotFrontage','Id' ], axis = 1)
object1 = train.select_dtypes(include=['object']).columns

object1

object_test = test.select_dtypes(include=['object']).columns

object_test
integer = train.select_dtypes(include=['int64']).columns

integer
integer_test = test.select_dtypes(include=['int64']).columns

integer_test
float1 = train.select_dtypes(include=['float64']).columns

float1



float_test = test.select_dtypes(include=['float64']).columns

float_test
for column in object1:

    train[column].fillna(train[column].mode()[0], inplace = True)

    #test[column].fillna(test[column].mode()[0], inplace=True)
for column in object_test:

    #train[column].fillna(train[column].mode()[0], inplace = True)

    test[column].fillna(test[column].mode()[0], inplace=True)
for column in integer:

    train[column].fillna(train[column].mean, inplace = True)

    #test[column].fillna(test[column].mean(), inplace=True) # does not conatin sale price 
for column in integer_test:

    #train[column].fillna(train[column].mean, inplace = True)

    test[column].fillna(test[column].mean(), inplace=True) 
for column in float1:

    train[column].fillna(train[column].mean(), inplace = True)

    test[column].fillna(test[column].mean(), inplace=True)
for column in float_test:

    #train[column].fillna(train[column].mean(), inplace = True)

    test[column].fillna(test[column].mean(), inplace=True)
train.head()
test.head()
from sklearn.preprocessing import LabelEncoder, StandardScaler

le= LabelEncoder()

sc = StandardScaler()
for column in object1:

    train[column] = le.fit_transform(train[column])

    test[column] = le.fit_transform(test[column])
#test["BsmtFinType2"].fillna(test["BsmtFinType2"].mode()[0], inplace=True)

#test["BsmtFinType1"].fillna(test["BsmtFinType1"].mode()[0], inplace=True)

#test["BsmtFinSF1"].fillna(test["BsmtFinSF1"].mode()[0], inplace=True)
test.head()

test1 = test.copy()

test1.info()
X = train.drop(['SalePrice'], 1)

y = train['SalePrice']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 43)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, BayesianRidge, ridge_regression

from sklearn import metrics

from sklearn.metrics import r2_score, mean_squared_error



lr = LogisticRegression(max_iter=1000)

rd = Ridge()

la = Lasso()

byrd = BayesianRidge()



models = [lr, rd, la, byrd]

for model in models:

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mod = model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2_score = metrics.r2_score(y_test, y_pred)

    RMSE = np.sqrt(metrics.mean_squared_error(y_test,y_pred))

    print('\n', model,'\n', 'R2_score:', r2_score,'\n', 'RMSE:', '\n', RMSE, '\n')
byrd.fit(X_train, y_train)

y_pred = model.predict(test1).round(3)

print(y_pred)
test_prediction_byrd = pd.DataFrame(y_pred, columns=['SalePrice'])

test_prediction_byrd.head()
ID = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test_id = ID['Id']

ID = pd.DataFrame(test_id, columns=['Id'])
result = pd.concat([ID,test_prediction_byrd], axis=1)

result
result.to_csv('submission.csv',index =False)