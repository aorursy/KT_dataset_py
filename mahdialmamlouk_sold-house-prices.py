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
train=pd.read_csv('../input/house-55/train (3).csv')

test=pd.read_csv('../input/house-55/test (3).csv')

print ("Train data shape:", train.shape)

print ("Test data shape:",test.shape)
train.SalePrice.describe()
print ("Skew is:", train.SalePrice.skew())
target = np.log(train.SalePrice)

print ("Skew is:", target.skew())
numeric_features = train.select_dtypes(include=[np.number])

numeric_features.dtypes

train.OverallQual.unique()
quality_pivot = train.pivot_table(index='OverallQual',

                  values='SalePrice', aggfunc=np.median)

quality_pivot
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])

nulls.columns = ['Null Count']

nulls.index.name = 'Feature'

nulls
print ("Unique values are:", train.MiscFeature.unique())
categoricals = train.select_dtypes(exclude=[np.number])

categoricals.describe()
print ("Original: \n")

print (train.Street.value_counts(), "\n")
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)

test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
print ('Encoded: \n')

print (train.enc_street.value_counts())
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
sum(data.isnull().sum() != 0)
y = np.log(train.SalePrice)

X = data.drop(['SalePrice', 'Id'], axis=1)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

X_train, X_test, y_train, y_test = train_test_split(

                          X, y, random_state=45, test_size=.23)
from sklearn import linear_model

lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
from xgboost import XGBRegressor

 

parameters = [{'n_estimators': list(range(10, 300, 100)), 'learning_rate': [x / 100 for x in range(5, 101, 5)]}]

from sklearn.model_selection import GridSearchCV

gsearch = GridSearchCV(estimator=XGBRegressor(), param_grid = parameters,

                       scoring='neg_mean_absolute_error', n_jobs=4,cv=5)



gsearch.fit(X_train, y_train)



gsearch.best_params_.get('n_estimators'), gsearch.best_params_.get('learning_rate')
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)

from sklearn.metrics import mean_squared_error

print ('RMSE is: \n', mean_squared_error(y_test, predictions))
submission = pd.DataFrame()

submission['Id'] = test.Id
feats = test.select_dtypes(

        include=[np.number]).drop(['Id'], axis=1).interpolate()
predictions = model.predict(feats)
final_predictions =  gsearch.best_params_.get('learning_rate')
print ("Original predictions are: \n", predictions[:5], "\n")

print ("Final predictions are: \n", final_predictions[:5])
submission['SalePrice'] = final_predictions

submission.head()
submission.to_csv('submission1.csv', index=True)
