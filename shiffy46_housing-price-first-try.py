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
pd.options.display.max_rows = 999
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

train_corr = train[['GrLivArea','GarageArea','1stFlrSF','TotalBsmtSF','SalePrice','LotFrontage']]
train.corr()['SalePrice']
import seaborn as sns

ax = sns.pairplot(train_corr)
correlation = train.corr()

highestCorrelationCols = list(correlation.nlargest(8,'SalePrice')[['SalePrice']].index)
import numpy as np
#train = train[highestCorrelationCols]

X = train_corr.drop('SalePrice',axis=1)

y = train_corr['SalePrice']
# https://medium.com/vickdata/a-simple-guide-to-scikit-learn-pipelines-4ac0d974bdcf

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures

from sklearn.impute import SimpleImputer

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(categories='auto'))])



numeric_features = ['GrLivArea','GarageArea','1stFlrSF','TotalBsmtSF','LotFrontage']

categorical_features = ['OverallQual','FullBath']

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(

transformers=[('num', numeric_transformer, numeric_features)])

#('cat', categorical_transformer, categorical_features)])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression

from sklearn.svm import LinearSVR

from sklearn.svm import SVR

rf = Pipeline(steps=[('preprocessor', preprocessor),

                      ("linear_svr", LinearRegression())])

rf.fit(X,y)

print("model score: %.3f" % rf.score(X_test, y_test))
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test_X = test[['GrLivArea','GarageArea','1stFlrSF','TotalBsmtSF','LotFrontage']]
predicted_prices = rf.predict(test_X)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

print(len(my_submission))

my_submission.to_csv('submission-Linear_SamShifflett.csv', index=False)