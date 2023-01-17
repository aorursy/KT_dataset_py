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
import pandas as pd



data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

data_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

sample = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

sample.head()
id_rows = data_test['Id']

data.count()
data.head()
data.info()
y = data['SalePrice']

X_test = data_test

X = data[['LotArea', 'TotalBsmtSF', 'FullBath']]

#X.drop(columns=['SalePrice'], inplace=True)

data.fillna(0, inplace=True)

X_test.fillna(0,inplace=True)




print(len(X_test.columns))

print(len(X.columns))
X_test.GarageCars.head()
X_test.count()
X_test.isna().sum()

X = pd.get_dummies(X, drop_first=True)

X_test = pd.get_dummies(X_test, drop_first=True)

X.head()
X_test.count()
from sklearn import preprocessing



cols = X.columns

x = X.values 

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

X = pd.DataFrame(x_scaled, index=X.index, columns=cols)
cols = X_test.columns

x = X_test.values 

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

new_output = pd.DataFrame(x_scaled, index=X_test.index, columns=cols)




print(len(new_output.columns))

print(len(X.columns))



for col in X:

    flag = False

    for col2 in new_output:

        if col2 == col:

            flag = True

    if flag == False:

        X.drop(columns=[col], inplace=True)

        

for col in new_output:

    flag = False

    for col2 in X:

        if col2 == col:

            flag = True

    if flag == False:

        new_output.drop(columns=[col], inplace=True)



print(len(new_output.columns))

print(len(X.columns))
new_output.count()
print(len(id_rows))
from sklearn import datasets, linear_model

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

from scipy import stats





X2 = sm.add_constant(X)

est = sm.OLS(y, X2)

est2 = est.fit()

print(est2.summary())
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
len(id_rows)
from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error, r2_score



regr = linear_model.LinearRegression()



regr.fit(X_train, y_train)



predictions = regr.predict(X_test)

final_predictions = regr.predict(new_output)

print(len(final_predictions))
my_submission = pd.DataFrame({'Id': id_rows, 'SalePrice': final_predictions})

my_submission.to_csv('submission.csv', index=False)
import math

result = mean_squared_error(y_test, predictions)

sqroot = math.sqrt(result)

format(sqroot, '.20f')
r2_score(y_test, predictions)
from sklearn import svm



clf = svm.SVC(decision_function_shape='ovo')

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
mean_squared_error(y_test, predictions)
from sklearn.tree import DecisionTreeRegressor



regr_1 = DecisionTreeRegressor(max_depth=2)

regr_2 = DecisionTreeRegressor(max_depth=4)

regr_1.fit(X_train, y_train)

regr_2.fit(X_train, y_train)

y_1 = regr_1.predict(X_test)

y_2 = regr_2.predict(X_test)

mean_squared_error(y_test, y_1)
mean_squared_error(y_test, y_2)
