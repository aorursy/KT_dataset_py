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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df
df.head()
df.columns




y = df['SalePrice']



X= df['LotArea']

X=X.values.reshape(-1,1)

X

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

X_test = test[predictor_cols]

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train)
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)

X_label=np.linspace(500, 200000, 1000).reshape(1000, 1)

y_pred = lin_reg.predict(X_label)



plt.plot(X,y,"b.")

plt.plot(X_label,y_pred,"r.")

plt.show()
y_pred =lin_reg.predict(X) 
y_pred = lin_reg.predict(X_test)

mean_squared_error(y_test, y_pred)
from math import sqrt

sqrt(mean_squared_error(y_test, y_pred))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

poly_features = PolynomialFeatures(degree=2, include_bias=False)

X_poly = poly_features.fit_transform(X_train)

print(X_poly)
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X_poly, y_train)
X_new_poly = poly_features.transform(X_new)

y_new = lin_reg.predict(X_new_poly)

X_new=np.linspace(500, 200000, 1000).reshape(1000, 1)



plt.plot(X,y,"b.")

plt.plot(X_new,y_new,"r.")

plt.show()
X_test_poly = poly_features.fit_transform(X_test)

y_pred = lin_reg.predict(X_test_poly)

mean_squared_error(y_test, y_pred)


from math import sqrt

sqrt(mean_squared_error(y_test, y_pred))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
import numpy as np

import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC





train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')



train_y = train.SalePrice

predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']





train_X = train[predictor_cols]



svm_clf = Pipeline([

    ("scaler", StandardScaler()),

    ("linear_svc", LinearSVC(C=1, loss="hinge")),

    ])



svm_clf.fit(train_X,train_y)

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test_X = test[predictor_cols]



predicted_prices = svm_clf.predict(test_X)



print(predicted_prices)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

my_submission.to_csv('submission_Zihao_Han.csv', index=False)