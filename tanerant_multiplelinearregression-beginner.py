



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

train_data.head()

test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

test_data.head()
train_data.shape
train_data.corr()
x = train_data._get_numeric_data() 
X_ = x.drop("SalePrice", axis = 1)

X = X_[["OverallQual","TotalBsmtSF","GrLivArea"]]
X.isnull().values.any()
y = x["SalePrice"]
y[0:3]
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict#Train test ayrımı
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.20, random_state= 42)
X_train.shape

X_valid.shape
y_train.shape
y_valid.shape
##Statsmodels

import statsmodels.api as sm 
model = sm.OLS(y_train, X_train).fit()

model.summary()
#scikit-learn model

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
lm = LinearRegression()

model = lm.fit(X_train, y_train)
model.intercept_  
model.coef_
new_data = [[6],[926],[1604]]

new_data = pd.DataFrame(new_data).T
model.predict(new_data)
rmse =np.sqrt(mean_squared_error(y_train,model.predict(X_train)))
rmse
rmse =np.sqrt(mean_squared_error(y_valid,model.predict(X_valid)))
rmse