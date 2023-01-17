# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# splitting 
from sklearn.model_selection import train_test_split

# metric
from sklearn.metrics import mean_squared_error

# standardize
from sklearn.preprocessing import StandardScaler

# model
from sklearn.linear_model import LinearRegression #vanilla regression
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge

# model selection
from sklearn.model_selection import GridSearchCV

# polynomial features

# helper functions
def rmse(data):
    return np.sqrt(mean_squared_error(data))

def create_submission(test, y_pred, csv_name):
    test_buff = test.copy()
    test_buff['count'] = y_pred
    
    # kick out other columns
    test_buff = test_buff['count']
    
    test_buff.to_csv(csv_name , header=True, index_label='datetime')
#read data
all_train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test_withoutLabel.csv")

all_train.set_index("datetime", inplace=True)
test.set_index("datetime", inplace=True)

y_all_train = all_train['count']
X_all_train = all_train.drop('count', axis=1)
#split
X_train, X_val, y_train, y_val = train_test_split(X_all_train, y_all_train)
#train linear regression

lr = LinearRegression()
lr.fit(X_all_train, y_all_train)

# predict
y_pred_lr = lr.predict(test)

# submit
#create_submission(test,y_pred_lr, "linear_regression.csv")

lr_ridge = Ridge(normalize=True)
params_ridge = {'alpha': np.logspace(-4,3,7)}
grid_ridge = GridSearchCV(estimator=lr_ridge, param_grid=params_ridge, n_jobs=-1, return_train_score=False, cv=10)

# fit
grid_ridge.fit(X_all_train, y_all_train)

y_pred_ridge = grid_ridge.predict(test)
#create_submission(test,y_pred_ridge, "ridge_regression.csv")
scaler = StandardScaler()


lr_kernelridge = KernelRidge()
params_kernelridge = {'alpha': np.logspace(-4,3,7), 'kernel': ['rbf'], 'gamma': np.logspace(-5,2,7)}
grid_kernelridge = GridSearchCV(estimator=lr_ridge, param_grid=params_ridge, n_jobs=-1, return_train_score=False, cv=10)

# fit
grid_kernelridge.fit(X_all_train, y_all_train)

y_pred_kernelridge = grid_kernelridge.predict(test)
create_submission(test,y_pred_kernelridge, "kernelridge_regression.csv")



