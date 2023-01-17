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

import numpy as np

from sklearn.datasets import load_boston

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import AdaBoostRegressor

from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Ridge

from sklearn.tree import DecisionTreeRegressor

from mlxtend.regressor import StackingRegressor

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error

import seaborn as sns

import pandas_profiling

from sklearn.model_selection import cross_val_score
boston=load_boston()



data=boston.data

columns=boston.feature_names



data1=pd.DataFrame(data,columns=columns)

targ=boston.target

target=pd.DataFrame(targ)

target.columns=['MEDV']
data1_tr,data1_tes,target_tr,target_tes=train_test_split(data1,target,test_size=0.33,random_state=34)
lin=LinearRegression(normalize=True)

reg_ran=RandomForestRegressor()

reg_dt = DecisionTreeRegressor(min_samples_leaf=11,min_samples_split=33, random_state=500)

reg_ridge =Ridge(random_state=500)



reg_stack = StackingRegressor(regressors=[reg_dt,reg_ran,reg_ridge],meta_regressor=lin)

#reg_stack.fit(data1_tr,target_tr)

reg_ada = AdaBoostRegressor(base_estimator=reg_stack,n_estimators=20, random_state=500)

reg_ada.fit(data1_tr,target_tr)
pred = reg_ada.predict(data1_tes)

print('MAE: {:.3f}'.format(mean_absolute_error(target_tes, pred)))

rmse = np.sqrt(mean_squared_error(target_tes,pred))

print('RMSE: {:.3f}'.format(rmse))



r2=r2_score(target_tes,pred)

print('R2: {:.3f}'.format(r2))
N = target_tes.size

p = data1_tr.shape[1]

adjr2score = 1 - ((1-r2_score(target_tes,pred))*(N - 1))/ (N - p - 1)

print("Adjusted R^2 Score %.2f" % adjr2score)
cv_dt = cross_val_score(estimator = reg_ada, X = data1_tr, y = target_tr, cv = 5)

print("Cross Validation R^2 score %.2f" % cv_dt.mean())
