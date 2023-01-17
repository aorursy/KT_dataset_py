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
import math 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
files = os.listdir("../input")
x_file = files[3]
y_file = files[2]
Xs = pd.read_csv("../input/" + x_file)
Ys = pd.read_csv("../input/" + y_file)
X_columns = Xs.columns 
Xs = Xs.drop(columns=[X_columns[0],X_columns[1],X_columns[2],X_columns[3], X_columns[4], X_columns[5]])
Xs = Xs.drop(columns=['EmployeeID', 'RatingTableID'])
Xs = Xs.drop(columns=['CNVersion'])
Xs; 
Ys_Overall = Ys['OverallScore']
Ys_Overall; 
Xs = Xs.drop(columns=['Tax_Year'])
Ys_Overall = Ys_Overall[np.isfinite(Ys_Overall)]
Xs = Xs.loc[Ys_Overall[np.isfinite(Ys_Overall)].index]

trainx, xtestvals,trainy,ytestvals = train_test_split(Xs, Ys_Overall, test_size = 0.20, random_state=42)
def rmse(p,targets): 
    return np.sqrt((p - targets)**2).mean() 
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV 
params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,6)], 'max_depth':[2,3,4],'subsample':[i/10.0 for i in range(6,11)]} 
xgb = XGBRegressor(nthread=-1) 
grid = GridSearchCV(xgb,params,scoring='neg_mean_squared_error')
grid.fit(trainx,trainy)

grid.best_score_
test_data = pd.read_csv("../input/testFeatures.csv")
test_data = test_data[trainx.columns]
test_vals = test_data.copy()
test_vals.head() 
predictions = grid.predict(test_vals)
output=pd.DataFrame({'Id':np.arange(1,2127),'OverallScore':predictions})
output.to_csv('out_xgb.csv', index = False)
