# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#data = pd.read_csv("../input/oasis_cross-sectional.csv")
data = pd.read_csv('../input/oasis_longitudinal.csv')

print(data.columns)
print(data.describe())

y = data.CDR
predictors = ["M/F","Age","EDUC","SES","MMSE","eTIV","nWBV","ASF"]
XX = data[predictors]
X = pd.get_dummies(XX)    # One-hot-encoding to convert categorical data into usable form

print(X.describe())

train_X, test_X, train_y, test_y = train_test_split(X,y,random_state=0)

#Impute missing values after train test split
my_imputer = SimpleImputer()
train_X_imputed = pd.DataFrame(my_imputer.fit_transform(train_X))
test_X_imputed = pd.DataFrame(my_imputer.fit_transform(test_X))

# Decision Tree
def get_mae(max_leaf_nodes, train_X, test_X, train_y, test_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_y = model.predict(test_X)
    mae = mean_absolute_error(test_y, preds_y)
    return(mae)

print("Decision Tree results with different number of leaf nodes:")
for max_leaf_nodes in [5, 50, 500, 5000,50000]:
    my_mae = get_mae(max_leaf_nodes, train_X_imputed, test_X_imputed, train_y, test_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %f" %(max_leaf_nodes, my_mae))

# Random Forest
forest_model = RandomForestRegressor(random_state=99)
forest_model.fit(train_X_imputed, train_y)
preds_y = forest_model.predict(test_X_imputed)
print("Random Forest Results, MAE: %f" %(mean_absolute_error(test_y, preds_y)))

# Random Forest with cross-validation
my_pipeline = make_pipeline(SimpleImputer(),RandomForestRegressor(random_state=99))
scores = cross_val_score(my_pipeline,X,y,scoring='neg_mean_absolute_error')
print('Random Forest with Cross-Validation, MAE: %2f' %(-1 * scores.mean()))

# XGBoost
my_pipeline = make_pipeline(SimpleImputer(),XGBRegressor())
my_pipeline.fit(train_X, train_y)
preds_y = my_pipeline.predict(test_X)
print("XGBoost Results, MAE: %f" %(mean_absolute_error(test_y, preds_y)))

# XGBoost with parameters tuning 
xgb_model = XGBRegressor(n_estimators=1000)
xgb_model.fit(train_X_imputed, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X_imputed, test_y)], verbose=False)
preds_y = xgb_model.predict(test_X_imputed)
print("XGBoost Results with parameters tuning, MAE: %f" %(mean_absolute_error(test_y, preds_y)))
