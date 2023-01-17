# Remove categorical variables
# Take the log on sales price
# Use decision Tree
# Best depth with CV
# only use columns 
# ['Id','LotArea', 'OverallQual','OverallCond','YearBuilt','TotRmsAbvGrd','GarageCars','WoodDeckSF',
# 'PoolArea','SalePrice']

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
columns_to_use = ['Id', 'LotArea', 'OverallQual','OverallCond','YearBuilt',
                  'TotRmsAbvGrd','GarageCars','WoodDeckSF','PoolArea','SalePrice']
columns_in_test = columns_to_use.copy()
columns_in_test.remove("SalePrice")
columns_in_test
df = pd.read_csv("../input/train.csv", usecols=columns_to_use)
df.set_index('Id', inplace=True)
pd.options.display.max_rows=5
df
df.isna().sum().sum()
y = np.log(df.SalePrice)
X = df.drop(['SalePrice'], 1)
from sklearn import tree
from sklearn.model_selection import train_test_split
X_train_and_validate, X_test, y_train_and_validate, y_test = train_test_split(
    X, y, test_size = 0.20, random_state = 1
)
print(f"X_train_and_validate shape is {X_train_and_validate.shape}")
print(f"X_test shape is {X_test.shape}")
# print(f"X_validate shape is {X_validate.shape}")
print(f"y_train_and_validate shape is {y_train_and_validate.shape}")
# print(f"y_validate shape is {y_validate.shape}")
print(f"y_test shape is {y_test.shape}")
from sklearn.metrics import mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    ''' Root mean squared error regression loss
    
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
    Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
    Estimated target values.
    '''
    return np.sqrt(mean_squared_error(y_true, y_pred))
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer
rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)
pipe_tree = make_pipeline(tree.DecisionTreeRegressor(random_state=1))
# make an array of depths to choose from, say 1 to 20
depths = np.arange(1, 21)
depths
num_leafs = [1, 5, 10, 20, 50, 100]
from sklearn.model_selection import GridSearchCV
param_grid = [{'decisiontreeregressor__max_depth':depths,
              'decisiontreeregressor__min_samples_leaf':num_leafs}]
gs = GridSearchCV(estimator=pipe_tree, param_grid=param_grid, scoring=rmse_scorer, cv=10)
gs = gs.fit(X_train_and_validate, y_train_and_validate)
print(-gs.best_score_)
print(gs.best_params_)
my_model = gs.best_estimator_
my_model.fit(X_train_and_validate, y_train_and_validate)

y_predicted = my_model.predict(X_test)
root_mean_squared_error(y_test, y_predicted)

# ここに来て、以下のすべてをtestではなく、submitと呼ぶことにする
submit = pd.read_csv('../input/test.csv', usecols=columns_in_test)
submit.set_index('Id', inplace=True)
# Treat the test data in the same way as training data. In this case, pull same columns.
submit_X = submit.replace(np.nan, 0)
# submit_X = encoder.transform(submit_X)
# test_X = test.replace(np.nan, 0, inplace=True)
# test_X = test[predictor_cols]
# Use the model to make predictions
predicted_prices = my_model.predict(submit_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)
# Get the exponent of prices
# The current predicted prices are the log of the prices
predicted_prices = np.exp(predicted_prices)
my_submission = pd.DataFrame({'Id': submit_X.index, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('07_decisiontree_CV_maxdepth5.csv', index=False)
import matplotlib.pyplot as plt
df.SalePrice.hist()
np.log10(df.SalePrice).hist()

