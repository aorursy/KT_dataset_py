# Remove categorical variables
# Take the log on sales price
# Use decision Tree
# CV the best set of max_depth
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
# import pandas_profiling as pdp
df = pd.read_csv("../input/train.csv", usecols=columns_to_use)
df.set_index('Id', inplace=True)
pd.options.display.max_rows=5
df
df.isna().sum().sum()
y = np.log(df.SalePrice)
X = df.drop(['SalePrice'], 1)
from sklearn import tree
# make an array of depths to choose from, say 1 to 20
depths = np.arange(1, 21)
depths
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.20, random_state = 1
)
print(f"X_train shape is {X_train.shape}")
print(f"X_test shape is {X_test.shape}")
print(f"y_train shape is {y_train.shape}")
print(f"y_test shape is {y_test.shape}")
my_model = tree.DecisionTreeRegressor(max_depth=3)
my_model.fit(train_X, train_y)
# 予測されたPrice
y_predicted = my_model.predict(X_test)
y_predicted[0:5]
# train-test splitで得られた、元の（正解の）Price, 何となく近そう
y_test.values[0:5]
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_test, y_predicted))
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
my_submission.to_csv('submission_1723.csv', index=False)
import matplotlib.pyplot as plt
df.SalePrice.hist()
np.log10(df.SalePrice).hist()

