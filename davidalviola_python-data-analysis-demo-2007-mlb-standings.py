# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# we'll import several things from various parts of sklearn.
# sklearn is the library that provides us machine learning models
# and various utility functions for machine learning
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# data visualization
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(14, 9))
df = pd.read_csv('/kaggle/input/2007-mlb-standings/MLB2007Standings.csv')
df
#Are offensive statistics (HR, RBI, BattingAvg) more indicative of winning percentage?
Offensive = df[['HR', 'RBI', 'BattingAvg']]
Offensive
#Are defensive statistics (strikeouts, ERA, HitsAllowed) more indicative of winning percenatage?
Defensive = df[['StrikeOuts', 'ERA', 'HitsAllowed']]
Defensive
win_pct = df[['WinPct']]
win_pct
# Split sample for offensive model
Offensive_train, Offensive_test, off_win_pct_train, off_win_pct_test = train_test_split(Offensive, win_pct)
print('Offensive training split')
print(Offensive_train)
print('Win percentage training split')
print(off_win_pct_train)
print('Offensive test split')
print(Offensive_test)
print('Win percentage test split')
print(off_win_pct_test)
#Split sample for defensive model
Defensive_train, Defensive_test, dfn_win_pct_train, dfn_win_pct_test = train_test_split(Defensive, win_pct)
#Fit the offensive model to the offensive training data
off_lr = LinearRegression()

off_lr.fit(Offensive_train, off_win_pct_train)

m = off_lr.coef_
b = off_lr.intercept_

print('Offensive model:  y = {:.5f}x + {:.2f}'.format(m[0,0], b[0]))
#Fit the defensive model to the defensive training data
dfn_lr = LinearRegression()

dfn_lr.fit(Defensive_train, dfn_win_pct_train)

m = dfn_lr.coef_
b = dfn_lr.intercept_

print('Defensive model:  y = {:.5f}x + {:.2f}'.format(m[0,0], b[0]))
off_lr_predicted_win_pct_train = off_lr.predict(Offensive_train)
plt.scatter(off_win_pct_train, off_lr_predicted_win_pct_train)
plt.xlabel('Actual win %')
plt.ylabel('Predicted win %')
plt.title('Offensive Model:  Predicted win % vs Actual win %')
dfn_lr_predicted_win_pct_train = dfn_lr.predict(Defensive_train)
plt.scatter(dfn_win_pct_train, dfn_lr_predicted_win_pct_train)
plt.xlabel('Actual win %')
plt.ylabel('Predicted win %')
plt.title('Defensive Model:  Predicted win % vs Actual win %')
homeruns = Offensive_train['HR']
actual_win_pct = off_win_pct_train

plt.scatter(homeruns, actual_win_pct, label='Actual')
plt.scatter(homeruns, off_lr_predicted_win_pct_train, label='Predicted')
plt.xlabel('Homeruns')
plt.ylabel('Win %')
plt.legend()
plt.title('Win %: Actual vs Linear Regression Offensive Model Predictions for Training Data')
strikeouts = Defensive_train['StrikeOuts']
actual_win_pct = dfn_win_pct_train

plt.scatter(strikeouts, actual_win_pct, label='Actual')
plt.scatter(strikeouts, dfn_lr_predicted_win_pct_train, label='Predicted')
plt.xlabel('Strikeouts')
plt.ylabel('Win %')
plt.legend()
plt.title('Win %: Actual vs Linear Regression Defensive Model Predictions for Training Data')
print("MSE for offensive model:  " + str(mean_squared_error(off_win_pct_train, off_lr_predicted_win_pct_train)))
print("MSE for defensive model:  " + str(mean_squared_error(dfn_win_pct_train, dfn_lr_predicted_win_pct_train)))
off_lr_train_residual = off_win_pct_train - off_lr_predicted_win_pct_train
dfn_lr_train_residual = dfn_win_pct_train - dfn_lr_predicted_win_pct_train

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.scatter(off_win_pct_train, off_lr_train_residual)
ax1.set(ylabel='Residual ($y - \hat{y}$)', xlabel='Win %', title='Offensive Model')

ax2.scatter(dfn_win_pct_train, dfn_lr_train_residual)
ax2.set(title='Defensive Model', xlabel='Win %')

fig.suptitle('Training Data Set Residuals')
off_lr_predicted_win_pct_test = off_lr.predict(Offensive_test)

plt.scatter(off_win_pct_test, off_lr_predicted_win_pct_test)
plt.xlabel('Actual win %')
plt.ylabel('Predicted win %')
plt.title('Offensive Model:  Predicted win % vs Actual win %')
dfn_lr_predicted_win_pct_test = dfn_lr.predict(Defensive_test)

plt.scatter(dfn_win_pct_test, dfn_lr_predicted_win_pct_test)
plt.xlabel('Actual win %')
plt.ylabel('Predicted win %')
plt.title('Defensive Model:  Predicted win % vs Actual win %')
homeruns = Offensive_test['HR']
actual_win_pct = off_win_pct_test

plt.scatter(homeruns, actual_win_pct, label='Actual')
plt.scatter(homeruns, off_lr_predicted_win_pct_test, label='Predicted')
plt.xlabel('Homeruns')
plt.ylabel('Win %')
plt.legend()
plt.title('Win %: Actual vs Linear Regression Offensive Model Predictions for Test Data')
strikeouts = Defensive_test['StrikeOuts']
actual_win_pct = dfn_win_pct_test

plt.scatter(strikeouts, actual_win_pct, label='Actual')
plt.scatter(strikeouts, dfn_lr_predicted_win_pct_test, label='Predicted')
plt.xlabel('Strikeouts')
plt.ylabel('Win %')
plt.legend()
plt.title('Win %: Actual vs Linear Regression Defensive Model Predictions for Test Data')
print("MSE for offensive model:  " + str(mean_squared_error(off_win_pct_test, off_lr_predicted_win_pct_test)))
print("MSE for defensive model:  " + str(mean_squared_error(dfn_win_pct_test, dfn_lr_predicted_win_pct_test)))
off_lr_test_residual = off_win_pct_test - off_lr_predicted_win_pct_test
dfn_lr_test_residual = dfn_win_pct_test - dfn_lr_predicted_win_pct_test

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.scatter(off_win_pct_test, off_lr_test_residual)
ax1.set(ylabel='Residual ($y - \hat{y}$)', xlabel='Win %', title='Offensive Model')

ax2.scatter(dfn_win_pct_test, dfn_lr_test_residual)
ax2.set(title='Defensive Model', xlabel='Win %')

fig.suptitle('Training Data Set Residuals')