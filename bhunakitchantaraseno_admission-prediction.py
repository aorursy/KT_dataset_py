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

        data = pd.read_csv(os.path.join(dirname, filename))

        break



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import *

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
sns.heatmap(data.corr())
data.hist(bins=50, figsize=(20, 15))

plt.show()
data_test = data.iloc[400:]

data_train = data.iloc[:400]

data_test.reset_index(inplace=True)

data_train.reset_index(inplace=True)
data_test = data_test.drop('index', axis=1)

data_train = data_train.drop('index', axis=1)
features_train = data_train.drop(['Chance of Admit ', 'Serial No.'], axis=1)

labels_train = data_train['Chance of Admit ']

features_test = data_test.drop(['Chance of Admit ', 'Serial No.'], axis=1)

labels_test = data_test['Chance of Admit ']
features_test
scaler = StandardScaler()

features_train = scaler.fit_transform(features_train)

features_test = scaler.fit_transform(features_test)
x_train, x_test, y_train, y_test = train_test_split(features_train, labels_train, test_size=0.2, random_state=42)
#Decision Tree Regressor
dt_reg = DecisionTreeRegressor(random_state=42)

dt_mse = -cross_val_score(dt_reg, x_train, y_train, cv=3, scoring='neg_mean_squared_error')



dt_rmse = np.sqrt(dt_mse)

dt_rmse
dt_reg.fit(x_train, y_train)



dt_pred = dt_reg.predict(x_test)



np.sqrt(mean_squared_error(dt_pred, y_test))
# Random Forest Regressor
rf_reg = RandomForestRegressor()



rf_reg.fit(x_train, y_train)



rf_pred = rf_reg.predict(x_test)



np.sqrt(mean_squared_error(rf_pred, y_test))
rf_mse = -cross_val_score(rf_reg, x_train, y_train, cv=3, scoring='neg_mean_squared_error')



rf_rmse = np.sqrt(rf_mse)



rf_rmse
# Forest is better



params_grid = [{'n_estimators':[50, 70, 80, 200, 400], 'max_features':['auto', 'log2', 'sqrt']}]



grid_search = GridSearchCV(rf_reg, params_grid, cv=3, scoring='neg_mean_squared_error', return_train_score=True)



grid_search.fit(x_train, y_train)
final_model = grid_search.best_estimator_
np.sqrt(mean_squared_error(final_model.predict(x_test), y_test))
final_pred = final_model.predict(features_test)



final_mse = mean_squared_error(final_pred, labels_test)



final_rmse = np.sqrt(final_mse)



final_rmse