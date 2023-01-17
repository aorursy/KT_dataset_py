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
#import all the required libraries.

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.metrics import mean_squared_error

from collections import Counter
# Import dataset

data_df = pd.read_csv("../input/ToyotaCorolla.csv")
# check the data using head()

data_df.head()
# Count the number of records in each column.

data_df.count()
# Check the stats of the data

data_df.describe()
# Check if there is any null value in the dataset.

data_df.isnull().sum()
# Check Correlation amoung parameters

corr = data_df.corr()

fig, ax = plt.subplots(figsize=(8,8))

# Generate a heatmap

sns.heatmap(corr, cmap = 'magma', annot = True, fmt = ".2f")

plt.xticks(range(len(corr.columns)), corr.columns)



plt.yticks(range(len(corr.columns)), corr.columns)



plt.show()
# plot regplots  for Age, KM, CC & HP against Price

f, axes = plt.subplots(2,2, figsize=(12,8))

# Age Vs Price

sns.regplot(x = 'Price', y = 'Age', data = data_df, ax = axes[0,0], scatter_kws={'alpha':0.6})

axes[0,0].set_xlabel('Price', fontsize = 14)

axes[0,0].set_ylabel('Age', fontsize=14)

axes[0,0].yaxis.tick_left()



# KM Vs Price

sns.regplot(x = 'Price', y = 'KM', data = data_df, ax = axes[0,1], scatter_kws={'alpha':0.6})

axes[0,1].set_xlabel('Price', fontsize = 14)

axes[0,1].set_ylabel('KM', fontsize=14)

axes[0,1].yaxis.set_label_position("right")

axes[0,1].yaxis.tick_right()



# CC Vs Price

sns.regplot(x = 'Price', y = 'CC', data = data_df, ax = axes[1,0], scatter_kws={'alpha':0.6})

axes[1,0].set_xlabel('Price', fontsize = 14)

axes[1,0].set_ylabel('CC', fontsize=14)

axes[1,0].yaxis.tick_left()



# Weight Vs Price

sns.regplot(x = 'Price', y = 'Weight', data = data_df, ax = axes[1,1], scatter_kws={'alpha':0.6})

axes[1,1].set_xlabel('Price', fontsize = 14)

axes[1,1].set_ylabel('Weight', fontsize=14)

axes[1,1].yaxis.set_label_position("right")

axes[1,1].yaxis.tick_right()



plt.show()
# Create the clasiification.

data_df = pd.get_dummies(data_df)
data_df.head()
from sklearn.linear_model import LinearRegression
X_simple_lreg = data_df[["Age"]].values

y_simple_lreg = data_df["Price"].values



print(X_simple_lreg[0:5])

print(y_simple_lreg[0:5])
# Create train test dataset

from sklearn.model_selection import train_test_split

X_train_slreg, X_test_slreg, y_train_slreg, y_test_slreg = train_test_split(X_simple_lreg,y_simple_lreg, test_size = 0.25, random_state = 4)

print('Train Dataset : ', X_train_slreg.shape, y_train_slreg.shape)

print('Test Dataset : ', X_test_slreg.shape, y_test_slreg.shape)
simple_lreg = LinearRegression()

simple_lreg.fit(X_train_slreg, y_train_slreg)

print('Intercept : ', simple_lreg.intercept_)

print('Slope : ', simple_lreg.coef_)
# Use the model to predict the test dataset.

y_simplelreg_pred_test = simple_lreg.predict(X_test_slreg)



# Use the model to predict the train dataset.

y_simplelreg_pred_train = simple_lreg.predict(X_train_slreg)
# Calculate the eualuation metrics of the model.

from sklearn.metrics import r2_score

r2_score_slreg_train = r2_score(y_simplelreg_pred_train, y_train_slreg)

r2_score_slreg_test = r2_score(y_simplelreg_pred_test, y_test_slreg)

rmse_slreg = np.sqrt(mean_squared_error(y_simplelreg_pred_test, y_test_slreg)**2)

print('r2_ score for train dataset for simple linear reg : ', r2_score_slreg_train)

print('r2_ score for test dataset for simple linear reg : ', r2_score_slreg_test)

print('root mean squared error for simple linear reg : ', rmse_slreg)
# Separating the independent and dependent variable.

X_multi_lreg = data_df.drop('Price', axis = 1).values

y_multi_lreg = data_df["Price"].values.reshape(-1,1)
# Create train test dataset

from sklearn.model_selection import train_test_split

X_train_mlreg, X_test_mlreg, y_train_mlreg, y_test_mlreg = train_test_split(X_multi_lreg,y_multi_lreg, test_size = 0.25, random_state = 4)

print('Train Dataset : ', X_train_mlreg.shape, y_train_mlreg.shape)

print('Test Dataset : ', X_test_mlreg.shape, y_test_mlreg.shape)
multi_lreg = LinearRegression()

multi_lreg.fit(X_train_mlreg, y_train_mlreg)

print('Intercept : ', multi_lreg.intercept_)

print('Slope : ', multi_lreg.coef_)
# Use the model to predict the test dataset.

y_mlreg_pred_test = multi_lreg.predict(X_test_mlreg)



# Use the model to predict the train dataset.

y_mlreg_pred_train = multi_lreg.predict(X_train_mlreg)
# Have a look at the predicted & actual values.

print(y_mlreg_pred_test[0:5])

print(y_test[0:5])



print(y_mlreg_pred_train[0:5])

print(y_train[0:5])
# Calculate the eualuation metrics of the model.

from sklearn.metrics import r2_score

r2_score_mlreg_train = r2_score(y_mlreg_pred_train, y_train_mlreg)

r2_score_mlreg_test = r2_score(y_mlreg_pred_test, y_test_mlreg)

rmse_mlreg = np.sqrt(mean_squared_error(y_mlreg_pred_test, y_test_mlreg)**2)

print('r2_ score for train dataset for multi linear reg : ', r2_score_mlreg_train)

print('r2_ score for test dataset for multi linear reg : ', r2_score_mlreg_test)

print('root mean squared error for multi linear reg : ', rmse_mlreg)
# Separating the independent and dependent variable.

X_ridge_reg = data_df.drop('Price', axis = 1).values

y_ridge_reg = data_df["Price"].values.reshape(-1,1)
# Create train test dataset

from sklearn.model_selection import train_test_split

X_train_ridge_reg, X_test_ridge_reg, y_train_ridge_reg, y_test_ridge_reg = train_test_split(X_ridge_reg,y_ridge_reg, test_size = 0.25, random_state = 4)

print('Train Dataset : ', X_train_ridge_reg.shape, y_train_ridge_reg.shape)

print('Test Dataset : ', X_test_ridge_reg.shape, y_test_ridge_reg.shape)
from sklearn.linear_model import Ridge



## training the model



ridgeReg = Ridge(alpha=0.05, normalize=True)



ridgeReg.fit(X_train_ridge_reg,y_train_ridge_reg)



# Use the model to predict the test dataset.

y_ridgereg_pred_test = ridgeReg.predict(X_test_ridge_reg)



# Use the model to predict the train dataset.

y_ridgereg_pred_train = ridgeReg.predict(X_train_ridge_reg)



# Calculate the eualuation metrics of the model.

from sklearn.metrics import r2_score

r2_score_ridgereg_train = r2_score(y_ridgereg_pred_train, y_train_ridge_reg)

r2_score_ridgereg_test = r2_score(y_ridgereg_pred_test, y_test_ridge_reg)

rmse_ridgereg = np.sqrt(mean_squared_error(y_ridgereg_pred_test, y_test_ridge_reg)**2)

print('r2_ score for train dataset for multi linear reg : ', r2_score_ridgereg_train)

print('r2_ score for test dataset for multi linear reg : ', r2_score_ridgereg_test)

print('root mean squared error for multi linear reg : ', rmse_ridgereg)
from sklearn.linear_model import Lasso



## training the model



lassoReg = Lasso(alpha=0.3, normalize=True)



lassoReg.fit(X_train_ridge_reg,y_train_ridge_reg)



# Use the model to predict the test dataset.

y_lassoreg_pred_test = lassoReg.predict(X_test_ridge_reg)



# Use the model to predict the train dataset.

y_lassoreg_pred_train = lassoReg.predict(X_train_ridge_reg)



# Calculate the eualuation metrics of the model.

from sklearn.metrics import r2_score

r2_score_lassoreg_train = r2_score(y_lassoreg_pred_train, y_train_ridge_reg)

r2_score_lassoreg_test = r2_score(y_lassoreg_pred_test, y_test_ridge_reg)

rmse_lassoreg = np.sqrt(mean_squared_error(y_lassoreg_pred_test, y_test_ridge_reg)**2)

print('r2_ score for train dataset for multi linear reg : ', r2_score_lassoreg_train)

print('r2_ score for test dataset for multi linear reg : ', r2_score_lassoreg_test)

print('root mean squared error for multi linear reg : ', rmse_lassoreg)
from sklearn.linear_model import ElasticNet



## training the model



elasticNetReg = ElasticNet(alpha=1, l1_ratio=0.5, normalize=True)



elasticNetReg.fit(X_train_ridge_reg,y_train_ridge_reg)



# Use the model to predict the test dataset.

y_elasticNetReg_pred_test = elasticNetReg.predict(X_test_ridge_reg)



# Use the model to predict the train dataset.

y_elasticNetReg_pred_train = elasticNetReg.predict(X_train_ridge_reg)



# Calculate the eualuation metrics of the model.

from sklearn.metrics import r2_score

r2_score_elasticNetReg_train = r2_score(y_elasticNetReg_pred_train, y_train_ridge_reg)

r2_score_elasticNetReg_test = r2_score(y_elasticNetReg_pred_test, y_test_ridge_reg)

rmse_elasticNetReg = np.sqrt(mean_squared_error(y_lassoreg_pred_test, y_test_ridge_reg)**2)

print('r2_ score for train dataset for multi linear reg : ', r2_score_elasticNetReg_train)

print('r2_ score for test dataset for multi linear reg : ', r2_score_elasticNetReg_test)

print('root mean squared error for multi linear reg : ', rmse_elasticNetReg)
from sklearn.tree import DecisionTreeRegressor

## training the model



DecisionTreeReg = DecisionTreeRegressor(random_state=0)



DecisionTreeReg.fit(X_train_ridge_reg,y_train_ridge_reg)



# Use the model to predict the test dataset.

y_DecisionTreeReg_pred_test = DecisionTreeReg.predict(X_test_ridge_reg)



# Use the model to predict the train dataset.

y_DecisionTreeReg_pred_train = DecisionTreeReg.predict(X_train_ridge_reg)



# Calculate the eualuation metrics of the model.

from sklearn.metrics import r2_score

r2_score_DecisionTreeReg_train = r2_score(y_DecisionTreeReg_pred_train, y_train_ridge_reg)

r2_score_DecisionTreeReg_test = r2_score(y_DecisionTreeReg_pred_test, y_test_ridge_reg)

rmse_DecisionTreeReg = np.sqrt(mean_squared_error(y_DecisionTreeReg_pred_test, y_test_ridge_reg)**2)

print('r2_ score for train dataset for multi linear reg : ', r2_score_DecisionTreeReg_train)

print('r2_ score for test dataset for multi linear reg : ', r2_score_DecisionTreeReg_test)

print('root mean squared error for multi linear reg : ', rmse_DecisionTreeReg)
from sklearn.ensemble import RandomForestRegressor

## training the model



RandomForestReg = RandomForestRegressor(n_estimators = 1200, random_state=0)



RandomForestReg.fit(X_train_ridge_reg,y_train_ridge_reg.ravel())





# Use the model to predict the test dataset.

y_RandomForestReg_pred_test = DecisionTreeReg.predict(X_test_ridge_reg)



# Use the model to predict the train dataset.

y_RandomForestReg_pred_train = DecisionTreeReg.predict(X_train_ridge_reg)



# Calculate the eualuation metrics of the model.

from sklearn.metrics import r2_score

r2_score_RandomForestReg_train = r2_score(y_RandomForestReg_pred_train, y_train_ridge_reg)

r2_score_RandomForestReg_test = r2_score(y_RandomForestReg_pred_test, y_test_ridge_reg)

rmse_RandomForestReg = np.sqrt(mean_squared_error(y_RandomForestReg_pred_test, y_test_ridge_reg)**2)

print('r2_ score for train dataset for multi linear reg : ', r2_score_RandomForestReg_train)

print('r2_ score for test dataset for multi linear reg : ', r2_score_RandomForestReg_test)

print('root mean squared error for multi linear reg : ', rmse_RandomForestReg)
Models = [('Simple Linear Regression', r2_score_slreg_train, r2_score_slreg_test, rmse_slreg),

          ('Multiplt Linear Regression', r2_score_mlreg_train, r2_score_mlreg_test, rmse_mlreg),

          ('Ridge Regression', r2_score_ridgereg_train, r2_score_ridgereg_test, rmse_ridgereg),

          ('Lasso Regression', r2_score_lassoreg_train, r2_score_lassoreg_test, rmse_lassoreg),

          ('Elastic Net Regression', r2_score_elasticNetReg_train, r2_score_elasticNetReg_test, rmse_elasticNetReg),

          ('Decision Tree Regressor', r2_score_DecisionTreeReg_train, r2_score_DecisionTreeReg_test, rmse_DecisionTreeReg),

          ('Random Forest Regressor', r2_score_RandomForestReg_train, r2_score_RandomForestReg_test, rmse_RandomForestReg)]
predict = pd.DataFrame(data = Models, columns = ['Models', 'r2_score Training', 'r2_score Testing', 'RMSE'])

predict
f, axes = plt.subplots(3,1, figsize=(18,8))



sns.barplot(x='Models', y='r2_score Training', data = predict, ax = axes[0])

axes[0].set_xlabel('Models')

axes[0].set_ylabel('r2_score Training')

axes[0].set_ylim(0,1.0)



sns.barplot(x='Models', y='r2_score Testing', data = predict, ax = axes[1])

axes[0].set_xlabel('Models')

axes[0].set_ylabel('r2_score Testing')

axes[0].set_ylim(0,1.0)



sns.barplot(x='Models', y='RMSE', data = predict, ax = axes[2])

axes[0].set_xlabel('Models')

axes[0].set_ylabel('RMSE')

axes[0].set_ylim(0,1.0)