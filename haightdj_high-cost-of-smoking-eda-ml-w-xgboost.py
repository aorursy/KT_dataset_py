# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output.
DF_data = pd.read_csv('../input/insurance.csv')
print(DF_data.shape)
print(DF_data.keys())
print(DF_data.dtypes)
DF_data.head() # preview top 5 rows
DF_data.describe() # display some very brief stats on numeric data
print('Missing Training Data:')
DF_data.isnull().sum() # count number of missing frames for each column
plt.figure(figsize=(10,6))
ax = sns.distplot(DF_data['charges'])
ax.set_title('Distribution of Medical Charges');
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
sns.countplot(DF_data.sex);
plt.subplot(1,3,2)
sns.countplot(DF_data.smoker);
plt.subplot(1,3,3)
sns.countplot(DF_data.region);
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
sns.distplot(DF_data.age);
plt.subplot(1,3,2)
sns.distplot(DF_data.bmi);
plt.subplot(1,3,3)
sns.distplot(DF_data.children);
# First See if charges differ within the categorical variables...
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.boxplot(x='sex',y='charges',data=DF_data)
plt.subplot(1,3,2)
sns.boxplot(x='smoker',y='charges',data=DF_data)
plt.subplot(1,3,3)
sns.boxplot(x='region',y='charges',data=DF_data);
# Next let's try the non-categorical data:
# sns.jointplot("age", "charges", data=DF_data, kind="reg");
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.regplot(x='age',y='charges',data=DF_data)
plt.subplot(1,3,2)
sns.regplot(x='bmi',y='charges',data=DF_data)
plt.subplot(1,3,3)
sns.regplot(x='children',y='charges',data=DF_data);
g = sns.pairplot(DF_data, hue='smoker', height=4)
# For sex... Lets Change 'Female' to 0 and 'Male' to 1
DF_data.loc[DF_data['sex'] == 'male', 'sex'] = 0
DF_data.loc[DF_data['sex'] == 'female', 'sex'] = 1

# For smoker... Lets Change 'no' to 0 and 'yes' to 1
DF_data.loc[DF_data['smoker'] == 'no', 'smoker'] = 0
DF_data.loc[DF_data['smoker'] == 'yes', 'smoker'] = 1

# For region... Lets Change to 1:4
DF_data.loc[DF_data['region'] == 'southwest', 'region'] = 0
DF_data.loc[DF_data['region'] == 'southeast', 'region'] = 1
DF_data.loc[DF_data['region'] == 'northwest', 'region'] = 2
DF_data.loc[DF_data['region'] == 'northeast', 'region'] = 3
# DF_data.head()
DF_data.head()
# Add weight classifications based on BMI
# underweight <18 , normal = 18-25, overweight = 25-30, obese= >30
# seems to mildly help RF model; no effect on XGB model

DF_data.loc[DF_data['bmi'] < 18, 'weightclass'] = 0 #Underweight
DF_data.loc[(DF_data['bmi'] >= 18) & (DF_data['bmi'] < 25), 'weightclass'] = 1 # Normal Weight
DF_data.loc[(DF_data['bmi'] >= 25) & (DF_data['bmi'] < 30), 'weightclass'] = 2 #overweight
DF_data.loc[DF_data['bmi'] >= 30, 'weightclass'] = 4 # Obese

DF_data.head()
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(x='weightclass',y='charges',data=DF_data);
plt.subplot(1,2,2)
sns.boxplot(x='weightclass',y='age',data=DF_data);
# create a feature called youngadult:
# seems to mildly help RF model. no effect on XGB model

DF_data.loc[DF_data['age'] < 30, 'youngadult'] = 1 
DF_data.loc[DF_data['age'] >= 30, 'youngadult'] = 0
DF_data.head()
DF_data.corr()['charges'].sort_values()
fig, (ax) = plt.subplots(1, 1, figsize=(10,6))

hm = sns.heatmap(DF_data.corr(), 
                 ax=ax, # Axes in which to draw the plot
                 cmap="coolwarm", # color-scheme
                 annot=True, 
                 fmt='.2f',       # formatting  to use when adding annotations.
                 linewidths=.05)

fig.suptitle('Health Costs Correlation Heatmap', 
              fontsize=14, 
              fontweight='bold');
# Create X, y
X = DF_data.copy().drop(['charges'], axis=1)

y = DF_data.copy().charges

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
RF_model = RandomForestRegressor(random_state=1)
RF_model.fit(train_X, train_y)

# make predictions
RF_predictions = RF_model.predict(val_X)
# Print MAE for initial XGB model
RF_mae = mean_absolute_error(RF_predictions, val_y)
print("Validation MAE for Random Forest Model : " + str(RF_mae))
# XGBoost model:
XGB_model = XGBRegressor(random_state=1)
XGB_model.fit(train_X, train_y, verbose=False)

# make predictions
XGB_predictions = XGB_model.predict(val_X)
# Print MAE for initial XGB model
XGB_mae = mean_absolute_error(XGB_predictions, val_y)
print("Validation MAE for XGBoost Model : " + str(XGB_mae))
%%time
# Slightly Tuned XGB Model:
XGB_model = XGBRegressor(random_state=1)

parameters = {'learning_rate': [0.02, 0.025, 0.05, 0.075, 0.1], #so called `eta` value
              'max_depth': [2, 3, 4, 5],
             'n_estimators': [100, 150, 200, 250, 300, 500]}

XGB_grid = GridSearchCV(XGB_model,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)

XGB_grid.fit(train_X, train_y)

print(XGB_grid.best_score_)
print(XGB_grid.best_params_)

# make predictions
XGB_grid_predictions = XGB_grid.predict(val_X)
# Print MAE for initial XGB model
XGB_grid_mae = mean_absolute_error(XGB_grid_predictions, val_y)
print("Validation MAE for grid search XGBoost Model : " + str(XGB_grid_mae))
