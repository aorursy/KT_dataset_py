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
# Import training data - review shape and columns

import pandas as pd

train = pd.read_csv('/kaggle/input/learn-together/train.csv', index_col='Id')

print(train.info())
# Reverse the one hot encoding for Wilderness Area and Soil Type to create one column each for Wilderness Areas and Soil Types which contain all all possible category values

train_1 = train.copy()

train_1['Wild_Area'] = train_1.iloc[:,10:14].idxmax(axis=1)

train_1['Soil_Type'] = train_1.iloc[:,14:54].idxmax(axis=1)

train_1.head()

# Drop the one hot encoded columns to a get a data frame only with numeric and the categorical columns

col_to_drop = np.arange(10,54)

col_to_drop

train_1.drop(train_1.columns[col_to_drop], axis = 1, inplace = True)

train_1.dtypes
# list of non zero soil types available in training data

avail_soil = train_1.Soil_Type.value_counts().index

print(avail_soil)



# Get all possible Soil Types available as features in the original training data as columns

all_soil = train.columns[14:54]

print(all_soil)

# Check the missing soil types from training data by comparing all_soil and avail_soil

miss_soil = np.setdiff1d(all_soil,avail_soil)

print(miss_soil)
# Make a copy of train df for ML experiments

train_2 = train.copy()

train_2.drop(['Soil_Type7','Soil_Type15', 'Soil_Type8','Soil_Type25'], axis = 1, inplace=True)

train_2.columns



# Separate feature and target arrays as X and y

X = train_2.drop('Cover_Type', axis = 1)

y=train_2.Cover_Type

print(X.columns)

y[:5]



# Split X and y into Train and Validation sets

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size = 0.2, stratify = y, random_state = 99)

print(X_train.shape)

print(y_train.shape)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score



# Function to execute random forest

def rf_grid(X_train,y_train, param_grid, cv=5):

    rf = RandomForestClassifier(random_state=99)

    rf_grid = GridSearchCV(rf,param_grid, cv=5)

    rf_grid.fit(X_train,y_train)

    y_pred = rf_grid.predict(X_val)

    print(pd.DataFrame(rf_grid.cv_results_)[['params','mean_test_score']])

    print('Random Forest Best Parameters: ',rf_grid.best_params_)

    print('Random Forest Best Training Score: ',rf_grid.best_score_)

    print('Random Forest Validation Accuracy is: ', accuracy_score(y_val,y_pred))



param_grid_rf = {'n_estimators': [700,800,1000,1200]}



# Execute Random Forest

rf_grid(X_train,y_train,param_grid_rf)

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score



# Function to execute Extremely Randomized Trees

def extra_trees_grid(X_train,y_train, param_grid, cv=5):

    extra_trees = ExtraTreesClassifier(random_state=99)

    extra_trees_grid = GridSearchCV(extra_trees,param_grid, cv=5)

    extra_trees_grid.fit(X_train,y_train)

    y_pred = extra_trees_grid.predict(X_val)

    print(pd.DataFrame(extra_trees_grid.cv_results_)[['params','mean_test_score']])

    print('Extra Trees Best Parameters: ',extra_trees_grid.best_params_)

    print('Extra Trees Best Training Score: ',extra_trees_grid.best_score_)

    print('Extra Trees Validation Accuracy is: ', accuracy_score(y_val,y_pred))

    

param_grid_extra = {'n_estimators': [700,1000,1400,1700,2000]}



# Execute Extremely Randomized Trees

extra_trees_grid(X_train,y_train,param_grid_extra)



from lightgbm import LGBMClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score



# Function to execute lightGBM classifier

def lgbm_grid(X_train,y_train, param_grid, cv=5):

    lgbm = LGBMClassifier(random_state=99)

    lgbm_grid = GridSearchCV(lgbm,param_grid, cv=5)

    lgbm_grid.fit(X_train,y_train)

    y_pred = lgbm_grid.predict(X_val)

    print(pd.DataFrame(lgbm_grid.cv_results_)[['params','mean_test_score']])

    print('LightGBM Best Parameters: ',lgbm_grid.best_params_)

    print('LightGBM Best Training Score: ',lgbm_grid.best_score_)

    print('LightGBM Validation Accuracy is: ', accuracy_score(y_val,y_pred))



param_grid_lgbm = {'n_estimators': [1000,1200,1400,1600,1800]}



# Execute LightGBM classifier

lgbm_grid(X_train,y_train,param_grid_lgbm)