# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Visualiztion:
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Machine Learning/Modleing:
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import os
print(os.listdir("../input"))

import warnings
# ignore warnings
warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output.
DF_data_2c = pd.read_csv("../input/column_2C_weka.csv")
DF_data_3c = pd.read_csv("../input/column_3C_weka.csv")

print('Preview of 2 Category data:')
print(DF_data_2c.shape)
print(DF_data_2c.keys())
print(DF_data_2c.dtypes)

print('\n Preview of 3 Category data:')
print(DF_data_3c.shape)
print(DF_data_3c.keys())
print(DF_data_3c.dtypes)
DF_data_3c.head()
DF_data_3c.describe()
print(DF_data_3c['class'].value_counts())
sns.countplot(DF_data_3c['class']);
vars = DF_data_3c.keys().drop('class')

# Here we use a simple for loop to quickly create subplot boxplots of each variable.
plt.figure(figsize=(20,10))
for idx, var in enumerate(vars):
    plt.subplot(2,3,idx+1)
    sns.boxplot(x='class', y=var, data=DF_data_3c)

# Alternatively, we can visualize the data using violin plots...
plt.figure(figsize=(20,10))
for idx, var in enumerate(vars):
    plt.subplot(2,3,idx+1)
    sns.violinplot(x='class', y=var, data=DF_data_3c)
# seaborn has an awesome tool (pairplot) to do this very easily:
g = sns.pairplot(DF_data_3c, hue='class', height=4)
# g.map_upper(sns.regplot) # some plot options: 'regplot', 'residplot', 'scatterplot'
# g.map_lower(sns.kdeplot)
#g.map_diag(plt.hist)
# Create X (independant vars) and y (dependant var) 
X = DF_data_3c.copy().drop(['class'], axis=1)
y = DF_data_3c["class"].copy()

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, test_size = 0.20)

print(train_X.shape)
print(val_X.shape)
DTC_model = DecisionTreeClassifier()
DTC_model.fit(train_X,train_y)

# Make PredicitonsL:
DTC_predictions = DTC_model.predict(val_X)

#Print accuracy Results for DTR model
DTC_accuracy =  DTC_model.score(val_X, val_y)
print("Accuracy score for Decision Tree Classifier Model : " + str(DTC_accuracy))

print('\nVariable Importance:')
for idx, var in enumerate(vars):
    print(var, ':', str(DTC_model.feature_importances_[idx]))
RF_model = RandomForestClassifier(random_state=1)
RF_model.fit(train_X, train_y)

# make predictions
RF_predictions = RF_model.predict(val_X)

# Print Accuracy for initial RF model
RF_accuracy = RF_model.score(val_X, val_y)
print("Accuracy score for Random Forest Model : " + str(RF_accuracy))

print('\nVariable Importance:')
for idx, var in enumerate(vars):
    print(var, ':', str(RF_model.feature_importances_[idx]))
XGBC_model = XGBClassifier(random_state=1)
XGBC_model.fit(train_X, train_y)

# make predictions
XGBC_predictions = XGBC_model.predict(val_X)

# Print Accuracy for initial RF model
XGBC_accuracy = accuracy_score(val_y, XGBC_predictions)
print("Accuracy score for XGBoost Classifier model : " + str(XGBC_accuracy))

print('\nVariable Importance:')
for idx, var in enumerate(vars):
    print(var, ':', str(XGBC_model.feature_importances_[idx]))
%%time
# Slightly Tuned XGB Model:
XGBC_model = XGBClassifier(random_state=1, objective = 'multi:softprob', num_class=3) # 

parameters = {'learning_rate': [0.01, 0.015, 0.02, 0.025], # also called `eta` value
              'max_depth': [2, 3, 4, 5],
              'min_child_weight': [0.75, 1.0, 1.25, 2, 5],
              'n_estimators': [100, 150, 200, 250, 300, 500]}

XGBC_grid = GridSearchCV(XGBC_model,
                        parameters,
                        cv = 3,
                        n_jobs = 5,
                        verbose=True)

XGBC_grid.fit(train_X, train_y)

#print(XGBC_grid.best_score_)
print(XGBC_grid.best_params_)

# make predictions
XGBC_grid_predictions = XGBC_grid.predict(val_X)
# Print MAE for initial XGB model
XGBC_grid_accuracy = accuracy_score(XGBC_grid_predictions, val_y)
print("Accuracy Score for Tuned XGBoost Classifier Model : " + str(XGBC_grid_accuracy))

print('\nVariable Importance:')
for idx, var in enumerate(vars):
    print(var, ':', str(XGBC_grid.best_estimator_.feature_importances_[idx]))