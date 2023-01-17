# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import seaborn as sns
from plotnine import *
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/projects.csv')
df.shape
df.head()
df.info()
df['case_type'].value_counts()
(df['case_type'].value_counts() / len(df)).plot.bar()
df['number_pages'].value_counts().sort_index().plot.line()
df['3d_modeling'].value_counts()
(df['3d_modeling'].value_counts() / len(df)).plot.bar()
df['hours'].value_counts().sort_index().plot.line()
type_pages = pd.crosstab(index = df['number_pages'],
                        columns = df['case_type'])
type_pages
type_pages.plot(kind = 'bar',
               figsize = (8,8),
               stacked=True)
type_modeling = pd.crosstab(index = df['3d_modeling'],
                        columns = df['case_type'])
type_modeling
type_modeling.plot(kind = 'bar',
               figsize = (8,8),
               stacked=True)
df.boxplot(column="hours",
           by= "case_type",
           figsize= (8,8))
pages_modeling = pd.crosstab(index = df['number_pages'],
                        columns = df['3d_modeling'])
pages_modeling
pages_modeling.plot(kind = 'bar',
               figsize = (8,8),
               stacked=True)
df.boxplot(column="hours",
           by= "number_pages",
           figsize= (8,8))
df.boxplot(column="hours",
           by= "3d_modeling",
           figsize= (8,8))
df.columns
# Create a new variable that records 'hours' / 'number_pages'
df['hour_page'] = df['hours'] / df['number_pages']
df.hour_page.describe()
# Create variables to store location of bin boundaries
hp_min = df.hour_page.min()
hp_max = df.hour_page.max()
hp_range = hp_max - hp_min
hp_bin = hp_range / 4


# Create variables to store location of difficulty bins
level_one = hp_min + hp_bin
level_two = level_one + hp_bin
level_three = level_two + hp_bin
# Create a function that will assign a difficulty to each project
# based on 'hour_page'

def get_difficulty(row):
    difficulty = 0
    if row.hour_page < level_one:
        difficulty = 1
    elif (row.hour_page >= level_one) & (row.hour_page < level_two):
        difficulty = 2
    elif (row.hour_page >= level_two) & (row.hour_page < level_three):
        difficulty = 3
    elif (row.hour_page >= level_three):
        difficulty = 4
    else:
        return difficulty
    
    return difficulty
df['difficulty'] = df.apply(get_difficulty, axis=1)
df.difficulty.value_counts()
df.head()
# First I'll import some libraries I know I'll need 
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
# I'll make a copy of the dataset so I can refer back to it
df_train = df.copy()
# Delete unneeded variables
del df_train['project_id']
del df_train['hour_page']

print('Data shape:', df_train.shape)
# Use label encoder on the 'case_type' and '3d_modeling' variables
labelencoder_X = LabelEncoder()

df_train['case_type'] = labelencoder_X.fit_transform(df_train['case_type'])
df_train['3d_modeling'] = labelencoder_X.fit_transform(df_train['3d_modeling'])
# utility = 1, yes = 1
df_train.head()
# Create X and y arrays for the dataset
X = df_train[['case_type', 'number_pages', '3d_modeling', 'difficulty']].copy()
y = df_train['hours'].values
y.shape
y = y.reshape(-1, 1)
y.shape
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print('Training data shape: {}'. format(len(X_train)))
print('Test data shape: {}'. format(len(X_test)))
# PIPELINE
my_pipeline = make_pipeline(StandardScaler(), SVR(kernel = 'linear'))

my_pipeline.fit(X_train, y_train)
y_pred = my_pipeline.predict(X_test)
svr_pipeline_score = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error for SVR: {}'.format(svr_pipeline_score))
# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)


regressor = SVR(kernel = 'linear')



# Applying Grid Search to find the best model and the best parameters
parameters = [{'C': [1, 5, 10, 15, 20], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = 'neg_mean_absolute_error',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

print('Best Accuracy: {}'.format(best_accuracy))
print('Best Parameters: {}'.format(best_parameters))
# Create X and y arrays for the dataset
X = df_train[['case_type', 'number_pages', '3d_modeling', 'difficulty']].copy()
y = df_train['hours'].values
y = y.reshape(-1, 1)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# PIPELINE
my_pipeline = make_pipeline(StandardScaler(), SVR(kernel = 'linear', C = 10))

my_pipeline.fit(X_train, y_train)
y_pred = my_pipeline.predict(X_test)
# Print out the accuracy score
svr_updated_pipeline_score = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error for SVR: {}'.format(svr_pipeline_score))
# Fitting XGBoost to the Training set
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# get predicted 'hours' on validation data
xg_score = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error for XGBoost: {}'.format(xg_score))
