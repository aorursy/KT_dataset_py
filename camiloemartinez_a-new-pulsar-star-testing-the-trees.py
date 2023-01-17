import numpy as np                # linear algebra

import pandas as pd               # data frames

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

import scipy.stats                # statistics

import time

from sklearn.preprocessing import power_transform

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint

from scipy.stats import uniform



import os

print(os.listdir("../input"))
df = pd.read_csv("../input/pulsar_stars.csv")

df_scale = df.copy()

columns =df.columns[:-1]

df_scale[columns] = power_transform(df.iloc[:,0:8],method='yeo-johnson')

df_scale.head()
# Create feature and target arrays

X = df_scale.iloc[:,0:8]

y = df_scale.iloc[:,-1]



# Split into training and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=123, stratify=y)

print('X_train:', X_train.shape)

print('X_test:', X_test.shape)
# Import necessary modules

from sklearn.tree import DecisionTreeClassifier



# Setup the parameters and distributions to sample from: param_dist

param_dist = {"max_depth": [3, None],

              "max_features": randint(1, 9),

              "min_samples_leaf": randint(1, 9),

              "criterion": ["gini", "entropy"]}



# Instantiate a Decision Tree classifier: tree

single = DecisionTreeClassifier()



# Instantiate the Grid Search

single_cv = RandomizedSearchCV(single, param_dist, cv=5, scoring='accuracy')



# Fit it to the data

start = time.time()

single_cv.fit(X_train, y_train)

single_t = time.time() - start





# Print the tuned parameters and score

print("Tuned Decision Tree Parameters: {}".format(single_cv.best_params_))

print("Best score is {}".format(single_cv.best_score_))

print("Run in (s) {}".format(single_t) )
# Import necessary modules

from sklearn.ensemble import RandomForestClassifier



# Setup the parameters and distributions to sample from: param_dist

param_dist = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 500, num = 10)],

              "max_depth": [3, None],

              "max_features": ['auto', 'sqrt'],

              'min_samples_split': randint(2, 9),

              'min_samples_leaf': randint(1, 9),

              'bootstrap': [True, False]}



# Instantiate a Random Forest Classifier

bagging_tree = RandomForestClassifier()



# Instantiate the Grid Search

bagging_cv = RandomizedSearchCV(bagging_tree, param_dist, cv=5, scoring='accuracy')



# Fit it to the data

start = time.time()

bagging_cv.fit(X_train, y_train)

bagging_t = time.time() - start





# Print the tuned parameters and score

print("Tuned Decision Tree Parameters: {}".format(bagging_cv.best_params_))

print("Best score is {}".format(bagging_cv.best_score_))

print("Run in (s) {}".format(bagging_t) )
# Import necessary modules

import xgboost as xgb



# Setup the parameters and distributions to sample from: param_dist

param_dist = {'n_estimators': randint(100, 300),

              'learning_rate': uniform(0.01, 0.6),

              'subsample': [0.3, 0.9],

              'max_depth': [3, 4, 5, 6, 7, 8, 9],

              'colsample_bytree': [0.5, 0.8],

              'min_child_weight': [1, 2, 3, 4]

             }



# Instantiate a XGBoost Classifier

boosting_tree = xgb.XGBClassifier(objective = 'binary:logistic')



# Instantiate the Grid Search

boosting_cv = RandomizedSearchCV(boosting_tree, param_dist, cv=5, scoring='accuracy')



# Fit it to the data

start = time.time()

boosting_cv.fit(X_train, y_train)

boosting_t = time.time() - start



# Print the tuned parameters and score

print("Tuned Decision Tree Parameters: {}".format(boosting_cv.best_params_))

print("Best score is {}".format(boosting_cv.best_score_))

print("Run in (s) {}".format(boosting_t) )
results = {'Algorithm': ['single', 'bagging', 'boosting'],

           'Acc': [single_cv.best_score_, bagging_cv.best_score_, boosting_cv.best_score_],

           'time': [single_t, bagging_t, boosting_t]}

df = pd.DataFrame.from_dict(results)



sns.scatterplot(x="time", y="Acc",

                     hue="Algorithm", sizes=(10, 200),

                     data=df)