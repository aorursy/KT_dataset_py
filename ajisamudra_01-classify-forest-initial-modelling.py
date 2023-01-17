# Library

import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib as mpl # data visualization

import matplotlib.pyplot as plt # data visualization

import seaborn as sns # data visualization 

sns.set(style="ticks", color_codes=True)



from sklearn.linear_model import LogisticRegression # logistic regression

from sklearn.linear_model import RidgeClassifier # ridge classifier

from sklearn.ensemble import RandomForestClassifier # random forest

from sklearn.ensemble import VotingClassifier # Ensemble model

from sklearn.model_selection import StratifiedKFold # Stratified K-Fold for cross-validation

from sklearn.model_selection import KFold # Stratified K-Fold for cross-validation

from sklearn.model_selection import GridSearchCV # to perform Grid search cross-validation 

from sklearn.model_selection import cross_val_score # to calculate cross-validation score

from scipy.ndimage.interpolation import shift # shift image for data augmentation

from sklearn import metrics

import random

import time

from datetime import datetime
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
# Read file

train = pd.read_csv("/kaggle/input/learn-together/train.csv")

test = pd.read_csv("/kaggle/input/learn-together/test.csv")

sample_submission = pd.read_csv("/kaggle/input/learn-together/sample_submission.csv")



train.describe()
# See file

train.head()

# test.describe()



# The target is Cover_Type

# We already have clean dataset, all features are already in numeric type

# The dataset is often called sparse dataset

# We could drop the Id
# Create dataset



# Drop id from file

def drop_id(df):

    return df.drop(columns = ['Id'], axis = 1)



df_train = drop_id(train)

df_test = drop_id(test)



df_train.describe()



X_train = df_train.loc[:, df_train.columns != 'Cover_Type']

y_train = df_train['Cover_Type']



X_train.shape, y_train.shape
# Functions



seed = 40

skfolds = StratifiedKFold(n_splits=3, random_state=1)



# Calculate Accuracy using cross-validation

def calculate_cv_accuracy(model, X, y):

    acc = cross_val_score(model, X, y_train, cv=3, scoring = "accuracy")

    return acc



# Evaluate Accuracy models

def evaluate_accuracy(models):



    for name, model in models:

        cv_score = calculate_cv_accuracy(model, X_train, y_train)

        model.fit(X_train, y_train)

        cv_results.append(cv_score)

        names.append(name)        

        msg = "%s: CV %f (%f)" % (name, cv_score.mean(), cv_score.std())

#         msg = "%s: CV %f (%f) | TEST %f " % (name, cv_score.mean(), cv_score.std(), test_score)

        print(msg)



# Create binary target

def create_binary_target(y, y_1, y_2, y_3, y_4, y_5, y_6, y_7):

    y_1 = (y == 1)

    y_2 = (y == 2)

    y_3 = (y == 3)

    y_4 = (y == 4)

    y_5 = (y == 5)

    y_6 = (y == 6)

    y_7 = (y == 7)   

    return y_1, y_2, y_3, y_4, y_5, y_6, y_7
# See proportion number of target

df_train['Cover_Type'].value_counts()



# The target value is balance between class
# See pair distribution between features

# Exclude SoilType



exclude_soiltype = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',

                    'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',

                    'Cover_Type']

df_train[exclude_soiltype].describe()
# Pairplot

sns.pairplot(data= df_train[exclude_soiltype] , hue="Cover_Type", diag_kind="kde")

# Correlation

cor_mtx = df_train.corr()

plt.subplots()

sns.heatmap(cor_mtx)



print(cor_mtx['Cover_Type'].sort_values(ascending=False))
# Modelling using multi-class classifier



# Multiclass Classifier

# I will be using three Multiclass Classifier i.e. (1) Logistic Regression, (2) Ridge Classifier, and (3) Random Forest

# Define Multiclass Classifier



# List of models

logistic_multi = LogisticRegression(multi_class='multinomial', solver = 'saga', penalty = 'l1', random_state = seed) # define as multiclass classifier

ridge_multi = RidgeClassifier(alpha = 0.7, random_state = seed)

rf_multi = RandomForestClassifier(n_estimators = 10, random_state = seed )

rf_multi100 = RandomForestClassifier(n_estimators = 10, random_state = seed )

rf_multi200 = RandomForestClassifier(n_estimators = 10, random_state = seed )

rf_multi500 = RandomForestClassifier(n_estimators = 10, random_state = seed )

rf_multi800 = RandomForestClassifier(n_estimators = 10, random_state = seed )



# prepare models

models = []

models.append(('logistic regression - multiclass', logistic_multi))

models.append(('ridge - multiclass', ridge_multi))

models.append(('random forest 10 - multiclass', rf_multi))

models.append(('random forest 100 - multiclass', rf_multi100))

models.append(('random forest 200 - multiclass', rf_multi200))

models.append(('random forest 500 - multiclass', rf_multi500))

models.append(('random forest 800 - multiclass', rf_multi800))

# Evaluate multi-class models

cv_results = []

names = []

evaluate_accuracy(models)



# From this, we can see random forest perform better than others.

# Lets try multi binary-classifier
# Create binary target



y_train1, y_train2, y_train3, y_train4, y_train5, y_train6, y_train7 = y_train, y_train, y_train, y_train, y_train, y_train, y_train

y_train1, y_train2, y_train3, y_train4, y_train5, y_train6, y_train7 = create_binary_target(y_train, y_train1, y_train2, y_train3, y_train4, y_train5, y_train6, y_train7)



y_train1.shape, y_train2.shape, y_train3.shape, y_train4.shape, y_train5.shape, y_train6.shape, y_train7.shape
# Lets define Binary Classifier for each Cover Type



rf1 = RandomForestClassifier(n_estimators = 10, random_state = seed)

rf2 = RandomForestClassifier(n_estimators = 10, random_state = seed)

rf3 = RandomForestClassifier(n_estimators = 10, random_state = seed)

rf4 = RandomForestClassifier(n_estimators = 10, random_state = seed)

rf5 = RandomForestClassifier(n_estimators = 10, random_state = seed)

rf6 = RandomForestClassifier(n_estimators = 10, random_state = seed)

rf7 = RandomForestClassifier(n_estimators = 10, random_state = seed)



# 0.783598 - 300 n_estimators

# 0.783796 - 200 n_estimators

# 0.781878 - 100 n_estimators



# prepare models

binary_models = []

binary_models.append(('random forest class 1', rf1))

binary_models.append(('random forest class 2', rf2))

binary_models.append(('random forest class 3', rf3))

binary_models.append(('random forest class 4', rf4))

binary_models.append(('random forest class 5', rf5))

binary_models.append(('random forest class 6', rf6))

binary_models.append(('random forest class 7', rf7))
# Evaluate binary-class models

cv_results = []

names = []

evaluate_accuracy(binary_models)
# Vote Classifier

# Using vote classifier as method to combine binary classifier

# Soft Voting will be used as we want the average class probabilities from each estimator



voting1 = VotingClassifier(estimators = binary_models, voting = 'soft')

voting2 = VotingClassifier(estimators = binary_models, voting = 'hard')



voting_models = []

voting_models.append(('voting classifier - soft', voting1))

voting_models.append(('voting classifier - hard', voting2))
# Evaluate ensemble models

evaluate_accuracy(voting_models)



# Finding

# Performance of several models we tried had the same performance

# For the sake of simplicity we will use random forest multi-class using n_estimators = 500
# Submission 1

rf_multi500 = RandomForestClassifier(n_estimators = 500, random_state = seed )

rf_multi500.fit(X_train, y_train)
# Create submission 1 file

y_pred = rf_multi500.predict(df_test)
submission1 = sample_submission

submission1.iloc[:,1] = (y_pred)

submission1.to_csv("submission1.csv", index=False)