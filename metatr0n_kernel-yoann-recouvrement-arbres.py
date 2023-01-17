# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection  import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



train_df.info()

print('_'*40)

test_df.info()

print('_'*40)

print(train_df.columns.values)

print('_'*40)

train_df.describe()

train_df.head()
train_df = train_df.drop(['Id'], axis=1)



train_df.shape, test_df.shape



X_trainTotal = train_df.drop("Cover_Type", axis=1)

Y_trainTotal = train_df["Cover_Type"]

X_test = test_df.drop("Id", axis=1).copy()



X_train, X_val, Y_train, Y_val = train_test_split(X_trainTotal, Y_trainTotal, random_state = 0)
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
# on drop les colonnes



cols = ['Hillshade_3pm', 'Aspect', 'Vertical_Distance_To_Hydrology', 'Soil_Type8', 'Soil_Type16', 'Hillshade_9am', 'Wilderness_Area1', 'Slope', 'Horizontal_Distance_To_Roadways', 'Soil_Type9', 'Soil_Type26', 'Soil_Type29', 'Soil_Type10', 'Horizontal_Distance_To_Fire_Points', 'Soil_Type6', 'Soil_Type15']

print(X_train.columns.size)

X_train = X_train.drop(cols, axis=1)

X_val = X_val.drop(cols, axis=1)

X_test = X_test.drop(cols, axis=1)

X_trainTotal = X_trainTotal.drop(cols, axis=1)

print(X_trainTotal.columns.size)



# Random Forest



def get_rf_score(n_estimators_, train_X, val_X, train_y, val_y):

    random_forest = RandomForestClassifier(n_estimators=100,max_leaf_nodes=5000)

    random_forest.fit(train_X, train_y)

    acc_random_forest = round(random_forest.score(val_X, val_y) * 100, 2)

    return(acc_random_forest)



for n_estimators in [5, 10, 50, 100, 200, 300, 400, 500]:

    my_score = get_rf_score(n_estimators, X_train, X_val, Y_train, Y_val)

    print("n_estimators: %d  \t\t Mean Absolute Error:  %d" %(n_estimators, my_score))



# XGBoost



from xgboost import XGBRegressor



my_model = XGBRegressor()

my_model.fit(X_train, Y_train, verbose=True)

acc_xgb = round(my_model.score(X_val, Y_val) * 100, 2)

acc_xgb
# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_val)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn



# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian



# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron



# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc



# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd



# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree



# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_trainTotal, Y_trainTotal)

Y_pred = random_forest.predict(X_test)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

        "Id": test_df["Id"],

        "Cover_Type": Y_pred

    })

submission.to_csv('submission.csv', index=False)