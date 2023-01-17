# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")

data.head()
# Checking for missing data



print(data.isnull().sum())

print('\n No missing data')
# Checking numbers of "Not Survived" (1) and "Survived" (0)



data["target"].value_counts()
# Defining variables and split data



y = data["target"]

X = data.drop(labels=["target"], axis = 1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



# Data dimension

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# pip install 'grid search'



!pip install scikit-optimize
# Create a function to train a model that will be used like parameters in 'tuning parameter method'



def treinar_modelo(params):

  max_leaf_nodes = params[0]

  n_estimators = params[1]



  rf = RandomForestClassifier(max_leaf_nodes = max_leaf_nodes, n_estimators = n_estimators)

  rf.fit(X_train, y_train)

  predict_rf = rf.predict_proba(X_test)[:,1]



  return -roc_auc_score(y_test, predict_rf)
# Import the library to tuning parameters

from skopt import dummy_minimize 



# Dummy minimize will be use to find parameters at randomly from a sample

space = [(2, 145), (50, 1000)]

resultado_random = dummy_minimize(treinar_modelo, dimensions=space, random_state=42, verbose=0)



# Best parameters

print(resultado_random.x)



# Score of the best model

print(resultado_random.fun)
# Bayesian optimization



from skopt import gp_minimize

resultados_bayesian = gp_minimize(treinar_modelo, space, n_calls=30, n_random_starts=20, random_state=42, verbose=0)



# Best parameters

print(resultados_bayesian.x)



# Score of the best model

print(resultados_bayesian.fun)
# The best model is:



best_rf = RandomForestClassifier(n_estimators=51, max_leaf_nodes=114)

best_rf.fit(X_train, y_train)
# Install library shap for understand the model

!pip install shap
# Import library

import shap



# Create the objects to understand the model

explainer = shap.TreeExplainer(best_rf)

shap_values = explainer.shap_values(X_train)
# init js

shap.initjs()



# Force plot

shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_train.iloc[0,:])
# summary plot - find the most important features



shap.initjs()

shap.summary_plot(shap_values[1], X_train)
# dependence plot - select one feature to especfic analysis



shap.initjs()

shap.dependence_plot("thal", shap_values[1], X_train, interaction_index=None)

shap.dependence_plot("thalach", shap_values[1], X_train, interaction_index=None)