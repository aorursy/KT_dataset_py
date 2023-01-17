# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import gc

import os

import logging

import datetime

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import lightgbm as lgb

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
# import Dataset in Data Frame

train= pd.read_csv("../input/train.csv")

test = pd.read_csv('../input/test.csv')
train.shape

#test.shape
train.head()
train.describe()
#to check for null values

train.isnull().sum()
#sns.heatmap
f,ax = plt.subplots(figsize=(300, 300))

sns.heatmap(train.corr(), annot=True, linewidths=.8, fmt= '.3f',ax=ax, cmap="YlGnBu")

plt.show()
ax = sns.boxplot(data=train, orient="h", palette="Set2")
# sns.pairplot(train)

# sns.plt.show()

train.corr()
def performance(Y_test, logist_pred):

    logist_pred_var = [0 if i < 0.5 else 1 for i in logist_pred]

    print('Confusion Matrix:')

    print(confusion_matrix(Y_test, logist_pred_var)) 

      

    #print(classification_report(Y_test, logist_pred)) 



    fpr, tpr, thresholds = roc_curve(Y_test, logist_pred, pos_label=1)

    print('AUC:')

    print(auc(fpr, tpr))

#DECISION TREE



from sklearn.model_selection import KFold, train_test_split

from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier



# Target variable from the Training Set

Target = train['target']



# Input dataset for Train and Test 

train_inp = train.drop(columns = ['target', 'ID_code'])

test_inp = test.drop(columns = ['ID_code'])



# List of feature names

features = list(train_inp.columns)
# Spliting the Train Dataset into training and validation 



X_train, X_test, Y_train, Y_test = train_test_split(train_inp, Target, 

                                                    test_size= 0.3, random_state = 2019)

# Create Decision Tree Classifier object with few parameters

tree_clf = DecisionTreeClassifier(class_weight='balanced', random_state = 2019, 

                                  max_features = 0.7, min_samples_leaf = 80)



# Fit the object on training data

tree_clf.fit(X_train, Y_train)
# Predict for validation set and check the performance

tree_preds = tree_clf.predict_proba(X_test)[:, 1]

performance(Y_test, tree_preds)
feature_importance_values = tree_clf.feature_importances_

feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

feature_importances.sort_values(by='feature', ascending=False).head(n=20)
#LightGBM model

import lightgbm as lgb

#Function to build the LightGBM model.

def run_lgb(X_train, Y_train, X_test, Y_test, test_inp):

    params = {

        "objective" : "binary",

        "metric" : "auc",

        "num_leaves" : 1000,

        "learning_rate" : 0.01,

        "bagging_fraction" : 0.8,

        "feature_fraction" : 0.8,

        "bagging_freq" : 5,

        "reg_alpha" : 1.728910519108444,

        "reg_lambda" : 4.9847051755586085,

        "random_state" : 42,

        "bagging_seed" : 2019,

        "verbosity" : -1,

        "max_depth": 18,

        "min_child_samples":100

       # ,"boosting":"rf"

    }

    

    lgtrain = lgb.Dataset(X_train, label=Y_train)

    lgval = lgb.Dataset(X_test, label=Y_test)

    evals_result = {}

    model = lgb.train(params, lgtrain, 2500, valid_sets=[lgval], 

                      early_stopping_rounds=50, verbose_eval=50, evals_result=evals_result)

    

    pred_test_y = model.predict(test_inp, num_iteration=model.best_iteration)

    return pred_test_y, model, evals_result



# Training the model #

pred_test, model, evals_result = run_lgb(X_train, Y_train, X_test, Y_test, test_inp)
# Feature Extraction & importance

feature_importance_values = model.feature_importance()

feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

feature_importances.sort_values(by='importance', ascending=False).head(n=20)
# Submission dataframe

pred_test[pred_test>1] = 1

pred_test[pred_test<0] = 0



submitLGB = test[['ID_code']]

submitLGB["target"] = pred_test



# Creating the Submission for Light GBM

#submitLGB.to_csv('LightGBM.csv', index = False)



submitLGB