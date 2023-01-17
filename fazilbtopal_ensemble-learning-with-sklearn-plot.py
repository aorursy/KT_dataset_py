import numpy as np 

import pandas as pd

import scikitplot as skplt

import matplotlib.pyplot as plt

from sklearn.metrics import *

import seaborn as sns

plt.style.use('ggplot')



import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head()
df_train.info()
df_train.describe()
# Check for null values

df_train.isnull().any(axis=None)
ax = sns.countplot(y=df_train['Activity'])

ax.set_title('Distribution of Labels')

plt.show()
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

le.fit(df_train['Activity'])

df_train['Activity'] = le.transform(df_train['Activity'])

df_test['Activity'] = le.transform(df_test['Activity'])

le.classes_
X = df_train.drop(['Activity', 'subject'], axis=1)

y = df_train['Activity']



X_test = df_test.drop(['Activity', 'subject'], axis=1)

y_test = df_test['Activity']
from sklearn.ensemble import BaggingClassifier



oob_list = list()

# Because the algorithm is so slow, we use just 4 different trees to see the outcomes.

tree_list = [20, 40, 50, 100] 



for n_trees in tree_list:

    BC = BaggingClassifier(n_estimators=n_trees, oob_score=True, random_state=42, n_jobs=-1)

    BC.fit(X, y)

    oob_error = 1 - BC.oob_score_   # Get the oob error

    oob_list.append(pd.Series({'n_trees': n_trees, 'oob': oob_error}))



err_bag = pd.concat(oob_list, axis=1).T.set_index('n_trees')
# Plot the result

ax = err_bag.plot(legend=False, marker='o')

ax.set_ylabel('out-of-bag error')

ax.set_title('OOB Error with Bagging Classifier')

plt.show()
# Bagging Classifier with 50 estimators

model = BaggingClassifier(n_estimators=50, oob_score=True, random_state=42, n_jobs=-1)

model = model.fit(X, y)

y_pred = model.predict(X_test)



print(classification_report(y_test, y_pred))
# Plot confusion Matrix

skplt.metrics.plot_confusion_matrix(y_test, y_pred, figsize=(10, 8))

plt.show()
y_probas = model.predict_proba(X_test)

skplt.metrics.plot_roc(y_test, y_probas, figsize=(10, 8))   # Plot ROC Curve

plt.show()
from sklearn.ensemble import RandomForestClassifier



RF = RandomForestClassifier(oob_score=True, random_state=42, warm_start=True, n_jobs=-1)



oob_list = list()

tree_list = [15, 20, 30, 40, 50, 100, 150, 200, 300, 400]



for n_trees in tree_list:

    RF.set_params(n_estimators=n_trees)

    RF.fit(X, y)

    oob_error = 1 - RF.oob_score_

    oob_list.append(pd.Series({'n_trees': n_trees, 'oob': oob_error}))

    

err_rf = pd.concat(oob_list, axis=1).T.set_index('n_trees')
# Plot the result

ax = err_rf.plot(legend=False, marker='o')

ax.set_ylabel('out-of-bag error')

ax.set_title('OOB Error with Random Forest')

plt.show()
# Random Forest with 100 estimators

model = RF.set_params(n_estimators=100)

y_pred = model.predict(X_test)



print(classification_report(y_test, y_pred))
# Plot confusion Matrix

skplt.metrics.plot_confusion_matrix(y_test, y_pred, figsize=(10, 8))

plt.show()
y_probas = model.predict_proba(X_test)

skplt.metrics.plot_roc(y_test, y_probas, figsize=(10, 8))   # Plot ROC Curve

plt.show()
from sklearn.ensemble import ExtraTreesClassifier



ET = ExtraTreesClassifier(oob_score=True, bootstrap=True, random_state=42, warm_start=True, n_jobs=-1)



oob_list = list()

tree_list = [15, 20, 30, 40, 50, 100, 150, 200, 300, 400]



for n_trees in tree_list:

    ET.set_params(n_estimators=n_trees)

    ET.fit(X, y)

    oob_error = 1 - ET.oob_score_

    oob_list.append(pd.Series({'n_trees': n_trees, 'oob': oob_error}))

    

err_et = pd.concat(oob_list, axis=1).T.set_index('n_trees')
# Plot the result

ax = err_et.plot(legend=False, marker='o')

ax.set_ylabel('out-of-bag error')

ax.set_title('OOB Error with Extra Trees')

plt.show()
# Extra Trees with 100 estimators

model = ET.set_params(n_estimators=100)

y_pred = model.predict(X_test)



print(classification_report(y_test, y_pred))
# Plot confusion Matrix

skplt.metrics.plot_confusion_matrix(y_test, y_pred, figsize=(10, 8))

plt.show()
y_probas = model.predict_proba(X_test)

skplt.metrics.plot_roc(y_test, y_probas, figsize=(10, 8))   # Plot ROC Curve

plt.show()
from sklearn.ensemble import GradientBoostingClassifier



error_list = list()



# Iterate through all of the possibilities for number of estimators

tree_list = [15, 50, 100, 200, 400]

for n_trees in tree_list:

    GBC = GradientBoostingClassifier(n_estimators=n_trees, subsample=0.5,

                                     max_features=4, random_state=42)

    GBC.fit(X, y)

    y_pred = GBC.predict(X_test)



    # Get the error

    error = 1. - accuracy_score(y_test, y_pred)

    error_list.append(pd.Series({'n_trees': n_trees, 'error': error}))



err_gbc = pd.concat(error_list, axis=1).T.set_index('n_trees')
# Plot the result

ax = err_gbc.plot(legend=False, marker='o')

ax.set_ylabel('deviance error')

ax.set_title('Error with Gradient Boosting')

plt.show()
# Extra Trees with 100 estimators

model = GradientBoostingClassifier(n_estimators=100, subsample=0.5,

                                     max_features=4, random_state=42)

model.fit(X, y)

y_pred = model.predict(X_test)



print(classification_report(y_test, y_pred))
# Plot confusion Matrix

skplt.metrics.plot_confusion_matrix(y_test, y_pred, figsize=(10, 8))

plt.show()
y_probas = model.predict_proba(X_test)

skplt.metrics.plot_roc(y_test, y_probas, figsize=(10, 8))   # Plot ROC Curve

plt.show()
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier



error_list = list()



# Iterate through all of the possibilities for number of estimators

tree_list = [15, 50, 100, 200, 400]



# Setting max_features=4 in the decision tree classifier used as the base classifier 

# for AdaBoost will increase the convergence rate

base = DecisionTreeClassifier(max_features=4)

for n_trees in tree_list:

    ABC = AdaBoostClassifier(base_estimator=base, n_estimators=n_trees, 

                             learning_rate=0.1, random_state=42)

    ABC.fit(X, y)

    y_pred = ABC.predict(X_test)



    # Get the error

    error = 1. - accuracy_score(y_test, y_pred)

    error_list.append(pd.Series({'n_trees': n_trees, 'error': error}))



err_abc = pd.concat(error_list, axis=1).T.set_index('n_trees')
# Plot the result

ax = err_abc.plot(legend=False, marker='o')

ax.set_ylabel('adaptive boosting error')

ax.set_title('Error with Adaptive Boosting')

plt.show()
model = AdaBoostClassifier(base_estimator=base, n_estimators=50, 

                             learning_rate=0.1, random_state=42)

model.fit(X, y)

y_pred = model.predict(X_test)



print(classification_report(y_test, y_pred))
# Plot confusion Matrix

skplt.metrics.plot_confusion_matrix(y_test, y_pred, figsize=(10, 8))

plt.show()
y_probas = model.predict_proba(X_test)

skplt.metrics.plot_roc(y_test, y_probas, figsize=(10, 8))   # Plot ROC Curve

plt.show()
from sklearn.linear_model import LogisticRegressionCV



# L2 regularized logistic regression

LR = LogisticRegressionCV(Cs=5, cv=4, penalty='l2', max_iter=1000)

LR.fit(X, y)
y_pred = LR.predict(X_test)

print(classification_report(y_pred, y_test))
# Plot confusion Matrix

skplt.metrics.plot_confusion_matrix(y_test, y_pred, figsize=(10, 8))

plt.show()
y_probas = LR.predict_proba(X_test)

skplt.metrics.plot_roc(y_test, y_probas, figsize=(10, 8))   # Plot ROC Curve

plt.show()
from sklearn.ensemble import VotingClassifier



# The combined model--logistic regression and gradient boosted trees

estimators = [('LR', LR), ('GBC', GBC)]



# Though it wasn't done here, it is often desirable to train 

# this model using an additional hold-out data set and/or with cross validation

VC = VotingClassifier(estimators, voting='soft', n_jobs=-1)

VC.fit(X, y)
y_pred = VC.predict(X_test)

print(classification_report(y_test, y_pred))
# Plot confusion Matrix

skplt.metrics.plot_confusion_matrix(y_test, y_pred, figsize=(10, 8))

plt.show()
y_probas = VC.predict_proba(X_test)

skplt.metrics.plot_roc(y_test, y_probas, figsize=(10, 8))   # Plot ROC Curve

plt.show()