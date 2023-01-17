import numpy as np 

import pandas as pd 

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.datasets import load_breast_cancer

from sklearn.datasets import load_digits

from sklearn import metrics

%matplotlib inline



import os

import warnings

warnings.filterwarnings('ignore')

cancer = load_breast_cancer()

digits = load_digits()



data = cancer
df = pd.DataFrame(data= np.c_[data['data'], data['target']],

                     columns= list(data['feature_names']) + ['target'])

df['target'] = df['target'].astype('uint16')
df
df.head()
# adaboost experiments

# create x and y train

X = df.drop('target', axis=1)

y = df[['target']]



# split data into train and test/validation sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
# check the average cancer occurence rates in train and test data, should be comparable

print(y_train.mean())

print(y_test.mean())
# base estimator: a weak learner with max_depth=2

shallow_tree = DecisionTreeClassifier(max_depth=2, random_state = 100)
# fit the shallow decision tree 

shallow_tree.fit(X_train, y_train)



# test error

y_pred = shallow_tree.predict(X_test)

score = metrics.accuracy_score(y_test, y_pred)

score
# adaboost with the tree as base estimator



estimators = list(range(1, 50, 3))



abc_scores = []

for n_est in estimators:

    ABC = AdaBoostClassifier(

    base_estimator=shallow_tree, 

    n_estimators = n_est)

    

    ABC.fit(X_train, y_train)

    y_pred = ABC.predict(X_test)

    score = metrics.accuracy_score(y_test, y_pred)

    abc_scores.append(score)

    
abc_scores
# plot test scores and n_estimators

# plot

plt.plot(estimators, abc_scores)

plt.xlabel('n_estimators')

plt.ylabel('accuracy')

plt.ylim([0.85, 1])

plt.show()