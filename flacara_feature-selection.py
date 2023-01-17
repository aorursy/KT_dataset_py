import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn import datasets

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import ExtraTreesClassifier

from sklearn import tree

#from sklearn.cross_validation import KFold, cross_val_score

#from sklearn.cross_validation import train_test_split

#matplotlib inline



path = '../input/adult.csv'

data = pd.read_csv(path)



# remove rows where occupation is unknown

data = data[data.occupation != '?']

raw_data = data[data.occupation != '?']
# create numerical columns representing the categorical data

data['workclass_num'] = data.workclass.map({'Private':0, 'State-gov':1, 'Federal-gov':2, 'Self-emp-not-inc':3, 'Self-emp-inc':4, 'Local-gov':5, 'Without-pay':6})

data['over50K'] = np.where(data.income == '<=50K', 0, 1)

data['marital_num'] = data['marital.status'].map({'Widowed':0, 'Divorced':1, 'Separated':2, 'Never-married':3, 'Married-civ-spouse':4, 'Married-AF-spouse':4, 'Married-spouse-absent':5})

data['race_num'] = data.race.map({'White':0, 'Black':1, 'Asian-Pac-Islander':2, 'Amer-Indian-Eskimo':3, 'Other':4})

data['sex_num'] = np.where(data.sex == 'Female', 0, 1)

data['rel_num'] = data.relationship.map({'Not-in-family':0, 'Unmarried':0, 'Own-child':0, 'Other-relative':0, 'Husband':1, 'Wife':1})

data.head()
X = data[['workclass_num', 'education.num', 'marital_num', 'race_num', 'sex_num', 'rel_num', 'capital.gain', 'capital.loss']]

y = data.over50K
# create a base classifier used to evaluate a subset of attributes

logreg = LogisticRegression()



# create the RFE model and select 3 attributes

rfe = RFE(logreg, 3)

rfe = rfe.fit(X, y)



# summarize the selection of the attributes

print(rfe.support_)

print(rfe.ranking_)
# fit an Extra Tree model to the data

extree = ExtraTreesClassifier()

extree.fit(X, y)



# display the relative importance of each attribute

relval = extree.feature_importances_



#from sklearn.datasets import load_iris

#from sklearn.feature_selection import SelectFromModel

#iris = load_iris()

#X, y = iris.data, iris.target

#X.shape

#(150, 4)

#clf = ExtraTreesClassifier()

#clf = clf.fit(X, y)

#clf.feature_importances_  

#array([ 0.04...,  0.05...,  0.4...,  0.4...])

#model = SelectFromModel(clf, prefit=True)

#X_new = model.transform(X)

#X_new.shape               

#(150, 2)



# horizontal bar plot of feature importance

pos = np.arange(8) + 0.5

plt.barh(pos, relval, align='center')

plt.title("Feature Importance")

plt.xlabel("Model Accuracy")

plt.ylabel("Features")

plt.yticks(pos, ('Working Class', 'Education', 'Marital Status', 'Race', 'Sex', 'Relationship Status', 'Capital Gain', 'Capital Loss'))

plt.grid(True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13)
# import

from sklearn.linear_model import LogisticRegression



# instantiate

logreg = LogisticRegression()



# fit

logreg.fit(X_train, y_train)



# predict

y_pred = logreg.predict(X_test)



print('LogReg %s' % metrics.accuracy_score(y_test, y_pred))
# KFolds and Cross_val_scores

kf = KFold(len(data), n_folds=10, shuffle=False)

print('KFold CrossValScore %s' % cross_val_score(logreg, X, y, cv=kf).mean())
clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

metrics.accuracy_score(y_test, y_pred)