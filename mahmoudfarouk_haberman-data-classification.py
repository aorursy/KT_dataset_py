#import necessary libraries

import numpy as np 

import pandas as pd

import os

import seaborn as sns

from sklearn.model_selection import train_test_split, KFold

from sklearn import cross_validation

from sklearn.metrics import accuracy_score, f1_score, r2_score, confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
# import data

names = ['age', 'op_year', 'pos_nodes', 'survived']

df = pd.read_csv('../input/haberman.csv', names=names)
df.head()
df.survived.value_counts(normalize=True)
# change the label 2 to 0, just for relief

df.survived = df.survived.apply(lambda x : 0 if x == 2 else x)
df.survived.value_counts(normalize=True)
df.describe()
# is there any nulls?

df.isnull().any().any()
sns.pairplot(df, hue='survived', diag_kind='kde')
print("The correlation Coefficient: survived -> age: ", df.survived.corr(df.age)  )

print("The correlation Coefficient: survived -> op_year: ", df.survived.corr(df.op_year)  )

print("The correlation Coefficient: survived -> pos_nodes: ", df.survived.corr(df.pos_nodes)  )
# the num of pos_nodes has a negative correlation with survived

# i suggest to remove op_year, i dont find it any relevant
X = df.drop(['survived', 'op_year'], axis=1)

y = df.survived
test_size = 0.3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
# test some classifiers

clfs = [

    ('LogisticRegression', LogisticRegression),

    ('GaussianNB', GaussianNB),

    ('SVC', SVC),

    ('DecisionTreeClassifier', DecisionTreeClassifier)

]
for name, clf in clfs:

    kf = cross_validation.KFold(n=X_train.shape[0], n_folds=3, random_state=250)

    cv_score = cross_validation.cross_val_score(estimator=clf, X=X_train, y=y_train, cv=kf, scoring='accuracy')

    print(name, "has accuracy: ", cv_score.mean())
# test LogisticRegression on test

clf = LogisticRegression()

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print('accurecy score: ', accuracy_score(pred, y_test))