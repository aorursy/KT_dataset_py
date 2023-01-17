# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn import tree 

from IPython.display import Image

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os





X=np.zeros((800,10))

Y=np.zeros(800)



max_f = 0

h2float={}

for i,l in enumerate(open('../input/Pokemon.csv')):

    if(i==0):

        continue

    t_l = l.rstrip('\r\n').split(',')

    for j,c in enumerate(t_l[2:-1]):

        if (j==0 or j==1):

            if c not in h2float:

                max_f += 1

                h2float[c]=max_f

            X[i-1,j]=h2float[c]

        else: X[i-1,j]=float(c)

    Y[i-1]= 0. if t_l[-1]=='False' else 1.

X_train, X_test = X[:160], X[160:]

Y_train, Y_test = Y[:160], Y[160:]



#clf = clf.fit(X_train, Y_train)

#print(clf.score(X_test,Y_test))   

clf = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(clf, X, Y, cv=10)

print(scores.mean())
#solver newton-cg score = 0.925

#solver liblinear  score = 0.903



clf1 = LogisticRegression(solver='newton-cg').fit(X_train,Y_train)





scores = clf1.score(X_test,Y_test)

print(scores)
clf2 = LogisticRegression(solver='newton-cg')

scores = cross_val_score(clf2, X, Y, cv=10)

print(scores.mean())
clf3 = GaussianNB().fit(X_train,Y_train)

scores = cross_val_score(clf3, X, Y, cv=10)

print("sans cross validation : ",clf3.score(X_test,Y_test),"\ncross validation :",scores.mean())
clf = tree.DecisionTreeClassifier(random_state=0).fit(X_train,Y_train)

scores = clf.score(X_test,Y_test)



print("sans cross validation: ",scores)

scores = cross_val_score(clf, X, Y, cv=10)

print("avec: ",scores.mean())

tree.export_graphviz(clf, out_file='tree.dot')

!dot -Tpng tree.dot -o tree_limited.png -Gdpi=600

Image(filename = 'tree_limited.png')