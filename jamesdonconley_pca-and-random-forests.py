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
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

#print(train.head)

targets = train['target']

train = train.drop(columns=['id', 'target'])

#print(test.head)

ids = test['id']

test = test.drop(columns=['id'])

from sklearn.decomposition import PCA

pca = PCA(n_components=50)

pca.fit(train.values)

print(train.values.shape)

print(test.shape)

print(pca.explained_variance_ratio_)

X = pca.transform(train.values)

transformed_test = pca.transform(test.values)
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier



kf = KFold(n_splits=10)



scores = [] 



for train_index, test_index in kf.split(X):

    x_train, x_test = X[train_index], X[test_index]

    y_train, y_test = targets[train_index], targets[test_index]

    clf = RandomForestClassifier(n_estimators=27, max_depth=4,random_state=0)

    clf.fit(x_train, y_train)

    score = clf.score(x_test, y_test)

    scores.append(score)

    print(score)

print("Average Score: ", sum(scores)/10)

clf.fit(X, targets)

predictions = clf.predict(transformed_test)



    

output = pd.DataFrame()

output['id'] = ids

output['target'] = predictions

output.to_csv('output.csv',index=None)