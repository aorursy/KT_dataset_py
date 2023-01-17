# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df.head()
a = df.iloc[50, 1:].values.reshape(28,28)



plt.imshow(a)
from sklearn import tree

from sklearn import ensemble

from sklearn.model_selection import train_test_split



target = 'label'



y = df[target]

X = df.drop(columns = target)





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



print('DecisionTreeClassifier')

for n in [3,4,5,6,7]:

    model = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = n, random_state=42)

    model.fit(X_train, y_train)

    print('max_depth: ', n, 'score: ', model.score(X_test,y_test))

    

print('RandomForestClassifier')

for k in [10, 300, 500, 2000]:

    for n in [3,4,5,6,7]:

        

        model = ensemble.RandomForestClassifier(n_estimators = k, criterion = 'gini', max_depth = n, random_state=42)



        model.fit(X_train, y_train)



        print('n_estimators: ', k, 'max_depth: ', n, 'score: ', model.score(X_test,y_test))
model.predict_proba(X_test)
ind = 34



print(model.predict(X_test.iloc[[ind], :]))

plt.imshow(X_test.iloc[ind, :].values.reshape(28,28))


