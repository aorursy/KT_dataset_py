# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



data = pd.read_csv('../input/heart.csv')

# Any results you write to the current directory are saved as output.
data.head()
data.target.value_counts()
from sklearn.model_selection import train_test_split



train, test = train_test_split(data,

                              test_size = 0.2,

                              random_state = 100)

train_x = train.drop('target', axis=1)

train_y = train['target']



test_x = test.drop('target', axis=1)

test_y = test['target']

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
models = [DecisionTreeClassifier(random_state=100), 

          RandomForestClassifier(random_state=100,n_estimators=300),

          GradientBoostingClassifier(random_state=100, n_estimators=1000), 

          AdaBoostClassifier(random_state=100),

          KNeighborsClassifier(n_neighbors=3)]



for i in models:

    model = i

    model.fit(train_x,train_y)

    pred = model.predict(test_x)

    acc_score = accuracy_score(pred, test_y)

    print('The accuracy for',i,'is',acc_score*100)