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
import pandas as pd
from sklearn.model_selection import train_test_split
import pdb

df = pd.read_csv('../input/heart.csv')
df = df.dropna()
m, n = df.shape

X = df[df.columns[0:n-1 ]]
Y = df[df.columns[ n-1 ]]

X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.5,random_state=0)

from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=5)
clf.fit(X_train,y_train)
score = clf.score(X_test,y_test)
print(score)