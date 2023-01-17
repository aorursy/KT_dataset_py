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
import matplotlib.pyplot as plt
dataset = pd.read_csv("../input/creditcard.csv")

dataset
X = dataset.iloc[:,0:30].values

y =dataset.iloc[:,-1].values


plt.scatter(X[y==0,0],X[y==0,29],color = 'g')

plt.scatter(X[y==1,0],X[y==1,29],color = 'r')

plt.show()
plt.hist(X[:,29],bins = 15)

plt.show
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y)
from sklearn.preprocessing import StandardScaler

Std = StandardScaler()

X_train = Std.fit_transform(X_train)

X_test = Std.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

log_reg.fit(X_train,y_train)
log_reg.score(X_test,y_test)

log_reg.score(X,y)
from sklearn.tree import DecisionTreeClassifier

d_t = DecisionTreeClassifier(max_depth = 10)

d_t.fit(X_train,y_train)
d_t.score(X_test,y_test)
d_t.score(X,y)