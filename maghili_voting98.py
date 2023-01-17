# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df.head()
X = df.drop(['label'], axis = 1)/255.
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = .2, random_state = 0)
#RandomForestClassifier
clf1 = RandomForestClassifier(n_estimators = 250)
clf1.fit(X_train, y_train)
print( clf1.score(X_test, y_test))
#NeuralNetworks
clf2 = MLPClassifier(hidden_layer_sizes = (400, 400, 400))
clf2.fit(X_train, y_train)
clf2.score(X_test, y_test)
#GradientBoostClassifier
clf3 = GradientBoostingClassifier(n_estimators = 200)
clf3.fit(X_train, y_train)
clf3.score(X_test, y_test)
#Voting
clf4 = VotingClassifier([('rf', clf1), ('mpl', clf2), ('gbc', clf3)], voting = 'soft')
clf4.fit(X_train, y_train)
clf4.score(X_test, y_test)
