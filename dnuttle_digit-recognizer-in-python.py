# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
features_train = train.iloc[1:35000,1:]
label_train = train.iloc[1:35000,0]
#features_test = test.iloc[1:, :]
features_test = train.iloc[35000:,1:]
label_test = train.iloc[35000:,0]
#Lesson learned: logistic regression is verrrrry slow on this kind of problem
"""
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(features_train, label_train)
clf.score(features_test, label_test)
"""
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=12)
clf.fit(features_train, label_train)

print(clf.score(features_test, label_test))
#for idx in range(0,10):
    #print(type(features_train.iloc[idx]))
    #print(clf.predict(features_train.iloc[idx].reshape(1, -1)), label_train.iloc[idx])
