# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.datasets import load_iris

iris = load_iris()
dir(iris)
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.2)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train, y_train)

lr.score(X_test, y_test)
from  sklearn.svm import SVC

svm = SVC()

svm.fit(X_train, y_train)

svm.score(X_test, y_test)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 40)

rf.fit(X_train, y_train)

rf.score(X_test, y_test)
from sklearn.model_selection import cross_val_score

cross_val_score(LogisticRegression(), iris.data, iris.target)
cross_val_score(RandomForestClassifier(n_estimators = 60), iris.data, iris.target)
cross_val_score(SVC(), iris.data, iris.target)