# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
df=pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

df
df.info()
df = df.sample(frac=1).reset_index(drop=True)

df
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.33)

train
train[['sex', 'target']].groupby(['sex'], as_index=False).mean().sort_values(by='target', ascending=False)

train[['cp', 'target']].groupby(['cp'], as_index=False).mean().sort_values(by='target', ascending=False)



train[['fbs', 'target']].groupby(['fbs'], as_index=False).mean().sort_values(by='target', ascending=False)

train[['restecg', 'target']].groupby(['restecg'], as_index=False).mean().sort_values(by='target', ascending=False)

train['restecg'].value_counts()
print(df[['restecg', 'fbs']])



train['fbs'].value_counts()
X_train = train.drop("target", axis=1)

Y_train = train["target"]

X_test=test.drop("target",axis=1)

Y_test=test["target"]

decision_tree = DecisionTreeClassifier()

model=decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
plt.scatter(Y_test,Y_pred)



model.score(X_test,Y_test)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

model.score(X_test,Y_test)
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log

model.score(X_test,Y_test)
Y_test
Y_pred
a = np.asarray(Y_test)

a