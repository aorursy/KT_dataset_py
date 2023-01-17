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
test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')

train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

sample = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
train
sample
test
from __future__ import print_function, division



%matplotlib inline



import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

import graphviz 

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import random

import seaborn as sns

from IPython.display import display

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.impute import SimpleImputer

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn import tree

from sklearn.svm import SVC

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn import tree

from sklearn import ensemble

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree.export import export_text
train.info()
train.keys()
train.isna().sum()
sns.countplot(train['target'])
train.describe()
train.nunique()
train.head()
train.target.value_counts() 
X_train = train.iloc[:, 2:].values

Y_train = train.iloc[:, 1].values

X_test = test.iloc[:, 1:].values

X_train
Y_train = Y_train.astype('float64') 

Y_train
print(Y_train.dtype) 
X_test
print('X_train shape is :', X_train.shape, '\nY_train shape is ',Y_train.shape)
le = LabelEncoder()

X_train[:,0] = le.fit_transform(X_train[:,0])

X_test[:,0] = le.fit_transform(X_test[:,0])
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, Y_train)
Y_preds_knn = knn.predict(X_test)
print(classification_report(Y_train, Y_preds_knn))
knn.p
lg = LogisticRegression()

lg.fit(X_train,Y_train)
Y_preds_lg = lg.predict(X_test)

Y_preds_lg
print(classification_report(Y_train, Y_preds_lg))
gnb = GaussianNB()
gnb.fit(X_train,Y_train)
Y_preds_gnb = gnb.predict(X_test)

Y_preds_gnb
print(classification_report(Y_train, Y_preds_gnb))
dt = DecisionTreeClassifier(criterion='entropy', max_depth=5)
dt.fit(X_train, Y_train)
Y_preds_dt = dt.predict(X_test)
print(classification_report(Y_train, Y_preds_dt))
tree.plot_tree(dt)
rfc=RandomForestClassifier()

svc=SVC()
Y_preds_rfc = rfc.predict(X_test)
print(classification_report(Y_train, Y_preds_rfc))
X = train.iloc[:, 2:].values

Y = train.iloc[:, [1]].values

X
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 33, stratify=Y)
knn.fit(X_train, Y_train)

lg.fit(X_train, Y_train)

gnb.fit(X_train, Y_train)

dt.fit(X_train, Y_train)

rfc.fit(X_train, Y_train)
Y_preds_knn = knn.predict(X_test)

Y_preds_lg = lg.predict(X_test)

Y_preds_gnb = gnb.predict(X_test)

Y_preds_dt = dt.predict(X_test)

Y_preds_rfc = rfc.predict(X_test)
print('accuracy KNN: ',accuracy_score(Y_test, Y_preds_knn), '\naccuracy LG: ' ,accuracy_score(Y_test, Y_preds_lg), '\naccuracy GNB: ' ,accuracy_score(Y_test, Y_preds_gnb), '\naccuracy DT: ' ,accuracy_score(Y_test, Y_preds_dt), '\naccuracy rfc: ' ,accuracy_score(Y_test, Y_preds_rfc))
print('accuracy KNN: ',accuracy_score(Y_test, Y_preds_knn), '\naccuracy LG: ' ,accuracy_score(Y_test, Y_preds_lg), '\naccuracy GNB: ' ,accuracy_score(Y_test, Y_preds_gnb), '\naccuracy DT: ' ,accuracy_score(Y_test, Y_preds_dt), '\naccuracy rfc: ' ,accuracy_score(Y_test, Y_preds_rfc))