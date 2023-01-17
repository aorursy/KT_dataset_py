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
df = pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')

df
x = df.drop(columns=['price_range'])

x
y = df['price_range']

y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

knn_clf = KNeighborsClassifier()

gnb_clf = GaussianNB()

rnd_clf = RandomForestClassifier()

dt_clf = DecisionTreeClassifier()
voting_clf = VotingClassifier( estimators=[ ('knn',knn_clf),('gnb', gnb_clf),('rf', rnd_clf),('dt',dt_clf)

 ], voting='hard' )

voting_clf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

for clf in (knn_clf,gnb_clf,rnd_clf,dt_clf,voting_clf):

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))