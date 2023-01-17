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

import numpy as np
bank=pd.read_csv('../input/UnivBank.csv')

bank.shape
bank.head(9)
bank.isna().sum()
bank.duplicated().sum()
bank.info()
bank.describe()
import seaborn as sns

bank.plot(kind='box', figsize=(60,30))
bank.corr()
sns.pairplot(bank,hue ='CreditCard')
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, accuracy_score

from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()

m_input = scaler.fit_transform(bank.drop(columns=['CreditCard']))

m_output = bank['CreditCard']

X_train, X_test, y_train, y_test = train_test_split(m_input, m_output, test_size=0.2, random_state=0)

log_model = LogisticRegression()

log_model.fit(X_train, y_train)

train_Pred = log_model.predict(X_train)

test_Pred = log_model.predict(X_test)
log_model
from sklearn.metrics import confusion_matrix

confusion_matrix(y_train,train_Pred)
confusion_matrix(y_test,test_Pred)
accuracy_score(y_train,train_Pred)
accuracy_score(y_test,test_Pred)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dep = {}

for i in range(1, 25):

    print('Accuracy score using max_depth =', i, end = ': ')

    dt = DecisionTreeClassifier(max_depth=i,random_state = 45)

    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)

    dep[i]=accuracy_score(y_test, y_pred)

    print(accuracy_score(y_test, y_pred))

ma=max(dep, key=dep.get)

print(ma, dep[ma])
dep = {}

for i in np.arange(0.1, 1.0, 0.1):

    print('Accuracy score using max_features =', i, end = ': ')

    dt = DecisionTreeClassifier(max_depth=3,random_state = 45,max_features=i)

    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)

    dep[i]=accuracy_score(y_test, y_pred)

    print(accuracy_score(y_test, y_pred))

ma=max(dep, key=dep.get)

print(ma, dep[ma])
d = []

for i in ['entropy','gini']:

    print('Accuracy score using criterion =', i, end = ': ')

    dt = DecisionTreeClassifier(criterion=i,max_depth=3,random_state = 45,max_features=0.8)

    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)

    d.append(accuracy_score(y_test, y_pred))

    print(accuracy_score(y_test, y_pred))

print('Max: ', max(d))
dep = {}

for i in range(2, 10):

    print('Accuracy score using min_samples_split =', i, end = ': ')

    dt = DecisionTreeClassifier(criterion='gini',max_depth=3,random_state = 45,max_features=0.8,min_samples_split=i)

    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)

    dep[i]=accuracy_score(y_test, y_pred)

    print(accuracy_score(y_test, y_pred))

ma=max(dep, key=dep.get)

print(ma, dep[ma])
dt = DecisionTreeClassifier(criterion='gini',max_depth=3,random_state = 45,max_features=0.8,min_samples_split=2)

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

print(accuracy_score(y_test, y_pred))

y_pred_train = dt.predict(X_train)

print(accuracy_score(y_train, y_pred_train))
from sklearn.svm import SVC

svm=SVC()

svm
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

print(accuracy_score(y_test, y_pred))

y_pred_train = svm.predict(X_train)

print(accuracy_score(y_train, y_pred_train))