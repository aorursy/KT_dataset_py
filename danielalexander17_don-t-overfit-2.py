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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
train = pd.read_csv('../input/dont-overfit-ii/train.csv')

test = pd.read_csv('../input/dont-overfit-ii/test.csv')

example = pd.read_csv('../input/dont-overfit-ii/sample_submission.csv')
train.info()

print('-'*20)

test.info()
from sklearn.model_selection import train_test_split

X = train.drop(['target'], axis=1)

Y = train['target']

X_train,  X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(X_train, Y_train)

Y_pred = rfc.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(Y_test, Y_pred))

print(classification_report(Y_test, Y_pred))

print(accuracy_score(Y_test, Y_pred))
rfc.fit(X_test, Y_test)

Y_pred = rfc.predict(X_train)

print(confusion_matrix(Y_train, Y_pred))

print(classification_report(Y_train, Y_pred))

print(accuracy_score(Y_train, Y_pred))
from sklearn.svm import SVC

model = SVC()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print(confusion_matrix(Y_test, Y_pred))

print(classification_report(Y_test, Y_pred))

print(accuracy_score(Y_test, Y_pred))
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X_train)

Y = scaler.fit_transform(X_test)
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(X, Y_train)

Y_pred = dtc.predict(Y)

print(confusion_matrix(Y_test, Y_pred))

print(classification_report(Y_test, Y_pred))

print(accuracy_score(Y_test, Y_pred))
i = range(15,20,1)

i = [*i]

acc_i = []

for max_depths in i:

    dtc = DecisionTreeClassifier(max_depth=max_depths)

    dtc.fit(X, Y_train)

    Y_pred = dtc.predict(Y)

    acc = accuracy_score(Y_test, Y_pred)

    acc_i.append(acc)
plt.plot(i, acc_i)
dtc = DecisionTreeClassifier(max_depth=19)

dtc.fit(X, Y_train)

Y_pred = dtc.predict(Y)

print(confusion_matrix(Y_test, Y_pred))

print(classification_report(Y_test, Y_pred))

print(accuracy_score(Y_test, Y_pred))
from sklearn.linear_model import LogisticRegression

reg = LogisticRegression()

reg.fit(X,Y_train)

Y_pred = dtc.predict(Y)

print(confusion_matrix(Y_test, Y_pred))

print(classification_report(Y_test, Y_pred))

print(accuracy_score(Y_test, Y_pred))
from sklearn.model_selection import cross_val_score

scores = cross_val_score(dtc, X, Y_train, scoring="accuracy", cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
from sklearn.model_selection import cross_val_score

scores = cross_val_score(reg, X, Y_train, scoring="accuracy", cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print(scores.mean())
# kesimpulan : model decision tree dan regresi logistik baik digunakan dalam kasus ini tetapi model reg log yang optimum 

# dari 4 model diatas untuk kasus ini.
scaler = StandardScaler()

X = scaler.fit_transform(train.drop(['target'], axis=1))

Y_train = train['target']

Y = scaler.fit_transform(test)



reg = LogisticRegression()

reg.fit(X,Y_train)

Y_pred = dtc.predict(Y)
submission = pd.DataFrame({'id':test['id'],

                           'target':Y_pred})

submission.head()
filename = 'submission.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)