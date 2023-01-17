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
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

df = pd.read_csv('../input/train.csv')
df2 = pd.read_csv('../input/test.csv')
df3 = pd.read_csv('../input/users.csv')
df4 = pd.read_csv('../input/sampleSubmission.csv')

#df.head(100000)

from sklearn.model_selection import train_test_split
#cross_validation
X = df.drop(labels = 'interested', axis = 1)
X = df.drop(labels = 'timestamp', axis = 1) #can't be processed so dropping
y = df['interested']
#y = df2['event_id', 'user_id']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, predictions))

print(confusion_matrix(y_test,predictions))

print("Accuracy on decision tree training set: {:.3f}".format(dtree.score(X_train, y_train)))
print("Accuracy on decision tree test set: {:.3f}".format(dtree.score(X_test, y_test)))


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=2000)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))
print("Accuracy on random forest training set: {:.3f}".format(rfc.score(X_train, y_train)))
print("Accuracy on random forest test set: {:.3f}".format(rfc.score(X_test, y_test)))