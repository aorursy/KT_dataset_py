import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from sklearn import datasets

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

import numpy as np

import pandas as pd

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score
churn_data =pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv',index_col='RowNumber')
churn_data.info()
# some columns are totally unproductive so let's remove them

churn_data.drop(['CustomerId','Surname'],axis=1,inplace=True)
Geography_dummies = pd.get_dummies(prefix='Geo',data=churn_data,columns=['Geography'])

Gender_dummies = Geography_dummies.replace(to_replace={'Gender': {'Female': 1,'Male':0}})

churn_data_encoded = Gender_dummies
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

sns.countplot(y=churn_data_encoded.Exited ,data=churn_data_encoded)

plt.xlabel("Count of each Target class")

plt.ylabel("Target classes")

plt.show()
X = churn_data_encoded.drop(['Exited'],axis=1)

y = churn_data_encoded.Exited
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
Model = SGDClassifier

param = {"loss": "log", "penalty": "l2"}

reg = Model(**param)

reg.fit(X_train, y_train)

predictions = reg.predict(X_test)

predictions = (predictions > 0.5)

accuracy = accuracy_score(y_test, predictions)

print ("Accuracy score of {0}: {1}".format(Model.__name__, accuracy))
#import classification_report

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))
from sklearn.svm import LinearSVC

Model = LinearSVC

clf = LinearSVC(random_state=0, tol=1e-5)

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

predictions = (predictions > 0.5)

accuracy = accuracy_score(y_test, predictions)

print ("Accuracy score of {0}: {1}".format(Model.__name__, accuracy))
#import classification_report

print(classification_report(y_test,predictions))