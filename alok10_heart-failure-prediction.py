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

import matplotlib.pyplot as plt

import seaborn as sns
data=pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")
data.head()
data.tail()
sns.countplot(x='sex',data=data,hue='DEATH_EVENT')
y=data['DEATH_EVENT']

X=data.drop(columns=['DEATH_EVENT'])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
X_train.shape
y_train.shape
data.shape
from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression()

rfe=RFE(classifier,20)

rfe=rfe.fit(X_train,y_train)

print(rfe.support_)
from  sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()

logmodel.fit(X_train,y_train)
pred=logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cf=confusion_matrix(y_test,pred)
sns.heatmap(cf,annot=True,fmt='g')
accuracy_score(y_test,pred)
from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()

X_train2=pd.DataFrame(sc_X.fit_transform(X_train))

X_test2=pd.DataFrame(sc_X.fit_transform(X_test))
X_train2.columns=X_train.columns.values

X_test2.columns=X_test.columns.values

X_train2.index=X_train.index.values

X_test2.index=X_test.index.values
X_train=X_train2

X_test=X_test2
logmodel.fit(X_train,y_train)
pred=logmodel.predict(X_test)
accuracy_score(y_test,pred)
cm=confusion_matrix(y_test,pred)
sns.heatmap(cm,annot=True,fmt='g')
from sklearn.svm import SVC

classifier = SVC(random_state = 0, kernel = 'linear')

classifier.fit(X_train, y_train)
pred=classifier.predict(X_test)
accuracy_score(y_test,pred)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,accuracy_score
model=RandomForestClassifier(random_state=0,n_estimators=100,criterion='entropy')
model.fit(X_train,y_train)
r_pred=model.predict(X_test)
accuracy_score(y_test,r_pred)
cn=confusion_matrix(y_test,r_pred)
sns.heatmap(cn,fmt='g',annot=True)