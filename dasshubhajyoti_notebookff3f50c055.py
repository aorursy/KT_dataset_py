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
data = pd.read_csv("/kaggle/input/heartcsv/heart.csv")

data.head(10)
data.shape
data.isnull().sum()
data.describe()
X = data.drop('target',axis = 1)
y = data['target']

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scalar.fit(X)
X = scalar.transform(X)
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
mod1 = lr.predict(X_test)
from sklearn.metrics import confusion_matrix 
cm1 = confusion_matrix(y_test,mod1)
accuracy1 = (float(cm1.diagonal().sum())/len(y_test))*100
print("\nAccuracy of LOGISTIC_REGRESSION For The Given Dataset :",accuracy1)
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)

mod2 = clf.predict(X_test)
from sklearn.metrics import confusion_matrix 
cm2 = confusion_matrix(y_test,mod2)
accuracy2 = (float(cm2.diagonal().sum())/len(y_test))*100
print("\nAccuracy of DECISION_TREE For The Given Dataset :",accuracy2)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')

model.fit(X_train,y_train)
mod3 = model.predict(X_test)
from sklearn.metrics import confusion_matrix 
cm3 = confusion_matrix(y_test,mod3)
accuracy3 = (float(cm3.diagonal().sum())/len(y_test))*100
print("\nAccuracy of RANDOM_FOREST For The Given Dataset :",accuracy3)
from sklearn.svm import SVC
classifier= SVC(kernel ='rbf', random_state = 1) 
classifier.fit(X_train,y_train)
mod4 = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix 
cm4 = confusion_matrix(y_test,mod4)
accuracy4 = (float(cm4.diagonal().sum())/len(y_test))*100
print("\nAccuracy of SVM_RBF For The Given Dataset :",accuracy4)
from sklearn.svm import SVC
classifier= SVC(kernel ='linear', random_state = 1) 
classifier.fit(X_train,y_train)
mod5 = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix 
cm5 = confusion_matrix(y_test,mod5)
accuracy5 = (float(cm5.diagonal().sum())/len(y_test))*100
print("\nAccuracy of SVM_LINEAR For The Given Dataset :",accuracy5)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb = gnb.fit(X_train,y_train)
mod6 = gnb.predict(X_test)
from sklearn.metrics import confusion_matrix 
cm6 = confusion_matrix(y_test,mod6)
accuracy6 = (float(cm6.diagonal().sum())/len(y_test))*100
print("\nAccuracy of naive_bayes For The Given Dataset :",accuracy6)
