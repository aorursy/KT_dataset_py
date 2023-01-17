# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/income/train.csv')
df
df.columns
sex = {"Male": 1, "Female": 2 }

df['gender'] = df['gender'].map(sex)

df
numeric_features = ['age','fnlwgt','educational-num','gender','capital-gain','capital-loss','hours-per-week','income_>50K']

# Identify Categorical features
cat_features = ['workclass','education','marital-status', 'occupation', 'relationship', 'race', 'native']
numeric_features

g = sns.heatmap(df[numeric_features].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
x=df.drop(['income_>50K'],axis=1)
y=df['income_>50K']
print(x.shape)
print(y.shape)
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size = 0.3, random_state = 0) 
print(X_train.shape)
print(X_test.shape)
X_train=X_train.drop(X_train.columns[[1,3,5,6,7,8,13]],axis=1)
X_train
from sklearn.tree import DecisionTreeClassifier  
classifier1 = DecisionTreeClassifier(criterion='gini')  
classifier1.fit(X_train, y_train) 
X_test=X_test.drop(X_test.columns[[1,3,5,6,7,8,13]],axis=1)
X_test
y_predtrain = classifier1.predict(X_train) 
y_predtrain
y_predtrainlist=list(y_predtrain)
TrainCountOfgt50K=y_predtrainlist.count(1)
TrainCountOflt50K=y_predtrainlist.count(0)
print(TrainCountOfgt50K)
print(TrainCountOflt50K)
y_pred = classifier1.predict(X_test)  
print(y_pred)
y_predlist=list(y_pred)
TestCountOfgt50K=y_predlist.count(1)
TestCountOflt50K=y_predlist.count(0)
print(TestCountOfgt50K)
print(TestCountOflt50K)
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))
from sklearn.metrics import accuracy_score #importing accuracy_score function from sklearn.metrics package
acc = accuracy_score(y_test,y_pred)
print("Accuracy for this model {} %".format(acc*100))
print(classifier1.feature_importances_)
from sklearn.neighbors import KNeighborsClassifier


classifier4 = KNeighborsClassifier(n_neighbors= 7)  
classifier4.fit(X_train, y_train)
y_pred_4 = classifier4.predict(X_test)  
print(y_pred_4)

acc_3 = accuracy_score(y_test,y_pred_4)
print("Accuracy  model {} %".format(acc_3*100))
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred_3))
print(classification_report(y_test, y_pred_3))
from sklearn.ensemble import RandomForestClassifier
rclf = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=0)

rclf.fit(X_train, y_train)
ry_pred = rclf.predict(X_test)  
print(ry_pred)
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, ry_pred))  
print(classification_report(y_test, ry_pred)) 
from sklearn.metrics import accuracy_score #importing accuracy_score function from sklearn.metrics package
acc = accuracy_score(y_test,ry_pred)
print("Accuracy for this model {} %".format(acc*100))
X_test
x=x.drop(x.columns[[1,3,5,6,7,8,13]],axis=1)
x
from sklearn.svm import SVC

clf = SVC()
clf.fit(x, y) 
svm_pred = clf.predict(X_test)  
print(svm_pred)
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, svm_pred))  
print(classification_report(y_test, svm_pred))
from sklearn.metrics import accuracy_score #importing accuracy_score function from sklearn.metrics package
acc = accuracy_score(y_test,svm_pred)
print("Accuracy for this model {} %".format(acc*100))
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train,y_train)

lr_pred = lr.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, lr_pred))  
print(classification_report(y_test, lr_pred))

from sklearn.metrics import accuracy_score #importing accuracy_score function from sklearn.metrics package
acc = accuracy_score(y_test,lr_pred)
print("Accuracy for this model {} %".format(acc*100))