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
#importing Data

data=pd.read_csv("../input/creditcard.csv")
data.info()
data.head(5)
#Checking subcount of class variable

data['Class'].value_counts()
#Checking Total missing value

data.isnull().sum(axis=0)
import matplotlib.pyplot as plt

import seaborn as sns
#Checking count of Target variable using countplot

sns.countplot('Class',data=data)


#sns.distplot(data['Class'],kde=True)
#checking Corelation between independent feature

plt.subplots(figsize=(20, 20))

sns.heatmap(data.corr(),annot=True, fmt=".1g")



#Corelation matrix

data.corr()
X=data.iloc[:,0:-1]
y=data.iloc[:,-1]
X.head(2)
y.head(2)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.tree import DecisionTreeClassifier

classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
#Fitting model

classifier.fit(X_train,y_train)
#predicting the value

y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
#Building confusion matrix

cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import roc_curve,roc_auc_score,precision_recall_curve,f1_score
probs=classifier.predict_proba(X_test)
probs=probs[:,1]
fpr, tpr, thresholds = roc_curve(y_test,probs)
#ROC Curve

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(fpr, tpr, marker='*')
# calculate precision-recall curve

precision, recall, thresholds = precision_recall_curve(y_test, probs)
#Calculate f1 score

f1 = f1_score(y_test, y_pred)
f1
from sklearn.ensemble import RandomForestClassifier
Rclassifier=RandomForestClassifier(n_estimators=500,criterion='entropy')
Rclassifier.fit(X_train,y_train)
y_pred1=Rclassifier.predict(X_test)
Rcm=confusion_matrix(y_test,y_pred1)
Rcm
#Calculate f1 score

R_f1 = f1_score(y_test, y_pred1)
R_f1