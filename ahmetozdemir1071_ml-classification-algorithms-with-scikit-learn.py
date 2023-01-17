# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('../input/voicegender/voice.csv')
data.info()
data.head()
data.isnull().values.any()
y=data.label

x_data=data.drop(["label"],axis=1)

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
data.label=[0 if each=="male" else 1 for each in data.label]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#  Distribution of gender 

print(y.value_counts())

plt.pie(y.value_counts(),labels=["Female","Male"],colors=["pink","green"],autopct='%1.0f%%')

plt.title("Distribution of gender:")

plt.show()
data_corr=data.corr()

plt.figure(figsize=(12,12))

sns.heatmap(data_corr,annot=True, fmt= '.2f')
plt.figure(figsize=(8,5))

sns.scatterplot(data=data, x="meanfun", y="sp.ent",hue="label")

plt.title("Scatter of Spectral Entropy by Average of Fundamental Frequency")

plt.xlabel("Average of Fundamental Frequency")

plt.ylabel("Spectral Entropy")

plt.show()
plt.figure(figsize=(8,5))

sns.scatterplot(data=data, x="meanfun", y="meanfreq",hue="label")

plt.title("Scatter of Spectral Entropy by Average of Fundamental Frequency")

plt.xlabel("Average of Fundamental Frequency")

plt.ylabel("Mean Frequency (in kHz)")

plt.show()
plt.figure(figsize=(7,5))

sns.stripplot(data=data, x="label", y="IQR",jitter=True)

plt.title("IQR by Gender (Stripplot)")

plt.xlabel("Gender")

plt.ylabel("IQR")

plt.show()
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x_train,y_train)

print("Test Accuracy: {} %".format(lr.score(x_test,y_test)*100))
#Confusion matrix:

y_pred_lr=lr.predict(x_test)

y_true=y_test

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_pred_lr,y_true)

f,ax=plt.subplots(figsize=(3,3))

sns.heatmap(cm,annot=True,linecolor="blue",linewidth=0.3,fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.neighbors import KNeighborsClassifier



scoreList=[]

#Let's find the optimal number k

for each in range(1,15):

    knn=KNeighborsClassifier(n_neighbors=each)

    knn.fit(x_train,y_train)

    scoreList.append(knn.score(x_test,y_test))

plt.figure(figsize=(7,5))

plt.plot(range(1,15),scoreList)

plt.xlabel("K Values")

plt.ylabel("Accuracy")

plt.show()
#Confusion matrix (k=9):

knn=KNeighborsClassifier(n_neighbors=9)

knn.fit(x_train,y_train)

print("Test Accuracy: {} %".format(knn.score(x_test,y_test)*100))

y_pred_knn=knn.predict(x_test)

y_true=y_test

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_pred_knn,y_true)

f,ax=plt.subplots(figsize=(3,3))

sns.heatmap(cm,annot=True,linecolor="blue",linewidth=0.3,fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.svm import SVC

svm=SVC(random_state=42)

svm.fit(x_train,y_train)

print("Test Accuracy: {} %".format(svm.score(x_test,y_test)*100))
#Confusion matrix:

y_pred_svm=svm.predict(x_test)

y_true=y_test

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_pred_svm,y_true)

f,ax=plt.subplots(figsize=(3,3))

sns.heatmap(cm,annot=True,linecolor="blue",linewidth=0.3,fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

nb.fit(x_train,y_train)

print("Test Accuracy: {} %".format(nb.score(x_test,y_test)*100))
#Confusion matrix:

y_pred_nb=nb.predict(x_test)

y_true=y_test

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_pred_nb,y_true)

f,ax=plt.subplots(figsize=(3,3))

sns.heatmap(cm,annot=True,linecolor="blue",linewidth=0.3,fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()

dt.fit(x_train,y_train)

print("Test Accuracy: {} %".format(dt.score(x_test,y_test)*100))
#Confusion matrix:

y_pred_dt=dt.predict(x_test)

y_true=y_test

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_pred_dt,y_true)

f,ax=plt.subplots(figsize=(3,3))

sns.heatmap(cm,annot=True,linecolor="blue",linewidth=0.3,fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=10,random_state=42)

rf.fit(x_train,y_train)

print("Test Accuracy = {} %".format(rf.score(x_test,y_test)*100))
#Confusion Matrix

y_pred_rf=rf.predict(x_test)

y_true=y_test

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_pred_rf,y_true)

f,ax=plt.subplots(figsize=(3,3))

sns.heatmap(cm,annot=True,linecolor="blue",linewidth=0.3,fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()