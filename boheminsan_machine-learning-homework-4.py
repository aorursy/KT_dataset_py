# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/weather-dataset-rattle-package/weatherAUS.csv')

results={}

data.tail(10)
data.info()
data.dropna(inplace=True)

data.drop(["Date","Location","RISK_MM","WindGustDir","WindDir9am","WindDir3pm","RainTomorrow"],axis=1, inplace=True)
data.tail()
data.info()
Rain=data[data.RainToday=="Yes"]

NoRain=data[data.RainToday=="No"]
sns.countplot(x="RainToday", data=data)

data.loc[:,'RainToday'].value_counts()
data['RainToday'] = [1 if each == 'Yes' else 0 for each in data['RainToday']]

y=data['RainToday'].values

x_data=data.drop(['RainToday'],axis=1)

dataf=x_data.corr()

plt.figure(figsize=(16,16))

ax=sns.heatmap(dataf,annot=True, linewidths=1, fmt= '.2f', annot_kws={"size": 12})

plt.show()
x=x_data-np.min(x_data)/(np.max(x_data)-np.min(x_data))

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=7)

knn.fit(x_train,y_train)

y_head_knn =knn.predict(x_test)

y_true=y_test

results["knn"]=knn.score(x_test,y_test)

print("{} nn score: {}".format(7,knn.score(x_test,y_test)))
from sklearn.metrics import confusion_matrix

cm_knn = confusion_matrix(y_true,y_head_knn)

import seaborn as sns

f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(cm_knn,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("ypred")

plt.ylabel("ytrue")

plt.show()
x=x_data-np.min(x_data)/(np.max(x_data)-np.min(x_data))

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.preprocessing import MinMaxScaler

scaling = MinMaxScaler(feature_range=(-1,1)).fit(x_train)

x_train2 = scaling.transform(x_train)

x_test2 = scaling.transform(x_test)
from sklearn.svm import SVC

svm=SVC(random_state=42,gamma=1)

svm.fit(x_train2,y_train)

y_head_svm = svm.predict(x_test2)

y_true=y_test

results["svm"]=svm.score(x_test2,y_test)

print("svm accuracy:{}%",format(svm.score(x_test2,y_test)))
from sklearn.metrics import confusion_matrix

cm_svm = confusion_matrix(y_true,y_head_svm)

import seaborn as sns

f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(cm_svm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("ypred")

plt.ylabel("ytrue")

plt.show()
from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

nb.fit(x_train,y_train)

y_head_nb = nb.predict(x_test)

y_true=y_test

results["nb"]=nb.score(x_test,y_test)

print("nb accuracy:{}%",format(nb.score(x_test,y_test)))
from sklearn.metrics import confusion_matrix

cm_nb = confusion_matrix(y_true,y_head_nb)

import seaborn as sns

f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(cm_nb,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("ypred")

plt.ylabel("ytrue")

plt.show()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier()

dtc.fit(x_train, y_train)

y_head_dtc = dtc.predict(x_test)

y_true=y_test

results["dtc"]=dtc.score(x_test,y_test)

print("decision tree accuracy:{}%",format(dtc.score(x_test,y_test)))
from sklearn.metrics import confusion_matrix

cm_dtc = confusion_matrix(y_true,y_head_dtc)

import seaborn as sns

f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(cm_dtc,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("ypred")

plt.ylabel("ytrue")

plt.show()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=40,random_state=1)

rf.fit(x_train,y_train)

y_head_rfc = rf.predict(x_test)

y_true=y_test

results["rf"]=rf.score(x_test,y_test)

print("random forest accuracy:{}%",format(rf.score(x_test,y_test)))
from sklearn.metrics import confusion_matrix

cm_rf = confusion_matrix(y_true,y_head_rfc)

import seaborn as sns

f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(cm_rf,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("ypred")

plt.ylabel("ytrue")

plt.show()
acc=list(results.values())

mets=list(results.keys())

plt.bar(mets, acc, align='center', alpha=0.7, color="#00ff00")

plt.show()
#I quoted that whole sequence from efeergun96. Thank you for that idea.



plt.figure(figsize=(24,12))



plt.suptitle("Confusion Matrixes",fontsize=24)



plt.subplot(2,2,1)

plt.title("Decision Tree Classifier Confusion Matrix")

plt.xlabel("ypred")

plt.ylabel("ytrue")

sns.heatmap(cm_dtc,cbar=False,annot=True,cmap="CMRmap_r",fmt="d")



plt.subplot(2,3,2)

plt.title("K Nearest Neighbors Confusion Matrix")

sns.heatmap(cm_knn,cbar=False,annot=True,cmap="CMRmap_r",fmt="d")



plt.subplot(2,3,3)

plt.title("Naive Bayes Confusion Matrix")

sns.heatmap(cm_nb,cbar=False,annot=True,cmap="CMRmap_r",fmt="d")



plt.subplot(2,3,4)

plt.title("Random Forest Confusion Matrix")

sns.heatmap(cm_rf,cbar=False,annot=True,cmap="CMRmap_r",fmt="d")



plt.subplot(2,3,5)

plt.title("Support Vector Machine Confusion Matrix")

sns.heatmap(cm_svm,cbar=False,annot=True,cmap="CMRmap_r",fmt="d")



plt.show()