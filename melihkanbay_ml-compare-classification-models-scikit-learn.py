# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
data.info()
data.head()
A = data[data["class"]=="Abnormal"]

N = data[data["class"]=="Normal"]
labels=data["class"].value_counts().index

sizes=data["class"].value_counts().values

plt.figure(figsize=(11,11))

plt.pie(sizes,labels=("Abnormal","Normal"),autopct="%1.f%%")

plt.title("Value counts of class",size=25)

plt.legend()

plt.show()

print("Numbers of Value counts\n",data.loc[:,'class'].value_counts())
plt.scatter(A.degree_spondylolisthesis,A.pelvic_incidence,color="red",label="Abnormal")

plt.scatter(N.degree_spondylolisthesis,N.pelvic_incidence,color="green",label="Normal")

plt.xlabel("degree_spondylolisthesis")

plt.ylabel("pelvic_incidence")

plt.legend()

plt.show()
data["class"]= [1 if i =="Abnormal" else 0 for i in data["class"]]

y = data["class"].values

x=data.drop(["class"],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(x_train,y_train)
print(" {}  knn score: {}".format(5,knn.score(x_test,y_test)))
score_list=[]

for i in range(1,25):

    knn2=KNeighborsClassifier(n_neighbors=i)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))



plt.figure(figsize=(12,8))

plt.plot(range(1,25),score_list)

plt.xlabel("K values")

plt.ylabel("Acuuracy")

plt.show()
y_pred = knn.predict(x_test)

y_true = y_test



from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_true,y_pred)



f, ax = plt.subplots(figsize=(8,8))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.title("Confision Matrix")

plt.show()

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(solver='lbfgs')

lr.fit(x_train,y_train)

print()
print("Accuracy score: ",lr.score(x_test,y_test))
y_pred = lr.predict(x_test)

y_true = y_test



from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_true,y_pred)



f, ax = plt.subplots(figsize=(8,8))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.title("Confision Matrix")

plt.show()

from sklearn.svm import SVC



svm = SVC(random_state = 1,gamma='auto' )

svm.fit(x_train,y_train)



print("Accuracy score: ",svm.score(x_test,y_test))
y_pred = svm.predict(x_test)

y_true = y_test



from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_true,y_pred)



f, ax = plt.subplots(figsize=(8,8))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.title("Confision Matrix")

plt.show()
from sklearn.naive_bayes import GaussianNB 



nb = GaussianNB()

nb.fit(x_train,y_train)



print("Accuracy score: ",nb.score(x_test,y_test)) 
y_pred = nb.predict(x_test)

y_true = y_test



from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_true,y_pred)



f, ax = plt.subplots(figsize=(8,8))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.title("Confision Matrix")

plt.show()
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()

dt.fit(x_train,y_train)



print("Accuracy score: ",dt.score(x_test,y_test)) 
y_pred = dt.predict(x_test)

y_true = y_test



from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_true,y_pred)



f, ax = plt.subplots(figsize=(8,8))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.title("Confision Matrix")

plt.show()
from sklearn.ensemble import RandomForestClassifier

rt=RandomForestClassifier(n_estimators=35,random_state=1)

rt.fit(x_train,y_train)



print("score: ",rt.score(x_test,y_test)) 
score_list2=[]

for i in range(1,50):

    rt2=RandomForestClassifier(n_estimators=i,random_state=1)

    rt2.fit(x_train,y_train)

    score_list2.append(rt2.score(x_test,y_test))



plt.figure(figsize=(12,8))

plt.plot(range(1,50),score_list2)

plt.xlabel("Esimator values")

plt.ylabel("Acuuracy")

plt.show()
y_pred = rt.predict(x_test)

y_true = y_test



from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_true,y_pred)



f, ax = plt.subplots(figsize=(8,8))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.title("Confision Matrix")

plt.show()
df = pd.DataFrame(

{"classification_models" : ["Random Forrest","KNN","Logistic Regression","Naive Bayes" ,"Decision Tree","SVM"],

"accuracy_score" : [0.8709677419354839,0.8548387096774194,0.8548387096774194,0.8225806451612904,0.7903225806451613,0.6774193548387096]},

index = [1,2,3,4,5,6])
df.head()
import plotly.express as px

fig = px.bar(df, x='classification_models', y='accuracy_score')

fig.show()