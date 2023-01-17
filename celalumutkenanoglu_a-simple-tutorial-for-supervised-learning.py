# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from collections import Counter

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/glass/glass.csv")
df.info()
df.head()
df.columns
X = df.iloc[:,:-1]

Y = df.iloc[:,9]
Counter(Y)
X_n = (X-np.min(X))/(np.max(X)-np.min(X))
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X_n,Y,test_size=0.2,random_state=42)
score_list = []

model = []

cross_val_score_list=[]
tuning= [100,200,300,400,500,600,700,800]
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier 

score=[]

a=0

c=0

for i in tuning:

    rf = RandomForestClassifier(n_estimators=i, random_state=42)



    rf.fit(x_train,y_train)

    accuracies = cross_val_score(estimator=rf,X = x_train,y = y_train,cv=3)

#%%

    score.append(np.mean(accuracies))

    if np.mean(accuracies)>a:

        a=np.mean(accuracies)

        c=i

print("acc = ",a," best number of estimator = ",c)

print("std= ",np.std(accuracies))
plt.plot(tuning,score)
rf = RandomForestClassifier(n_estimators=100, random_state=42) #variable = our model



rf.fit(x_train,y_train) #fit our model with our train datas

print(rf.score(x_test,y_test)) #check score with our test datas

model.append("Random Forest Classifier") #store our models name for our graph

score_list.append(rf.score(x_test,y_test)) #store our test score for our graph
from sklearn.metrics import confusion_matrix

y_pred= rf.predict(x_test) #we use this code to predict our labels with our model

#confusion matrix part

categories = [1,2,3,5,6,7] 

cm = confusion_matrix(y_test,y_pred)

f , ax = plt.subplots(figsize=(5,5))



sns.heatmap(cm,annot = True,linewidths =0.5,linecolor ="Red",fmt=".0f",ax=ax)

ax.set_xticklabels(categories)

ax.set_yticklabels(categories)

plt.xlabel("y_pred")

plt.ylabel("y_true")







plt.show()
from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier()

accuracies = cross_val_score(estimator=rf,X = x_train,y = y_train,cv=3)

#%%

print("acc = ", np.mean(accuracies))

print("std= ",np.std(accuracies))
dt.fit(x_train,y_train)

print("acc = ", dt.score(x_test,y_test))

model.append("Decision Tree Classifier")

score_list.append(dt.score(x_test,y_test))
y_pred= dt.predict(x_test)



categories = [1,2,3,5,6,7]

cm = confusion_matrix(y_test,y_pred)

f , ax = plt.subplots(figsize=(5,5))



sns.heatmap(cm,annot = True,linewidths =0.5,linecolor ="Red",fmt=".0f",ax=ax)

ax.set_xticklabels(categories)

ax.set_yticklabels(categories)

plt.xlabel("y_pred")

plt.ylabel("y_true")



plt.show()
from sklearn.neighbors import KNeighborsClassifier

score =[]

a=0

c=0

for i in range(2,20):

    knn = KNeighborsClassifier(n_neighbors = i)

    accuracies = cross_val_score(estimator=knn,X = x_train,y = y_train,cv=3)

    score.append(np.mean(accuracies))

    if np.mean(accuracies)>a:

        a=np.mean(accuracies)

        c=i

    

print("acc = ",a," best number of neighbors = ",c)

print("std= ",np.std(accuracies))
plt.plot(range(2,20),score)
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train,y_train)

print("acc = ", knn.score(x_test,y_test))

model.append("K-Nearest Neighbors")

score_list.append(knn.score(x_test,y_test))
y_pred= knn.predict(x_test)



categories = [1,2,3,5,6,7]

cm = confusion_matrix(y_test,y_pred)

f , ax = plt.subplots(figsize=(5,5))



sns.heatmap(cm,annot = True,linewidths =0.5,linecolor ="Red",fmt=".0f",ax=ax)

ax.set_xticklabels(categories)

ax.set_yticklabels(categories)

plt.xlabel("y_pred")

plt.ylabel("y_true")



plt.show()
from sklearn.svm import SVC



svm = SVC(random_state=1)

accuracies = cross_val_score(estimator=knn,X = x_train,y = y_train,cv=3)

print("acc = " ,np.mean(accuracies))

print("std= ",np.std(accuracies))


svm.fit(x_train,y_train)

print("acc = " ,svm.score(x_test,y_test))

model.append("Support Vector Machines")

score_list.append(svm.score(x_test,y_test))
y_pred= svm.predict(x_test)



categories = [1,2,3,5,6,7]

cm = confusion_matrix(y_test,y_pred)

f , ax = plt.subplots(figsize=(5,5))



sns.heatmap(cm,annot = True,linewidths =0.5,linecolor ="Red",fmt=".0f",ax=ax)

ax.set_xticklabels(categories)

ax.set_yticklabels(categories)

plt.xlabel("y_pred")

plt.ylabel("y_true")



plt.show()
from sklearn.naive_bayes import GaussianNB



nb=GaussianNB()



accuracies = cross_val_score(estimator=nb,X = x_train,y = y_train,cv=3)

print("acc = ", np.mean(accuracies))

print("std= ",np.std(accuracies))
nb.fit(x_train,y_train)

print("acc = " ,nb.score(x_test,y_test))

model.append("Naive Bayes")

score_list.append(nb.score(x_test,y_test))
y_pred= nb.predict(x_test)



categories = [1,2,3,5,6,7]

cm = confusion_matrix(y_test,y_pred)

f , ax = plt.subplots(figsize=(5,5))



sns.heatmap(cm,annot = True,linewidths =0.5,linecolor ="Red",fmt=".0f",ax=ax)

ax.set_xticklabels(categories)

ax.set_yticklabels(categories)

plt.xlabel("y_pred")

plt.ylabel("y_true")

labels = ["True Neg","False Pos","False Neg","True Pos"]





plt.show()
cv_result = pd.DataFrame({"Scores":score_list, "ML Models":model})

g = sns.barplot("Scores", "ML Models",data=cv_result)

g.set_xlabel("Test Name")

g.set_title("Test Scores")