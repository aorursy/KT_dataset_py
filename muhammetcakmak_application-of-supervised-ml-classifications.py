# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Read the data and assign it as df

df=pd.read_csv("../input/heart.csv")
# Let's have a quick look into data.This code shows first 5 rows and all columns

df.head()
# If there is unknown,missing or unproper data, this codes shows the number of them

# We can also learn about features such as data type of the features

df.info()

# statistical data is important to learn about balance inside or among the features.

df.describe()
# Seaborn countplot gives the number of data in the each class

sns.countplot(x="target", data=df)

# y has target data (clases) such as 1 and 0. 

y = df.target.values

# This means that take target data out from the datasets and assign them to x_data variable

x_data = df.drop(["target"],axis=1)
#Normalization is used to handle with unbalanced features

#This gives the values to the features which range from zero to 1.

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
#The data is splited into two part for training and testing

#Here test_size=0.2 means %20 is splited as test_data

#we need to give any number to random_state in order to split data in the same way when it is reruned

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)
# Build Linear Regression Algorithm

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)

# Here lr.score first predict the y_test and then gives the accuracy

print("test accuracy {}".format(lr.score(x_test,y_test)))



lr_score=lr.score(x_test,y_test)

# Here we use confusion matrix to evaluate the linear regression algorithm

from sklearn.metrics import confusion_matrix

y_prediction = lr.predict(x_test)

y_actual=y_test

cm = confusion_matrix(y_actual,y_prediction)
# Heatmap visualization of cunfusion matrix of Linear regression model

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_prediction")

plt.ylabel("y_actual")

plt.show()

# KNN Classification Model

from sklearn.neighbors import KNeighborsClassifier

k = 3

knn = KNeighborsClassifier(n_neighbors = k)

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

print(" {} nn score: {}".format(k,knn.score(x_test,y_test)))



knn_score = knn.score(x_test,y_test)

# We can determine best k values with plotting k values versus accuracy

# Here we give values to k from 1 to 15 and calculate the accuracy each time,then plot them.

score_list = []

for each in range(1,15):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,15),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()

# Here we use confusion matrix to evaluate the KNN Classification Model

from sklearn.metrics import confusion_matrix

y_prediction = knn.predict(x_test)

y_actual=y_test

cm = confusion_matrix(y_actual,y_prediction)
# Heatmap visualization of cunfusion matrix of the KNN Classification Model

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_prediction")

plt.ylabel("y_actual")

plt.show()
# Build Decision Tree Classification Model

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state = 42)

dt.fit(x_train,y_train)



print("score: ", dt.score(x_test,y_test))



dt_score=dt.score(x_test,y_test)
# Here we use confusion matrix to evaluate the Decision Tree Classification Model

from sklearn.metrics import confusion_matrix

y_prediction = dt.predict(x_test)

y_actual = y_test

cm = confusion_matrix(y_actual,y_prediction)

# Heatmap visualization of cunfusion matrix of the Decision Tree Classification Model

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_prediction")

plt.ylabel("y_actual")

plt.show()
# Visulization of the Decision Tree Classification Model

from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

import pydotplus

dot_data = StringIO()

export_graphviz(dt, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())





# Build Random Forest Classification Model

from sklearn.ensemble import RandomForestClassifier

# n_estimators = 100 means this model will use 100 subsets.

rf = RandomForestClassifier(n_estimators = 100,random_state = 42)

rf.fit(x_train,y_train)

print("random forest algo result: ",rf.score(x_test,y_test))



rf_score = rf.score(x_test,y_test)

# Here we use confusion matrix to evaluate the Random Forest Classification Model

from sklearn.metrics import confusion_matrix

y_prediction = rf.predict(x_test)

y_actual = y_test

cm = confusion_matrix(y_actual,y_prediction)
# Heatmap visualization of cunfusion matrix of the Random Forest Classification Model

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_prediction")

plt.ylabel("y_actual")

plt.show()
# Build Support Vector Machine Model

from sklearn.svm import SVC

svm = SVC(random_state = 42)

svm.fit(x_train,y_train)

# prediction and accuracy 

print("print accuracy of svm algo: ",svm.score(x_test,y_test))



svm_score = svm.score(x_test,y_test)
# Here we use confusion matrix to evaluate the Support Vector Machine Model

from sklearn.metrics import confusion_matrix

y_prediction = svm.predict(x_test)

y_actual = y_test

cm = confusion_matrix(y_actual,y_prediction)
# Heatmap visualization of cunfusion matrix of the Support Vector Machine Model

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_prediction")

plt.ylabel("y_actual")

plt.show()
# Build Naive Bayes Classification Model

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

 

print("print accuracy of naive bayes algo: ",nb.score(x_test,y_test))



nb_score = nb.score(x_test,y_test)

 
# Here we use confusion matrix to evaluate the Support Vector Machine Model

from sklearn.metrics import confusion_matrix

y_prediction = nb.predict(x_test)

y_actual = y_test

cm = confusion_matrix(y_actual,y_prediction)
# Heatmap visualization of cunfusion matrix of the Support Vector Machine Model

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_prediction")

plt.ylabel("y_actual")

plt.show()
class_name = ("Logistic Regression","KNN","Decision Tree","Random Forest","SVM","Naive Bayes")

class_score = (lr_score,knn_score,dt_score,rf_score,svm_score,nb_score)

y_pos= np.arange(len(class_score))

colors = ("red","gray","purple","green","orange","blue")

plt.figure(figsize=(20,12))

plt.bar(y_pos,class_score,color=colors)

plt.xticks(y_pos,class_name,fontsize=20)

plt.yticks(np.arange(0.00, 1.05, step=0.05))

plt.ylabel('Accuracy')

plt.grid()

plt.title(" Confusion Matrix Comparision of the Classes",fontsize=15)

plt.savefig('graph.png')

plt.show()
