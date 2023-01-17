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
#load-data

data = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
data.head()
data.info()
data.columns
data.describe()
#correlation 

data.corr()
#correlation map



f,ax = plt.subplots(figsize=(10,10))

sns.heatmap(data.corr(),annot = True , linewidth = .5, fmt = '.2f',ax = ax)

plt.show()
#counts of quality types

data.quality.value_counts()
#visualization



#pie_graph

plt.figure(1, figsize=(8,8))

data.quality.value_counts().plot.pie(autopct="%1.1f%%")
sns.countplot(x="quality", data=data)

data.loc[:,'quality'].value_counts()
#scatter plot

data.plot(kind = "scatter" , x = "alcohol" , y = "quality" , color = "red")

plt.xlabel("alcohol")

plt.ylabel("quality")

plt.title("alcohol-quality Scatter Plot")

plt.show()
#line plot 

data.sulphates.plot(kind = "line" , color = "red" , alpha = 0.5 , grid = True , linestyle = ":",label = "sulphates")

data.chlorides.plot(kind = "line" , color = "green" , alpha = 0.5 , linestyle = "-",label = "chlorides")

plt.legend(loc = "upper right")

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.title("sulphates-chlorides Scatter Plot")

plt.show()
y = data.quality.values

x_data = data.drop(["quality"],axis = 1)

#normalization

x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))
#train-test split

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2,random_state = 42 )
#predict with 3 neighbor

from sklearn.neighbors import KNeighborsClassifier 

knn = KNeighborsClassifier(n_neighbors = 10) #n_neighbors = k

knn.fit(x_train,y_train)

print("{} nn score : {}".format(3,knn.score(x_test,y_test)))
#finding best accuracy for n-neighbor

score_list = []

for each in range(1,15):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,15),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")    

plt.show()
#confusion matrix for KNN

from sklearn.metrics import confusion_matrix



y_pred = knn.predict(x_test)

y_true = y_test

cm = confusion_matrix(y_true,y_pred)

f,ax = plt.subplots(figsize = (5,5))

sns.heatmap(cm,annot = True,linewidth = 0.5,linecolor = "red",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)



print("score : {}".format(dt.score(x_test,y_test)))
#confusion matrix for DecisionTree

from sklearn.metrics import confusion_matrix



y_pred = dt.predict(x_test)

y_true = y_test

cm = confusion_matrix(y_true,y_pred)

f,ax = plt.subplots(figsize = (5,5))

sns.heatmap(cm,annot = True,linewidth = 0.5,linecolor = "red",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100 , random_state = 1)

rf.fit(x_train,y_train)



print("score : {}".format(rf.score(x_test,y_test)))
#finding best accuracy for n_estimators

score_list = []

for each in range(1,50):

    rf2 = RandomForestClassifier(n_estimators = each)

    rf2.fit(x_train,y_train)

    score_list.append(rf2.score(x_test,y_test))

    

plt.plot(range(1,50),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")    

plt.show()
#confusion matrix for RandomForestClassification

from sklearn.metrics import confusion_matrix



y_pred = rf.predict(x_test)

y_true = y_test

cm = confusion_matrix(y_true,y_pred)

f,ax = plt.subplots(figsize = (5,5))

sns.heatmap(cm,annot = True,linewidth = 0.5,linecolor = "red",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.svm import SVC



svm = SVC(random_state = 1)

svm.fit(x_train,y_train)

print("Accuracy of svm :  {}".format(svm.score(x_test,y_test)))
#confusion matrix for SVM

from sklearn.metrics import confusion_matrix



y_pred = svm.predict(x_test)

y_true = y_test

cm = confusion_matrix(y_true,y_pred)

f,ax = plt.subplots(figsize = (5,5))

sns.heatmap(cm,annot = True,linewidth = 0.5,linecolor = "red",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print("Accuracy of naive-bayes : {}".format(nb.score(x_test,y_test)))
#confusion matrix for Naive-Bayes

from sklearn.metrics import confusion_matrix



y_pred = nb.predict(x_test)

y_true = y_test

cm = confusion_matrix(y_true,y_pred)

f,ax = plt.subplots(figsize = (5,5))

sns.heatmap(cm,annot = True,linewidth = 0.5,linecolor = "red",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()