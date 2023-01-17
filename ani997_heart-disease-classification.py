# Special thanks to Heart Disease - Classifications (Machine Learning) for providing this kernel.

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# reading our data

df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")

# Displaying First 5 rows of our data

df.head()
# have disease-1,no disease-0

df.target.value_counts()

sns.countplot(x="target", data=df, palette="bwr")

plt.show()
countNoDisease = len(df[df.target == 0])

countHaveDisease = len(df[df.target == 1])

print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target))*100)))

print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100)))
# sex, female-0, male-1

sns.countplot(x='sex', data=df, palette="mako_r")

plt.xlabel("Sex (0 = female, 1= male)")

plt.show()

countFemale = len(df[df.sex == 0])

countMale = len(df[df.sex == 1])

print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(df.sex))*100)))

print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(df.sex))*100)))
df.groupby('target').mean()
# Heart disease frequency vs Age

pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(15,6))

plt.title('Heart Disease Frequency for Different Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.show()
# Heart disease frequency vs Sex

pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(15,6))

plt.title('Heart Disease Frequency as per Sex')

plt.xlabel('Sex(0=female,1=male)')

plt.ylabel('Frequency')

plt.legend(["No Disease","Have Disease"])

plt.show()
# Scatter plot of Max Heart Rate vs Age

plt.scatter(x=df.age[df.target==1],y=df.thalach[(df.target==1)], c='red')

plt.scatter(x=df.age[df.target==0],y=df.thalach[(df.target==0)], c='blue')

plt.legend(["Disease","No Disease"])

plt.xlabel("Age")

plt.ylabel("Max Heart Rate")

plt.show()

# Heart disease frequency vs Peak Excercise

pd.crosstab(df.slope,df.target).plot(kind="bar",figsize=(15,6),color=['#DAF7A6','#FF5733' ])

plt.title('Heart Disease Frequency for Slope')

plt.xlabel('The Slope of The Peak Exercise ST Segment ')

plt.xticks(rotation = 0)

plt.ylabel('Frequency')

plt.show()
# Heart disease frequency vs Fasting Blood Sugar>120

pd.crosstab(df.fbs,df.target).plot(kind="bar",figsize=(15,6),color=['#FFC300','#581845' ])

plt.title('Heart Disease Frequency According To FBS')

plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')

plt.xticks(rotation = 0)

plt.legend(["Haven't Disease", "Have Disease"])

plt.ylabel('Frequency of Disease or Not')

plt.show()
# Heart disease frequency vs Chest Pain Type

pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(15,6),color=['#11A5AA','#AA1190' ])

plt.title('Heart Disease Frequency According To Chest Pain Type')

plt.xlabel('Chest Pain Type')

plt.xticks(rotation = 0)

plt.ylabel('Frequency of Disease or Not')

plt.show()
# Creation of Dummy Variables

a = pd.get_dummies(df['cp'],prefix = "cp")

b = pd.get_dummies(df['thal'],prefix = "thal")

c = pd.get_dummies(df['slope'],prefix = "slope")
frames = [df,a,b,c]

df = pd.concat(frames, axis = 1)

df.head()
df = df.drop(columns = ['cp', 'thal', 'slope'])

df.head()
y = df.target.values

x_data = df.drop(['target'], axis = 1)
# Normalize X

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
#transpose matrices

x_train = x_train.T

y_train = y_train.T

x_test = x_test.T

y_test = y_test.T
accuracies = {}



#sklearn logistic regression

lr = LogisticRegression()

lr.fit(x_train.T,y_train.T)

acc = lr.score(x_test.T,y_test.T)*100

accuracies['Logistic Regression'] = acc

print("Test Accuracy {:.2f}%".format(acc))

# KNN Model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)  # n_neighbors means k

knn.fit(x_train.T, y_train.T)

acc = knn.score(x_test.T,y_test.T)*100

accuracies['KNN'] = acc

print("Test Accuracy of KNN Algorithm: {:.2f}%".format(acc))
# SVM Algorithm

from sklearn.svm import SVC

svm = SVC(random_state = 1)

svm.fit(x_train.T, y_train.T)



acc = svm.score(x_test.T,y_test.T)*100

accuracies['SVM'] = acc

print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc))
#Naive Bayes Classifier

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train.T, y_train.T)



acc = nb.score(x_test.T,y_test.T)*100

accuracies['Naive Bayes'] = acc

print("Accuracy of Naive Bayes: {:.2f}%".format(acc))
#Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(x_train.T, y_train.T)



acc = dtc.score(x_test.T, y_test.T)*100

accuracies['Decision Tree'] = acc

print("Decision Tree Test Accuracy {:.2f}%".format(acc))
# Random Forest Classification

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)

rf.fit(x_train.T, y_train.T)



acc = rf.score(x_test.T,y_test.T)*100

accuracies['Random Forest'] = acc

print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))
#Comparing performance of different algorithms

colors = ["purple", "green", "orange", "magenta","red","blue"]



sns.set_style("whitegrid")

plt.figure(figsize=(16,5))

plt.yticks(np.arange(0,100,10))

plt.ylabel("Accuracy %")

plt.xlabel("Algorithms")

sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)

plt.show()
#Confusion Matrix

# Predicted values

y_head_lr = lr.predict(x_test.T)

y_head_knn = knn.predict(x_test.T)

y_head_svm = svm.predict(x_test.T)

y_head_nb = nb.predict(x_test.T)

y_head_dtc = dtc.predict(x_test.T)

y_head_rf = rf.predict(x_test.T)
from sklearn.metrics import confusion_matrix



cm_lr = confusion_matrix(y_test,y_head_lr)

cm_knn = confusion_matrix(y_test,y_head_knn)

cm_svm = confusion_matrix(y_test,y_head_svm)

cm_nb = confusion_matrix(y_test,y_head_nb)

cm_dtc = confusion_matrix(y_test,y_head_dtc)

cm_rf = confusion_matrix(y_test,y_head_rf)
plt.figure(figsize=(24,12))



plt.suptitle("Confusion Matrixes",fontsize=24)

plt.subplots_adjust(wspace = 0.4, hspace= 0.4)



plt.subplot(2,3,1)

plt.title("Logistic Regression Confusion Matrix")

sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,2)

plt.title("K Nearest Neighbors Confusion Matrix")

sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,3)

plt.title("Support Vector Machine Confusion Matrix")

sns.heatmap(cm_svm,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,4)

plt.title("Naive Bayes Confusion Matrix")

sns.heatmap(cm_nb,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,5)

plt.title("Decision Tree Classifier Confusion Matrix")

sns.heatmap(cm_dtc,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,6)

plt.title("Random Forest Confusion Matrix")

sns.heatmap(cm_rf,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.show()