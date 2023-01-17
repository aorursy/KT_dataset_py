import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
df=pd.read_csv('../input/heart.csv')

df.head(10)
print (df.target.value_counts())
sns.countplot(x="target", data=df, palette="muted")

plt.savefig('CountOfDisease.png')

plt.show()
disTrue=len(df[df.target==1])

disFalse=len(df[df.target==0])

print ("Percentage of people having heart disease is:  %.2f" %((disTrue*100)/len(df.target)))

print ("Percentage of people not having heart disease is:  %.2f" %((disFalse*100)/len(df.target)))
MalePatient=len(df[df.sex==1])

FemPatient=len(df[df.sex==0])

print ("percentage of Male patients is:  %.2f" %((MalePatient*100)/len(df.sex)) + "%")

print ("percentage of Female patients is:  %.2f" %((FemPatient*100)/len(df.sex)) + "%")
pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.savefig('heartDiseaseAndAges.png')

plt.show()
pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(9,4),color=("green","orange"))

plt.title('Heart Disease Frequency for Sex')

plt.xlabel('Sex : 0-Female, 1-Male')

plt.xticks(rotation=0)

plt.legend(["Haven't disease","Have disease"])

plt.ylabel('Frequency')

plt.savefig('heartDiseaseAndSex.png')

plt.show()
pd.crosstab(df.fbs,df.target).plot(kind="bar",figsize=(15,6),color=['green','red' ])

plt.title('Heart Disease Frequency According To FBS')

plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')

plt.xticks(rotation = 0)

plt.legend(["Haven't Disease", "Have Disease"])

plt.ylabel('Frequency of Disease or Not')

plt.savefig('heartDiseaseAndFBS.png')

plt.show()
plt.scatter(x=df.age[df.target==1],y=df.thalach[df.target==1],color="red")

plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)],color="green")

plt.legend(["Disease", "Not Disease"])

plt.xlabel("Age")

plt.ylabel("Maximum Heart Rate")

plt.savefig('heartrateANDage.png')

plt.show()
x_data = df.drop(['target'], axis = 1)

y_data = df.target.values
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size = 0.2)
x_train=(x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train)).values

x_test=(x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test)).values
logRegr=LogisticRegression()

logRegr.fit(x_train,y_train)

print("Test Accuracy {:.2f}%".format(logRegr.score(x_test,y_test)*100))
from sklearn.neighbors import KNeighborsClassifier

scoreList = []

for i in range(1,15):

    knn = KNeighborsClassifier(n_neighbors = i)

    knn.fit(x_train, y_train)

    scoreList.append(knn.score(x_test, y_test))

    

plt.plot(range(1,15), scoreList)

plt.xticks(np.arange(1,15,1))

plt.xlabel("K value")

plt.ylabel("Score")

plt.show()





print("Maximum KNN Score is:  %.2f" %((max(scoreList))*100))
from sklearn import svm

clf = svm.SVC(kernel='rbf')

clf.fit(x_train, y_train) 
print("Test Accuracy of SVM Algorithm:  %.2f" %(clf.score(x_test,y_test)*100))
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(x_train, y_train)

print("Decision Tree Test Accuracy:  %.2f" %(dtc.score(x_test, y_test)*100))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train, y_train)

print("Accuracy of Naive Bayes:  %.2f" %(nb.score(x_test,y_test)*100))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)

rf.fit(x_train, y_train)

print("Random Forest Algorithm Accuracy Score : %.2f" %(rf.score(x_test,y_test)*100))
methods = ["Logistic Regression", "KNN", "SVM", "Decision Tree", "Naive Bayes", "Random Forest"]

accuracy = [83.61, 83.61, 78.69, 75.41, 75.41, 85.25]

colors = ["violet", "indigo", "blue", "green","yellow","orange"]



sns.set_style("whitegrid")

plt.figure(figsize=(16,5))

plt.yticks(np.arange(0,100,10))

plt.ylabel("Accuracy %")

plt.xlabel("Algorithms")

sns.barplot(x=methods, y=accuracy, palette=colors)

plt.show()
# Predicted values

y_head_lr = logRegr.predict(x_test)

knn7 = KNeighborsClassifier(n_neighbors = 7)

knn7.fit(x_train, y_train)

y_head_knn = knn7.predict(x_test)

y_head_svm = clf.predict(x_test)

y_head_nb = nb.predict(x_test)

y_head_dtc = dtc.predict(x_test)

y_head_rf = rf.predict(x_test)
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

sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False)



plt.subplot(2,3,2)

plt.title("K Nearest Neighbors Confusion Matrix")

sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False)



plt.subplot(2,3,3)

plt.title("Support Vector Machine Confusion Matrix")

sns.heatmap(cm_svm,annot=True,cmap="Blues",fmt="d",cbar=False)



plt.subplot(2,3,4)

plt.title("Decision Tree Classifier Confusion Matrix")

sns.heatmap(cm_dtc,annot=True,cmap="Blues",fmt="d",cbar=False)



plt.subplot(2,3,5)

plt.title("Naive Bayes Confusion Matrix")

sns.heatmap(cm_nb,annot=True,cmap="Blues",fmt="d",cbar=False)



plt.subplot(2,3,6)

plt.title("Random Forest Confusion Matrix")

sns.heatmap(cm_rf,annot=True,cmap="Blues",fmt="d",cbar=False)



plt.show()