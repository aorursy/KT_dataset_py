# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/heart.csv")

df.head(n=4)
df.target.value_counts()

df.sex.value_counts()
nodisease=len(df[df.target==0])

disease=len(df[df.target==1])
print("total percentage of disease:{:2f}%,".format((disease/(len(df.target))*100)))

print("total percentage with no disease:{:2f}%,".format((nodisease/(len(df.target))*100)))
df.groupby('target').mean()
a=pd.get_dummies(df['cp'], prefix="cp")

b=pd.get_dummies(df['thal'], prefix="thal")

c=pd.get_dummies(df['slope'], prefix='slope')

frames=[df,a,b,c]

df=pd.concat(frames,axis=1)

df=df.drop(columns=['cp','thal','slope'])

df.head()
y=df.target.values

x_data=df.drop(['target'],axis=1)
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
x_train = x_train.T

y_train = y_train.T

x_test = x_test.T

y_test = y_test.T
lr = LogisticRegression()

lr.fit(x_train.T,y_train.T)

print("Test Accuracy {:.2f}%".format(lr.score(x_test.T,y_test.T)*100))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)  # n_neighbors means k

knn.fit(x_train.T, y_train.T)

prediction = knn.predict(x_test.T)



print("{} NN Score: {:.2f}%".format(2, knn.score(x_test.T, y_test.T)*100))
scoreList = []

for i in range(1,20):

    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k

    knn2.fit(x_train.T, y_train.T)

    scoreList.append(knn2.score(x_test.T, y_test.T))

    

plt.plot(range(1,20), scoreList)

plt.xticks(np.arange(1,20,1))

plt.xlabel("K value")

plt.ylabel("Score")

plt.show()





print("Maximum KNN Score is {:.2f}%".format((max(scoreList))*100))
svm = SVC(random_state = 1)

svm.fit(x_train.T, y_train.T)

print("Test Accuracy of SVM Algorithm: {:.2f}%".format(svm.score(x_test.T,y_test.T)*100))
nb = GaussianNB()

nb.fit(x_train.T, y_train.T)

print("Accuracy of Naive Bayes: {:.2f}%".format(nb.score(x_test.T,y_test.T)*100))

from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier()

dtc.fit(x_train.T, y_train.T)

print("Accuracy of Decision Tree Classifier:{:.2f}%".format(nb.score(x_test.T,y_test.T)*100))
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators = 1000, random_state = 1)

rfc.fit(x_train.T,y_train.T)

print("Accuracy of Random Forest Classifier:{:.2f}%".format(nb.score(x_test.T,y_test.T)*100))
methods = ["Logistic Regression", "KNN", "SVM", "Naive Bayes", "Decision Tree", "Random Forest"]

accuracy = [86.89, 88.52, 86.89, 86.89, 78.69, 88.52]

colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE"]



sns.set_style("whitegrid")

plt.figure(figsize=(16,5))

plt.yticks(np.arange(0,100,10))

plt.ylabel("Accuracy %")

plt.xlabel("Algorithms")

sns.barplot(x=methods, y=accuracy, palette=colors)

plt.show()
y_head_lr = lr.predict(x_test.T)

knn3 = KNeighborsClassifier(n_neighbors = 7)

knn3.fit(x_train.T, y_train.T)

y_head_knn = knn3.predict(x_test.T)

y_head_svm = svm.predict(x_test.T)

y_head_nb = nb.predict(x_test.T)

y_head_dtc = dtc.predict(x_test.T)

y_head_rfc = rfc.predict(x_test.T)



from sklearn.metrics import confusion_matrix



cm_lr = confusion_matrix(y_test,y_head_lr)

cm_knn = confusion_matrix(y_test,y_head_knn)

cm_svm = confusion_matrix(y_test,y_head_svm)

cm_nb = confusion_matrix(y_test,y_head_nb)

cm_dtc = confusion_matrix(y_test,y_head_dtc)

cm_rfc = confusion_matrix(y_test,y_head_rfc)



plt.figure(figsize=(24,12))



plt.suptitle("Confusion Matrixes",fontsize=24)

plt.subplots_adjust(wspace = 0.4, hspace= 0.4)



plt.subplot(2,3,6)

plt.title("Logistic Regression Confusion Matrix")

sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False)



plt.subplot(2,3,1)

plt.title("K Nearest Neighbors Confusion Matrix")

sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False)



plt.subplot(2,3,3)

plt.title("Support Vector Machine Confusion Matrix")

sns.heatmap(cm_svm,annot=True,cmap="Blues",fmt="d",cbar=False)



plt.subplot(2,3,4)

plt.title("Naive Bayes Confusion Matrix")

sns.heatmap(cm_nb,annot=True,cmap="Blues",fmt="d",cbar=False)



plt.subplot(2,3,5)

plt.title("Decision Tree Classifier Confusion Matrix")

sns.heatmap(cm_dtc,annot=True,cmap="Blues",fmt="d",cbar=False)



plt.subplot(2,3,2)

plt.title("Random Forest Confusion Matrix")

sns.heatmap(cm_rfc,annot=True,cmap="Blues",fmt="d",cbar=False)



plt.show()