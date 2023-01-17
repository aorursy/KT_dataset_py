# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/diabetes.csv")
data.sample(5)
data.info()
sns.countplot(data.Outcome)

plt.title("Diabates Status",color="black",fontsize=15)
data.Outcome.value_counts()
f,ax=plt.subplots(figsize=(15,15))

sns.heatmap(data.corr(),annot=True,linecolor="blue",fmt=".2f",ax=ax)

plt.show()
g = sns.pairplot(data, hue="Outcome",palette="Set2",diag_kind = "kde",kind = "scatter")
x=data.drop(["Outcome"],axis=1)

y=data.Outcome.values.reshape(-1,1)
#Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x = sc.fit_transform(x)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)

lr_prediction= lr.predict(x_test)



lr_cm = confusion_matrix(y_test,lr_prediction)

print("Logistic Regression Accuracy :",lr.score(x_test, y_test))
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 16)

knn.fit(x_train,y_train)

knn_prediction= knn.predict(x_test)



knn_cm = confusion_matrix(y_test,knn_prediction)

print("KNN Classification Accuracy :",knn.score(x_test,y_test))
score_list = []

for each in range(1,25):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,25),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()



print("Best accuracy is {} with K = {}".format(np.max(score_list),1+score_list.index(np.max(score_list))))
from sklearn.svm import SVC



svm=SVC(random_state=1)

svm.fit(x_train,y_train)

svm_prediction= svm.predict(x_test)



svm_cm = confusion_matrix(y_test,svm_prediction)

print("Support Vector Classification Accuracy :",svm.score(x_test,y_test))
from sklearn.naive_bayes import GaussianNB



nb=GaussianNB()

nb.fit(x_train,y_train)

nb_prediction= nb.predict(x_test)



nb_cm = confusion_matrix(y_test,nb_prediction)

print("Naive Bayes Classification Accuracy :",nb.score(x_test,y_test))
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()

dt.fit(x_train,y_train)

dt_prediction= dt.predict(x_test)



dt_cm = confusion_matrix(y_test,dt_prediction)

print("Decision Tree Classification Accuracy :",dt.score(x_test,y_test))
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()

rf.fit(x_train,y_train)

rf_prediction= rf.predict(x_test)



rf_cm = confusion_matrix(y_test,rf_prediction)

print("Random Forest Classification Accuracy :",rf.score(x_test,y_test))
fig = plt.figure(figsize=(15,15))



ax1 = fig.add_subplot(3, 3, 1) # row, column, position

ax1.set_title('Logistic Regression Classification')



ax2 = fig.add_subplot(3, 3, 2)

ax2.set_title('KNN Classification')



ax3 = fig.add_subplot(3, 3, 3)

ax3.set_title('SVM Classification')



ax4 = fig.add_subplot(3, 3, 4)

ax4.set_title('Naive Bayes Classification')



ax5 = fig.add_subplot(3, 3, 5)

ax5.set_title('Decision Tree Classification')



ax6 = fig.add_subplot(3, 3, 6)

ax6.set_title('Random Forest Classification')





sns.heatmap(data=lr_cm, annot=True, linewidth=0.5, linecolor='mintcream', fmt='.0f', ax=ax1, cmap='RdGy')

sns.heatmap(data=knn_cm, annot=True, linewidth=0.5, linecolor='mintcream', fmt='.0f', ax=ax2, cmap='RdGy')   

sns.heatmap(data=svm_cm, annot=True, linewidth=0.5, linecolor='mintcream', fmt='.0f', ax=ax3, cmap='RdGy')

sns.heatmap(data=nb_cm, annot=True, linewidth=0.5, linecolor='mintcream', fmt='.0f', ax=ax4, cmap='RdGy')

sns.heatmap(data=dt_cm, annot=True, linewidth=0.5, linecolor='mintcream', fmt='.0f', ax=ax5, cmap='RdGy')

sns.heatmap(data=rf_cm, annot=True, linewidth=0.5, linecolor='mintcream', fmt='.0f', ax=ax6, cmap='RdGy')

plt.show()