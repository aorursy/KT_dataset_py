# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv),

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/lower-back-pain-symptoms-datasetlabelled/Dataset_spine.csv")
data.head()
data.info()
data.Class_att=[1 if each =="Abnormal" else 0 for each in data.Class_att]
y=data.Class_att.values

x_data=data.drop(["Class_att"],axis=1)



x=(x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
from sklearn.linear_model import LogisticRegression



lr=LogisticRegression()

lr.fit(x_train,y_train)



print("Logistic Regression score:",lr.score(x_test,y_test))
from sklearn.neighbors import KNeighborsClassifier



knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)

prediction=knn.predict(x_test)



print("{} nn score: {}".format(3,knn.score(x_test,y_test)))



score_list=[]



for each in range(1,15):

    knn=KNeighborsClassifier(n_neighbors=each)

    knn.fit(x_train,y_train)

    score_list.append(knn.score(x_test,y_test))



plt.plot(range(1,15),score_list)

plt.xlabel("k")

plt.ylabel("Scores")

plt.show()



knn=KNeighborsClassifier(n_neighbors=7)

knn.fit(x_train,y_train)



print("according to the graph the max score value is at k = 7 and score is :",knn.score(x_test,y_test))
from sklearn.svm import SVC



svm=SVC(random_state=42)

svm.fit(x_train,y_train)



print("svm score:",svm.score(x_test,y_test))
from sklearn.naive_bayes import GaussianNB



nb=GaussianNB()

nb.fit(x_train,y_train)



print("nb score:",nb.score(x_test,y_test))
from sklearn.tree import DecisionTreeClassifier



df=DecisionTreeClassifier(random_state=42)

df.fit(x_train,y_train)



print("Desicion Tree Classification score:",df.score(x_test,y_test))
from sklearn.ensemble import RandomForestClassifier



rf=RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(x_train,y_train)



print("Random Forest Classification score:",rf.score(x_test,y_test))
y_pred=rf.predict(x_test)

y_true=y_test



from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_true,y_pred)



f,ax=plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
class_method=[lr,knn,svm,nb,df,rf]

method_acc=[]

method_name=["lr","knn","svm","nb","df","rf"]



for each in class_method:

    method_acc.append(each.score(x_test,y_test)*100)

    

plt.plot(method_name,method_acc)

plt.xlabel("Methods of Classifiaction")

plt.ylabel("Accuracies of Methods (%)")

plt.show()