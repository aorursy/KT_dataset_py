import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
data=pd.read_csv("../input/Iris.csv")
data.head()
data.isnull().sum()
data=data.drop("Id",axis=1)
data.head()
data.shape
type(data)
data.info()
data.describe()
#its for overall data set not for specific species

data.plot(kind="box",sharex=False,sharey=False,figsize=(10,10))

plt.show()
#its for overall data set not for specific species

data.hist(edgecolor="red",linewidth=2,figsize=(10,10))

plt.show()
#its for individual species

data.boxplot(by="Species",figsize=(10,10))

plt.show()
plt.bar(data["Species"],data["SepalLengthCm"])

plt.show()
plt.bar(data["Species"],data["SepalWidthCm"])

plt.show()
plt.bar(data["Species"],data["PetalLengthCm"])

plt.show()
plt.bar(data["Species"],data["PetalWidthCm"])

plt.show()
data.plot(figsize=(10,10))

plt.show()
sns.violinplot(data=data,x="Species",y="SepalLengthCm")

plt.show()
sns.violinplot(data=data,x="Species",y="SepalWidthCm")

plt.show()
sns.violinplot(data=data,x="Species",y="PetalLengthCm")

plt.show()
sns.violinplot(data=data,x="Species",y="PetalWidthCm")

plt.show()
from pandas.plotting import scatter_matrix

scatter_matrix(data,figsize=(20,15))

plt.show()
sns.pairplot(data,hue="Species")

plt.show()
#deivde the data into dependent and independent variables

x=data.iloc[:,:-1].values

y=data.iloc[:,-1].values
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#Logistic_regression

from sklearn.linear_model import LogisticRegression

L_class=LogisticRegression()

L_class.fit(x_train,y_train)
y_L_class=L_class.predict(x_test)
y_L_class
y_test
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

cm=confusion_matrix(y_test,y_L_class)
#prediction

print(cm)
c_r=classification_report(y_test,y_L_class)
print(c_r)
accuracy_logistic=accuracy_score(y_test,y_L_class)

print(accuracy_score(y_test,y_L_class))
#or accuracy can be calculated as

correct=cm[0][0]+cm[1][1]+cm[2][2]

total=cm.sum()

print("accuracy:-",correct/total)
#try with KNN

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=10,metric="minkowski",p=2)

"""metric is minkowski that means we are using ecludien distance

p=2 beacuse when we calculate distance between two point it's cordinates have power 2"""

knn.fit(x_train,y_train)

y_pred_knn=knn.predict(x_test)

accuracy_knn=accuracy_score(y_test,y_pred_knn)
print(y_pred_knn)
print(confusion_matrix(y_test,y_pred_knn))
print(classification_report(y_test,y_pred_knn))
#try with svm

from sklearn.svm import SVC

svc=SVC(kernel="linear")

svc.fit(x_train,y_train)

y_pred_svc=svc.predict(x_test)

#check accuracy matrix

print(confusion_matrix(y_test,y_pred_svc))

#classification_report

print(classification_report(y_test,y_pred_svc))

accuracy_svc=accuracy_score(y_test,y_pred_svc)
#try with kernel svm

from sklearn.svm import SVC

ksvc=SVC(kernel="rbf")

ksvc.fit(x_train,y_train)

y_pred_ksvc=ksvc.predict(x_test)

#check accuracy matrix

print(confusion_matrix(y_test,y_pred_ksvc))

#classification_report

print(classification_report(y_test,y_pred_ksvc))

accuracy_ksvc=accuracy_score(y_test,y_pred_ksvc)
#try with naive_bayes

from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

nb.fit(x_train,y_train)

y_pred_nb=nb.predict(x_test)

print(confusion_matrix(y_test,y_pred_nb))

print(classification_report(y_test,y_pred_nb))

accuracy_nb=accuracy_score(y_test,y_pred_nb)
#try with decision_tree

from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier(criterion="entropy")

dtc.fit(x_train,y_train)

y_pred_dtc=dtc.predict(x_test)

print(confusion_matrix(y_test,y_pred_dtc))

print(classification_report(y_test,y_pred_dtc))

accuracy_dtc=accuracy_score(y_test,y_pred_dtc)
#try with Random forest classifier

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(criterion="entropy",n_estimators=10)

rfc.fit(x_train,y_train)

y_pred_rfc=rfc.predict(x_test)

print(confusion_matrix(y_test,y_pred_rfc))

print(classification_report(y_test,y_pred_rfc))

accuracy_rfc=accuracy_score(y_test,y_pred_rfc)
acc_of_models=[accuracy_logistic,accuracy_knn,accuracy_svc,accuracy_ksvc,accuracy_nb,accuracy_dtc,accuracy_rfc]
acc_of_models