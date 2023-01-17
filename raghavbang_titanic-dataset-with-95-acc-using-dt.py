import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC,LinearSVC

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
train=pd.read_csv("../input/titanic/train.csv")

test=pd.read_csv("../input/titanic/test.csv")

train.columns

train.describe()

test.describe()

train.info()

a=train["Cabin"].isnull().sum()

train=train.drop(["PassengerId","Name","Cabin","Ticket"],axis=1)

test=test.drop(["PassengerId","Name","Cabin","Ticket"],axis=1)

plt.bar(train["Sex"].unique(),train["Sex"].value_counts())
train["Sex"]=pd.get_dummies(train["Sex"],drop_first=True)

test["Sex"]=pd.get_dummies(test["Sex"],drop_first=True)

train.isnull().sum()

test.isnull().sum()

train["Age"].describe()

train["Age"].fillna("29",inplace=True)

test["Age"].fillna("29",inplace=True)

train["Embarked"].value_counts()

train["Embarked"].fillna("S",inplace=True)

a=train["Sex"].value_counts()

test["Fare"].describe()

test["Fare"].fillna("14",inplace=True)

men_lived=np.where(train["Sex"]==1,train["Survived"],0)

men_lived=sum(men_lived)



female_lived=np.where(train["Sex"]==0,train["Survived"],0)

female_lived=sum(female_lived)



men_died=sum(train["Sex"])-men_lived

female_died= len(train["Sex"])- (men_lived+men_died+female_lived)
x=["Death","Survied"]

plt.bar(x,train["Survived"].value_counts(),width=0.35)

plt.show()
lived=[female_lived,men_lived]

dead=[female_died,men_died]

labels=["female","male"]

x=np.arange(len(labels))

width=0.2

plt.bar(x-width/2,train["Sex"].value_counts(),width=0.2,label="Total")

plt.bar(x+width/1.9,lived,width=0.2,label="Survived")

plt.bar(x+width/0.64,dead,width=0.2,label="dead")

plt.ylabel("Number")

plt.title("Dead and survied")

plt.xticks(x,labels)

plt.legend(["Total","Survived","Dead"])

plt.show()
plt.figure(figsize=(5,5))

plt.bar(train["Pclass"].unique(),train["Pclass"].value_counts(),color="Red")

plt.xticks(train["Pclass"].unique())

plt.ylabel("Probability Count")

plt.xlabel("Pclass")

plt.show()
plt1=train[['Pclass', 'Survived']].groupby('Pclass').mean().Survived.plot(kind='bar',color="green")

plt1.set_xlabel("Plcass")

plt1.set_ylabel("Survival Probability")

plt.show()
sns.catplot('Sex', col='Pclass',data=train, kind = 'count')

plt.xticks(x,labels)

plt.show()
plt1=train[['Embarked', 'Survived']].groupby('Embarked').mean().Survived.plot(kind='bar')

plt1.set_ylabel("Survival Probability")

plt.show()
plt.bar(train["Embarked"].unique(),train["Embarked"].value_counts(),color="RED")

plt.ylabel("Embarked Count")

plt.show()
sns.catplot('Embarked', col='Pclass',data=train, kind = 'count')

plt.show()
test.info()
train.info()
train["Age"]=train["Age"].astype(int)

test["Age"]=test["Age"].astype(int)

train["Fare"]=train["Fare"].astype(int)

test["Fare"]=test["Fare"].astype(int)

train["Embarked"]=pd.get_dummies(train["Embarked"],drop_first=True)

test["Embarked"]=pd.get_dummies(test["Embarked"],drop_first=True)
train["Total_number"]=train["SibSp"]+train["Parch"]+1

test["Total_number"]=test["SibSp"]+test["Parch"]+1

train["Fareperhead"]=train["Fare"]/(train["Total_number"])

test["Fareperhead"]=test["Fare"]/(test["Total_number"])

train.describe()
train.info()
train["Fareperhead"]=train["Fareperhead"].astype(int)

test["Fareperhead"]=test["Fareperhead"].astype(int)
X_train = train.drop("Survived", axis=1)

Y_train = train["Survived"]

X_test  = test.copy()
X_train=preprocessing.scale(X_train)

random_forest = RandomForestClassifier(n_estimators=30)

random_forest.fit(X_train, Y_train)

Y_predict = random_forest.predict(X_test)

acc_rand=random_forest.score(X_train, Y_train)

Y1_predict=random_forest.predict(X_train)

print("Confusion Matrix")

print(confusion_matrix(Y_train, Y1_predict))

print ('Accuracy Score :',accuracy_score(Y_train, Y1_predict) )

print ('Report : ')

print (classification_report(Y_train, Y1_predict) )
logreg = LogisticRegression(solver='lbfgs')

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log=logreg.score(X_train, Y_train)

Y1_predict=logreg.predict(X_train)

print("Confusion Matrix")

print(confusion_matrix(Y_train, Y1_predict))

print ('Accuracy Score :',accuracy_score(Y_train, Y1_predict) )

print ('Report : ')

print (classification_report(Y_train, Y1_predict) )
decision_tree = DecisionTreeClassifier() 

decision_tree.fit(X_train, Y_train)  

Y_pred = decision_tree.predict(X_test)  

acc_dec=decision_tree.score(X_train, Y_train)

Y1_predict=decision_tree.predict(X_train)

print("Confusion Matrix")

print(confusion_matrix(Y_train, Y1_predict))

print ('Accuracy Score :',accuracy_score(Y_train, Y1_predict) )

print ('Report : ')

print (classification_report(Y_train, Y1_predict) )
classifier = KNeighborsClassifier(n_neighbors = 1)

classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)  

acc_knn=classifier.score(X_train, Y_train)

Y1_predict=classifier.predict(X_train)

print("Confusion Matrix")

print(confusion_matrix(Y_train, Y1_predict))

print ('Accuracy Score :',accuracy_score(Y_train, Y1_predict) )

print ('Report : ')

print (classification_report(Y_train, Y1_predict) )
linear_svc=LinearSVC(max_iter=10000)

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_svm=linear_svc.score(X_train, Y_train)

Y1_predict=linear_svc.predict(X_train)

print("Confusion Matrix")

print(confusion_matrix(Y_train, Y1_predict))

print ('Accuracy Score :',accuracy_score(Y_train, Y1_predict) )

print ('Report : ')

print (classification_report(Y_train, Y1_predict) )
model=["RandomForestClass","Logistic REG","KNN","Decision Tree","SVM"]

Acc=[acc_rand*100,acc_log*100,acc_knn*100,acc_dec*100,acc_svm*100]

plt.barh(model,Acc)

plt.show()
