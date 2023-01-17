import numpy as np
import pandas as pd
import seaborn as sns

train_data=pd.read_csv('Classification_train.csv')
test_data=pd.read_csv('Classification_test.csv')

train_data.count()
test_data.count()
print(train_data["Embarked"].value_counts())

train_data= train_data.drop(['Cabin'], axis=1)

train_data["Age"]=train_data["Age"].fillna(train_data["Age"].median())
train_data["Embarked"] =train_data["Embarked"].fillna("S")

train_data = train_data.dropna()
train_data.count()
print(test_data["Embarked"].value_counts())

test_data= test_data.drop(['Cabin'], axis=1)

test_data["Age"]=test_data["Age"].fillna(test_data["Age"].median())
test_data["Embarked"] =test_data["Embarked"].fillna("S")

test_data=test_data.dropna()
test_data.count()
train_data['Alone']=np.where((train_data["SibSp"]+train_data["Parch"])>0,0,1)
train_data= train_data.drop(['SibSp','Parch'], axis=1)
train=pd.get_dummies(train_data,columns=["Pclass","Embarked","Sex"])
train=train.drop(['Sex_female','PassengerId','Name','Ticket'],axis=1)

test_data['Alone']=np.where((test_data["SibSp"]+test_data["Parch"])>0,0,1)
test_data= test_data.drop(['SibSp','Parch'], axis=1)

test=pd.get_dummies(test_data,columns=["Pclass","Embarked","Sex"])
test=test.drop(['Sex_female','PassengerId','Name','Ticket'],axis=1)

x=train.iloc[:,1:11].values

y=train.iloc[:,0:1].values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.75,random_state=42)
xtrain.shape,xtest.shape,ytrain.shape,ytest.shape
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(xtrain,ytrain)
pred=knn.predict(xtest)
from sklearn.metrics import classification_report,confusion_matrix
cm=confusion_matrix(ytest,pred)
print(cm)
from sklearn.metrics import classification_report
print(classification_report(ytest,pred))

pred=knn.predict(test)

test['Survived']=pred
test['PassengerId']=test_data['PassengerId']
submission=test[['PassengerId','Survived']]
submission.to_csv("submission.csv")
aaaa=pd.read_csv('submission.csv')
print(aaaa['Survived'].value_counts())


