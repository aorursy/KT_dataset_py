

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import confusion_matrix, accuracy_score

#Import Train and Dataset

train=pd.read_csv("../input/train.csv")

train.drop("PassengerId",axis=1,inplace=True)

train.drop("Name",axis=1,inplace=True)





test=pd.read_csv("../input/test.csv")

test.drop("PassengerId",axis=1,inplace=True)

test.drop("Name",axis=1,inplace=True)



#count-plot of people survided 

sns.set_style('whitegrid')

sns.countplot(x='Survived', hue='Sex', data=train, palette='RdBu_r')

plt.show()



#distribution plot of age of the people

sns.distplot(train['Age'].dropna(), kde=False, bins=30, color='Green')

plt.show()



#Checking NA values in Dataset by column

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')  #Detection of Na/Null values by column

plt.show()



#It gives us sum of NA values by column

print(train.isna().sum())



#Checking NA values in Dataset by column

sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')  #Detection of Na/Null values by column

plt.show()



#It gives us sum of NA values by column

print(test.isna().sum())

#Fill NA values with Mode

train.groupby('Embarked').size()

train["Embarked"].fillna("S",inplace=True)



#Drop Cabin column which has most NA values

train.drop("Cabin",axis=1,inplace=True)



#Fill NA values with Median

train["Age"].fillna(train.Age.median(),inplace=True)



#check after Filling NA

print(train.isna().sum())







#Drop Cabin column which has most NA values

test.drop("Cabin",axis=1,inplace=True)



#Fill NA values with Median

test["Age"].fillna(test.Age.median(),inplace=True)

test["Fare"].fillna(test.Fare.median(),inplace=True)

#check after Filling NA

print(test.isna().sum())

#We have to create Dummy variable for Train Dataset

sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)



#drop the sex,embarked,name and tickets columns

train.drop(['Sex','Embarked','Ticket'],axis=1,inplace=True)



#concatenate new sex and embark column to our train trainframe

train = pd.concat([train,sex,embark],axis=1)



#check the head of trainframe

print(train.head())



#We have to create Dummy variable for Test Dataset

sex = pd.get_dummies(test['Sex'],drop_first=True)

embark = pd.get_dummies(test['Embarked'],drop_first=True)



#drop the sex,embarked,name and tickets columns

test.drop(['Sex','Embarked','Ticket'],axis=1,inplace=True)



#concatenate new sex and embark column to our train trainframe

test = pd.concat([test,sex,embark],axis=1)



#check the head of trainframe

print(test.head())
#Splitting Data Set

X=train.values[:,1:]

y=train.values[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#Logistic Regression

LR=LogisticRegression()

model_LR=LR.fit(X_train, y_train)
pred_LR=model_LR.predict(X_test)

print("Confusion Matrix :",confusion_matrix(y_test, pred_LR))

print("Accuracy :",accuracy_score(y_test, pred_LR)*100)  
##Logistic Regression Gives 79% Accuracy. This means our Prediction is 79% Accurate. Now check Other Algorithm
svm=SVC()

model_svm=svm.fit(X_train, y_train)

pred_svm=model_svm.predict(X_test)

accuracy_score(y_test, pred_svm)

print("Confusion Matrix :",confusion_matrix(y_test, pred_svm))

print("Accuracy :",accuracy_score(y_test, pred_svm)*100)
#Random Forest

RF=RandomForestClassifier(n_estimators=500,random_state=0)

model_RF=RF.fit(X_train, y_train)

pred_RF=model_RF.predict(X_test)

print("Confusion Matrix :",confusion_matrix(y_test, pred_RF))

print("Accuracy :",accuracy_score(y_test, pred_RF)*100)
#Random Forest Algorithm Gives 77% Accuracy. 
#XGBOOST 

XG=GradientBoostingClassifier()

model_XG=XG.fit(X_train, y_train)

pred_XG=model_XG.predict(X_test)

print("Confusion Matrix :",confusion_matrix(y_test, pred_XG))

print("Accuracy :",accuracy_score(y_test, pred_XG)*100)
#XGBOOST gives 81% Accuracy
#Predict Survived for Test Dataset

predict= model_XG.predict(test)

predict
#Convert predict file np.array to DataFrame

predict=pd.DataFrame(predict)
#Export predict to CSV file 

#predict.to_csv("../input/predict.csv")