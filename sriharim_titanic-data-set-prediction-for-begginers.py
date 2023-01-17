#importing the required packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#importing training and test dataset into data frame using pandas
training_data_set= pd.read_csv("../input/train.csv")
test_data_set= pd.read_csv("../input/test.csv")
original_test_data= pd.read_csv("../input/test.csv")
#view the train and test data
training_data_set.head(n=5)

test_data_set.head(n=5)
#Name, PassengerId and Ticket as nothing to do with the survival
training_data_set.drop(columns=["Name","PassengerId","Ticket"], inplace=True)

test_data_set.drop(columns=["Name","PassengerId","Ticket"], inplace=True)
#describe the dataset
train_stats=training_data_set.describe(include='all')
test_stats=test_data_set.describe(include='all')
#finding the missing values
train_stats.loc['count']
test_stats.loc['count']
#cabin column as many missing values so drop it
training_data_set.drop(columns='Cabin', inplace=True)
test_data_set.drop(columns='Cabin', inplace=True)
#update stats
train_stats=training_data_set.describe(include='all')

test_stats=test_data_set.describe(include='all')
#try to visualize to find co-relation between features
sns.barplot(x=training_data_set.Pclass,y=training_data_set.Survived)
plt.ylabel('Passengers Survived %')
sns.barplot(x=training_data_set.Sex,y=training_data_set.Survived)
sns.kdeplot(training_data_set.loc[training_data_set['Survived'] ==0, 'Age'].dropna(),color='red',label='Did not survive')
sns.kdeplot(training_data_set.loc[training_data_set['Survived'] ==1, 'Age'].dropna(),color='green',label='Survived')
plt.xlabel('Age')
plt.ylabel('Survived')
sns.barplot(training_data_set.Embarked,training_data_set.Survived)
plt.ylabel("Passengers survived")
#Most of the passengers survived are from Embarked=c
sns.barplot(training_data_set.SibSp,training_data_set.Survived)
#We see survival rate with SibSp is more for 1,2 and is distributed
sns.barplot(training_data_set.Parch,training_data_set.Survived)
#we don't see any co-relation for parch
sns.pointplot(training_data_set.Survived,training_data_set.Fare)
#we see the survival rate increases with increase in fare
print(train_stats)
print(test_stats)
#In train data age and embarked have missing values, In test data age and fare have missing values
#we see the age is continous data hence we can fill the missing values with mean
training_data_set["Age"]=training_data_set.Age.fillna(train_stats.loc['mean','Age'])
test_data_set["Age"]=test_data_set.Age.fillna(test_stats.loc['mean','Age'])
#we see majority of embarked are S so we fill the missing value with mode of S
sns.countplot(x=training_data_set.Embarked)


from statistics import mode
embarked_mode=mode(training_data_set.Embarked)
training_data_set["Embarked"]=training_data_set.Embarked.fillna(embarked_mode)
#we will fill fare with passenger's fare which has similar data
empty_fare=test_data_set[test_data_set['Fare'].isnull()]
print(empty_fare)
use_fare=test_data_set[(test_data_set['Pclass']==3) & (test_data_set['Sex']=='male') & (test_data_set['SibSp']==0) & (test_data_set['Parch']==0) & (test_data_set['Embarked']=='S')]
test_data_set['Fare']=test_data_set['Fare'].fillna(use_fare['Fare'].iloc[0])
#update stats
train_stats=training_data_set.describe(include='all')
test_stats=test_data_set.describe(include='all')
print(test_stats.loc['count','Fare'])
#now  we have complete data with no missing values and outliers.
#we shall convert the age feature into categorical feature.
training_data_set['kid/teenager']=np.where(training_data_set['Age']<=20,1,0)
training_data_set['young/adult']=np.where((training_data_set['Age']>20) & (training_data_set['Age']<=40),1,0)
training_data_set['elder']=np.where((training_data_set['Age']>40) & (training_data_set['Age']<=60),1,0)
training_data_set['old']=np.where(training_data_set['Age']>60,1,0)

test_data_set['kid/teenager']=np.where(test_data_set['Age']<=20,1,0)
test_data_set['young/adult']=np.where((test_data_set['Age']>20) & (test_data_set['Age']<=40),1,0)
test_data_set['elder']=np.where((test_data_set['Age']>40) & (test_data_set['Age']<=60),1,0)
test_data_set['old']=np.where(test_data_set['Age']>60,1,0)
#drop age column from data
training_data_set.drop('Age',axis=1,inplace=True)
training_data_set.head()






test_data_set.drop('Age', axis=1, inplace=True)
test_data_set.head()
#we will combine sibsp and parch features into one
training_data_set['Family']=np.where((training_data_set['SibSp'] + training_data_set['Parch'])>0,1,0)
training_data_set.drop(['SibSp','Parch'],axis=1,inplace=True)
training_data_set.head(n=10)

test_data_set['Family']=np.where((test_data_set['SibSp'] + test_data_set['Parch'])>0,1,0)
test_data_set.drop(['SibSp','Parch'],axis=1,inplace=True)
test_data_set.head(n=10)
#we see Embarked and Pclass feature is not in binary form to convert it into binary we shall remove the dummies
training_data_set=pd.get_dummies(data=training_data_set, prefix=['Embarked'], columns=['Embarked'])
training_data_set=pd.get_dummies(data=training_data_set, prefix=['Pclass'], columns=['Pclass'])
training_data_set.drop(['Embarked_S','Pclass_3','old'],axis=1,inplace=True)
training_data_set.rename(index=str,columns={'Embarked_C':'Embarked_1','Embarked_Q':'Embarked_2'},inplace=True)
training_data_set.head(n=10)

test_data_set=pd.get_dummies(data=test_data_set, prefix=['Embarked'], columns=['Embarked'])
test_data_set=pd.get_dummies(data=test_data_set, prefix=['Pclass'], columns=['Pclass'])
test_data_set.drop(['Embarked_S','Pclass_3','old'],axis=1,inplace=True)
test_data_set.rename(index=str,columns={'Embarked_C':'Embarked_1','Embarked_Q':'Embarked_2'},inplace=True)
test_data_set.head(n=10)

#now let's encode the Sex feature
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
training_data_set['Sex']=encoder.fit_transform(training_data_set['Sex'])
training_data_set.head(n=10)
test_data_set['Sex']=encoder.fit_transform(test_data_set['Sex'])
test_data_set.head(n=10)
#Now we have the data ready let's apply logistic regression
X_train=training_data_set.drop('Survived', axis=1)
Y_train=training_data_set['Survived']
X_test=test_data_set.copy()
from sklearn.linear_model import LogisticRegression
logreg= LogisticRegression()
logreg.fit(X_train,Y_train)
logreg.predict(X_test)
logreg.score(X_train,Y_train)
#we will try SVM as accuracy is less
from sklearn import svm
svc= svm.SVC()
svc.fit(X_train,Y_train)
svc.predict(X_test)
svc.score(X_train,Y_train)
#we will try random forest with estimators equal to 100
from sklearn import ensemble
rfc=ensemble.RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,Y_train)
Y_pred=rfc.predict(X_test)
rfc.score(X_train,Y_train)

#Use random forest classifier
submission = pd.DataFrame({"PassengerId": original_test_data["PassengerId"],"Survived": Y_pred})
submission.to_csv('sub_preds.csv', index=False)