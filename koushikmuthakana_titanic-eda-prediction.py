import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
sns.set(style='whitegrid',color_codes=True)
import os
#print(os.listdir("../input"))


#Reading train CSV Data into DataFrame
train=pd.read_csv('../input/train.csv')

#Reading test CSV Data into DataFrame
test=pd.read_csv('../input/test.csv')

train.head()

print('The number of observations in train dataset is {0}'.format(len(train)))
test.head()
print('The number of observations in test dataset is {0}'.format(len(test)))
train.info()
# we have total 891 entries of 12 columns.
# We have null values in Age, Cabin, Embarked attributes 
#Checking null values
print(train.isnull().sum())

# % of missing values
print("% of missing values of 'Age' column: {0}".format(round((train.Age.isnull().sum()/len(train))*100)))
print("% of missing values of 'Cabin' column: {0}".format(round((train.Cabin.isnull().sum()/len(train))*100)))
print("% of missing values of 'Embarked' column: {0}".format((train.Embarked.isnull().sum()/len(train))*100))
train_data=train.drop('Cabin',axis=1)
train_data.head()
# Missing value columns: 'Age'
train_data['Age'].hist(bins=15,density=True,alpha=0.7)
train_data['Age'].plot(kind='density',color='green')
plt.xlabel('Age')


plt.figure(figsize=(20,10))
sns.countplot(x='Age',data=train_data)
# Missing value column: 'Embarked'
sns.countplot(x='Embarked',data=train_data)
# Impute the Age column with median value
print("The median of 'Age' :",train_data.Age.median())
train_data['Age'].fillna(train_data.Age.median(),inplace=True)

#impute the 'Embarked column with most common boarding port'
print("The most common boarding port  :", train_data['Embarked'].value_counts().idxmax())
train_data['Embarked'].fillna('S',inplace=True)
# New adjusted training data
print(train_data.info())
#train_data.isnull().sum()

#Top 5 rows of new train data
train_data.head()
train_data['TAlone']=[0 if (train_data['SibSp'][i]+train_data['Parch'][i])>0  else 1  for i in range(len(train_data)) ]
train_data.drop(['SibSp','Parch'],axis=1,inplace=True)
# Adding categorical varibales for  pclass,Sex and Embarked 
train_Data=pd.get_dummies(train_data,columns=['Pclass','Sex','Embarked'])
train_Data.drop(['Name','Ticket','PassengerId','Sex_female'],inplace=True,axis=1)
final_train=train_Data.copy()
#Top 5 rows of fianl train data
final_train.head()

#Checking the null values
test.isnull().sum()
#Imputing the missing values
test.Age.fillna(test.Age.median(),inplace=True)
test.Fare.fillna(test.Fare.value_counts().idxmax(),inplace=True)

# Combining Parch and SibSp variables into TAlone
test['TAlone']=[0 if (test['SibSp'][i]+test['Parch'][i])>0 else 1for i in range(len(test))]

#Adding categorical variables
test_data=pd.get_dummies(test,columns=['Pclass','Sex','Embarked'])

#Drop the un wanted variables
test_data.drop(['Cabin','PassengerId','Name','Ticket','Sex_female','Parch','SibSp'],axis=1,inplace=True)
test_data.isnull().sum()
final_test=test_data.copy()
final_test.head()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X=final_train.drop('Survived',axis=1)
Y=final_train['Survived'].values
X.head()
logreg=LogisticRegression()
logreg.fit(X,Y)
y_pred=logreg.predict_proba(X)
print('AUC score is' ,roc_auc_score(y_score=y_pred[:,1],y_true=Y))
