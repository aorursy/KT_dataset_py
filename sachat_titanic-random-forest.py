# importing required Libraries

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


import os
print(os.listdir("../input"))


# read CSV file as DataFrame
train = pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
train.head()
# check for missing data

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# boxplot for relation between age and Pclass
plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=train)
# Survived Count
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')
# Survived count by different parameters
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
# Distribution Plot
sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)
# Age Histogram
train['Age'].hist(bins=30,color='darkred',alpha=0.7)
#Countplot
sns.countplot(x='SibSp',data=train)
train['Fare'].hist(color='green',bins=40,figsize=(8,4))
# cufflinks plot
import cufflinks as cf
cf.go_offline()
train['Fare'].iplot(kind='hist',bins=30,color='green')
# function for returning mean age for different Pclass

def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
       
        elif Pclass==2:
            return 29
       
        else:
            return 24
    else:
        return Age
# Apply function to the Dataframe
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
#dropping columns with missing values
train.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)
train.head()
# Converting categorical Values
sex=pd.get_dummies(train['Sex'],drop_first=True)
embark=pd.get_dummies(train['Embarked'],drop_first=True)
#dropping string columns
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train=pd.concat([train,sex,embark],axis=1)
train.head()
test.head()
# Test Data processing
test['Age']=test[['Age','Pclass']].apply(impute_age,axis=1)
test.drop('Cabin',axis=1,inplace=True)
test.dropna(inplace=True)
sex=pd.get_dummies(test['Sex'],drop_first=True)
embark=pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
test=pd.concat([test,sex,embark],axis=1)
# training and test set
X_test=test.drop('PassengerId',axis=1).copy()
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), train['Survived'], test_size=0.3, random_state=101)
#Random Forest
rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
acc_random_forest=print(classification_report(y_test,rfc_pred))