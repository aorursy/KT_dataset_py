import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

train = pd.read_csv("../input/titanic/titanic_train.csv")
train.head()
train.info()
plt.figure(figsize=(12,6))
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,hue='Sex',palette='RdBu_r')
sns.countplot(x='Survived',data=train,hue='Pclass',palette='rainbow')
train['Age'].hist(bins=30,color='darkred',alpha=0.4)
sns.countplot(x='SibSp',data=train)
train[train['SibSp'] == 0]['Age'].hist(bins=30)
train[train['Fare']<70]['Fare'].hist(color='green',bins=50,figsize=(12,6))
plt.figure(figsize=(12,6))
sns.boxplot(x='Pclass',y='Age',data=train)
def inputar_idade(cols):
    Idade = cols[0]
    Class = cols[1]
    if pd.isnull(Idade):
        if Class == 1:
            return 37
        elif Class == 2:
            return 29
        else:
            return 24
        
    else:
        return Idade
train['Age'] = train[['Age','Pclass']].apply(inputar_idade,axis=1)
plt.figure(figsize=(12,6))
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
del train['Cabin']
plt.figure(figsize=(12,6))
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.dropna(inplace=True)
plt.figure(figsize=(12,6))
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark =  pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','PassengerId','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
train.head(50)
del train['Embarked']
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(train.drop('Survived',axis=1),train['Survived'],test_size=0.3)
logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)
preditions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(Y_test,preditions))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test,preditions))