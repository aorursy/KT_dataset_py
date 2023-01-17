import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train = pd.read_csv('../input/titanic-train-public-dataset/titanic_train.csv')
train.head()
train.isnull()
sns.heatmap(train.isnull(),cbar=False,yticklabels=False,cmap='viridis')
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)
sns.countplot(x='Survived',hue='Sex',data=train)
sns.countplot(x='Survived',hue='Pclass',data=train)
sns.distplot(train['Age'].dropna(),kde=False,bins=50)
train['Age'].plot.hist(bins=50)
train.info()
sns.countplot(x='SibSp',data=train)
train['Fare'].hist(bins=50,figsize=(10,4))
import cufflinks as cf
cf.go_offline()
train['Fare'].iplot(kind='hist',bins=50)
plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass',y='Age',data=train)
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
      
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
        
    else:
        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.drop('Cabin',axis=1,inplace=True)
train.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.dropna(inplace=True)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
pd.get_dummies(train['Sex'])
sex = pd.get_dummies(train['Sex'],drop_first=True)
sex
pd.get_dummies(train['Embarked'])
embark = pd.get_dummies(train['Embarked'],drop_first=True)
embark
train = pd.concat([train,sex,embark],axis=1)
train.head()
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train.head()
train.tail()
train.drop('PassengerId',axis=1,inplace=True)
train.head()
X = train.drop('Survived',axis=1)
y = train['Survived']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,predictions)
