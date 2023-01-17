import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train=pd.read_csv('../input/titanic/train.csv')
train.head()
test=pd.read_csv('../input/titanic/test.csv')
test.head()
train.describe()
sns.heatmap(train.isnull(),cbar=False,yticklabels=False,cmap='viridis')
sns.heatmap(test.isnull(),cbar=False,yticklabels=False,cmap='viridis')
train.info()
test.info()
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)
sns.countplot(x='Survived',data=train,hue='Sex')
sns.countplot(x='Survived',data=train,hue='Sex')
sns.countplot(x='Survived',data=train,hue='Pclass')
sns.distplot(train['Age'].dropna(),kde=False,color='blue')
sns.countplot(x='SibSp',data=train)
train['Fare'].hist(color='green',bins=40,figsize=(8,4))
plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=train)
class_group=train.groupby('Pclass')
class_group.mean()
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 38.2

        elif Pclass == 2:
            return 29.8

        else:
            return 25.1

    else:
        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train.isnull(),cbar=False,yticklabels=False,cmap='viridis')
train.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)
train.info()
class_group=test.groupby('Pclass')
class_group.mean()
def fill_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 40.6

        elif Pclass == 2:
            return 28.8

        else:
            return 24.3

    else:
        return Age
test['Age'] = test[['Age','Pclass']].apply(fill_age,axis=1)
test.drop('Cabin',axis=1,inplace=True)
test[test['Fare'].isnull()]
test.fillna(value=12.4,inplace=True)
test.info()
train['Sex']=train['Sex'].map({'female':0, 'male':1}).astype(int)
train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test['Sex']=test['Sex'].map({'female':0, 'male':1}).astype(int)
test['Embarked'] = test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
test.drop(['Name','Ticket'],axis=1,inplace=True)
x_train=train.drop('Survived',axis=1)
y_train=train['Survived']
x_test=test.drop('PassengerId',axis=1)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression(max_iter=1000)
logmodel.fit(x_train,y_train)
predictions=logmodel.predict(x_test)
logmodel.score(x_train,y_train)
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(x_train,y_train)
Tpredictions=dtree.predict(x_test)
dtree.score(x_train,y_train)
df=pd.DataFrame({'PassengerId': test['PassengerId'],
                  'Predictions': Tpredictions  })
df.to_csv('submission1.csv',index=False)