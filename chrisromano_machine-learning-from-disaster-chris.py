import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.info()
test.head()
test.info()
sns.heatmap(train.isnull(),yticklabels=False, cbar=False)
train.drop('Cabin',axis=1, inplace=True)
test.drop('Cabin',axis=1, inplace=True)
sns.heatmap(train.isnull(),yticklabels=False, cbar=False)
sns.boxplot(x='Pclass',y='Age',hue='Sex',data=train)
train_class_sex=train[['Pclass','Sex','Age']]
female_class=train_class_sex.loc[train_class_sex['Sex']=='female']
female_class.head()
male_class=train_class_sex.loc[train_class_sex['Sex']=='male']
male_class.head()
male_class.loc[male_class['Pclass']==1].Age.mean()
male_class.loc[male_class['Pclass']==2].Age.mean()
male_class.loc[male_class['Pclass']==3].Age.mean()
female_class.loc[female_class['Pclass']==1].Age.mean()
female_class.loc[female_class['Pclass']==2].Age.mean()
female_class.loc[female_class['Pclass']==3].Age.mean()
#Based on the graph of age VS Pclass, impute the values:

def ImputeAge(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
train['Age'] = train[['Age','Pclass']].apply(ImputeAge,axis=1)
train.head(10)


sns.heatmap(train.isnull(),yticklabels=False,cbar=False)
sns.boxplot(x='Pclass',y='Age', hue='Sex',data=train)   
train['Sex'] = train['Sex'].map({'male':1,'female':0}).astype(int)
test['Sex'] = test['Sex'].map({'male':1, 'female':0}).astype(int)
train.head()
test.head()
train.drop(['Name','Ticket'],axis=1, inplace=True)
test.drop(['Name', 'Ticket'],axis=1, inplace=True)

train.head(2)
test.head(2)
sns.set_style('whitegrid')
sns.pairplot(train,hue='Survived')
sns.countplot(x='Pclass',hue='Survived',data=train)
sns.countplot(x='Embarked',hue='Survived',data=train)
train['Embarked']=train['Embarked'].map({'S':0,'C':1,'Q':2}).astype(float)

test['Embarked']=test['Embarked'].map({'S':0,'C':1,'Q':2}).astype(float)

train['Embarked'].head()
test['Embarked'].head()
train.head(10)
test.head(10)
train.shape
test.shape
train.info()
test.info()
for train['Embarked'] in train['Embarked']:
    train['Embarked'] = train['Embarked'].fillna(0)
for test['Embarked'] in test['Embarked']:
    test['Embarked'] = test['Embarked'].fillna(0)
train.info()
test.fillna(value=0,inplace=True)
test.info()
X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
X_test = test
X_train.shape, y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predict = logmodel.predict(X_train)

from sklearn.metrics import classification_report
print(classification_report(y_train, predict))
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
final_svc = round(svc.score(X_train, y_train)*100, 2)
final_svc
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)
final_knn = round(knn.score(X_train,y_train)*100,2)
final_knn

from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
random_forest.score(X_train,y_train)
final_random_forest = round(random_forest.score(X_train,y_train)*100,2)
final_random_forest

submission = pd.DataFrame({'PassengerID':test['PassengerId'],'Survived':y_pred})
submission.to_csv('submission.csv',index=False)
submission.head()
