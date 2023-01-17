# Import Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
print(os.listdir("../input"))

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)
train['Age'].hist(bins=30,color='darkred',alpha=0.7)
sns.countplot(x='SibSp',data=train)
train['Fare'].hist(color='green',bins=40,figsize=(8,4))
import cufflinks as cf
cf.go_offline()
train['Fare'].iplot(kind='hist',bins=30,color='green')
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
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
train.dropna(inplace=True)
train.info()
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
train.head()

sex_test = pd.get_dummies(test['Sex'],drop_first=True)
embark_test = pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['Sex','Embarked','Name','Ticket','Cabin'],axis=1,inplace=True)
test = pd.concat([test,sex_test,embark_test],axis=1)
test.head()
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=test,palette='winter')
def impute_age1(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 42

        elif Pclass == 2:
            return 27

        else:
            return 25

    else:
        return Age
test['Age'] = test[['Age','Pclass']].apply(impute_age1,axis=1)
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.figure(figsize = (16,10))
sns.heatmap(test.corr(),annot=True, fmt = ".2f", cmap = "coolwarm")

plt.figure(figsize = (12,10))
sns.boxplot(x='Pclass',y='Fare',data=test,palette='winter')
plt.figure(figsize = (30,10))
sns.countplot(x='Fare',data= test)
plt.tight_layout()
test.describe()
test['Fare'].fillna(35.627188,inplace =True)

sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')


x_train = train.drop('Survived',axis=1)
y_train = train['Survived']
x_test  = test.copy()
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier( n_estimators=800,oob_score=True)
#rfc = RandomForestClassifier(criterion='gini', 
                             #n_estimators=700,
                             #min_samples_split=16,
                           #  min_samples_leaf=1,
                            # max_features='auto',
                             #oob_score=True,
                          #   random_state=1,
                            # n_jobs=-1) 

rfc.fit(x_train, y_train)
print("%.4f" % rfc.oob_score_)

Y_pred = rfc.predict(x_test)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('csv_to_submit.csv', index = False)






