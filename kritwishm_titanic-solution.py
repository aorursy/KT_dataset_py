import pandas as pd
import numpy as np

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.info()
test.info()
train.isnull().sum()
test.isnull().sum()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style(style='whitegrid')
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.countplot(x='Survived',data=train,hue='Sex')
sns.countplot(x='Survived',hue='Pclass',data=train)
sns.distplot(train['Age'].dropna(),kde=False,bins=30,color='darkred')
sns.countplot(x='SibSp',data=train)
train['Fare'].hist(bins=30,color='green',figsize=(8,5))
plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        
        if Pclass==1:
            return 37
        
        if Pclass==2:
            return 28
        
        if Pclass==3:
            return 24
    
    else:
        return Age
train['Age']=train[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.drop('Cabin',axis=1,inplace=True)
sex= pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train=pd.concat([train,sex,embark],axis=1)
train.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=104)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
test=pd.read_csv('../input/test.csv')
test.drop('Cabin',axis=1,inplace=True)
sex_test= pd.get_dummies(test['Sex'],drop_first=True)
embark_test = pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
test=pd.concat([test,sex_test,embark_test],axis=1)
test.head()
test['Age']=test[['Age','Pclass']].apply(impute_age,axis=1)
nan_rows = test[test.isnull().T.any().T]
nan_rows
test.groupby('Pclass').mean()
test['Fare'][152]=12.459678
test['Fare'][152]
X_train=train.drop('Survived',axis=1)
y_train=train['Survived']
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
Survived=logmodel.predict(test)
Survived
test['Survived']=Survived
test.head()
prediction=test.drop(['Pclass','Age','SibSp','Parch','Fare','male','Q','S'],axis=1)
prediction.head()
prediction.to_csv('prediction_titanic',index=False)
prediction.info()