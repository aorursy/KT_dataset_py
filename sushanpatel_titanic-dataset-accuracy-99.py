# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train=pd.read_csv('../input/titanic/train.csv')

X_test=pd.read_csv('../input/titanic/test.csv')
y_test=pd.read_csv('../input/titanic/gender_submission.csv')
train.describe()
train.isnull()
train.info()
plt.figure(figsize=(10,10))
sns.heatmap(train.isnull(),yticklabels=False,cbar=True)

plt.figure(figsize=(15,10))
sns.boxplot('Pclass','Age',data=train)
def Age_1(cols) :
    Age=cols[0]
    Pclass=cols[1] 
    
    if pd.isnull(Age) :
        if Pclass==1 :
            return 38
        elif Pclass==2:
            return 29
        else :
            return 24
    else :
        return Age
                
train['Age']=train[['Age','Pclass']].apply(Age_1,axis=1)
train.drop(['PassengerId','Ticket','Cabin','Name'],inplace=True,axis=1)
X_test.drop(['PassengerId','Ticket','Cabin','Name'],inplace=True,axis=1)
train=train.dropna(axis=0,how='any')
plt.figure(figsize=(10,10))
sns.heatmap(train.isnull(),yticklabels=False,cbar=True)

plt.figure(figsize=(15,10))
sns.boxplot('Pclass','Age',data=X_test)

def Age_1(cols) :
    Age=cols[0]
    Pclass=cols[1] 
    
    if pd.isnull(Age) :
        if Pclass==1 :
            return 42 
        elif Pclass==2:
            return 25
        else :
            return 22
    else :
        return Age
                
X_test['Age']=X_test[['Age','Pclass']].apply(Age_1,axis=1)
sns.countplot('Survived',hue='Sex',data=train)

plt.figure(figsize=(15,10))
sns.boxplot('Pclass','Fare',data=train)
sns.countplot('Survived',hue='Pclass',data=train)
plt.figure(figsize=(15,5))
sns.distplot(train['Fare'],bins=40)
train_fare = train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
plt.figure(figsize=(15,5))
sns.distplot(train_fare,bins=40)

fare=train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
sex=pd.get_dummies(train['Sex'],drop_first=True)
emb=pd.get_dummies(train['Embarked'],drop_first=True)
y_train=train['Survived']
train.drop(['Embarked','Sex','Survived','Fare'],inplace=True,axis=1)
X_train=pd.concat([train,sex,emb,fare],axis=1)
y_test.drop('PassengerId',inplace=True,axis=1)
X_train
test_fare=X_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
test_sex=pd.get_dummies(X_test['Sex'],drop_first=True)
test_emb=pd.get_dummies(X_test['Embarked'],drop_first=True)
X_test.drop(['Embarked','Sex','Fare'],inplace=True,axis=1)
X_test=pd.concat([X_test,test_sex,test_emb,test_fare],axis=1)
Y_test=pd.concat([X_test,test_sex,test_emb,test_fare,y_test],axis=1)
Y_test=Y_test.dropna(axis=0,how='any')

y_test=Y_test['Survived']
from sklearn.linear_model import LogisticRegression

from sklearn import metrics 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
lgr=LogisticRegression(max_iter=500)
lgr_train=lgr.fit(X_train,y_train)
y_pred = lgr.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
from sklearn.ensemble import RandomForestClassifier
rnd_clf=RandomForestClassifier(n_estimators=1000,max_leaf_nodes=16,n_jobs=-1)
rnd_clf.fit(X_train,y_train)
y_pred=rnd_clf.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

params = [{'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}]
b_clf=BaggingClassifier(GridSearchCV(DecisionTreeClassifier(random_state=42),params,cv=3,verbose=1),n_estimators=1000,max_samples=100,bootstrap=True,n_jobs=-1)
b_clf.fit(X_train,y_train)
y_pred=b_clf.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
y_submit=pd.DataFrame(y_pred)
y_submit.to_csv("Submission.csv")
Submission