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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
print(train.shape)
print(test.shape)
train.head()
train.info()
sns.heatmap(train.isnull())
sns.countplot(x='Survived',data=train)
sns.countplot(x='Survived',hue='Sex',data=train)
sns.countplot(x='Survived',hue='Pclass',data=train)
sns.distplot(train['Age'].dropna(),kde=False, bins=30)
sns.countplot(x='SibSp', data=train)
train['Fare'].hist(figsize=(10,4),bins=40)
plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass',y='Age', data=train)
train.groupby('Pclass')['Age'].mean()
def impute_age(cols):
    Age  = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 30
        else:
            return 25
    else:
        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
sum(train['Age'].isnull())
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
# too many mising values for cabin column, so drop it
train.drop('Cabin',axis=1,inplace=True)
train.head()
train.isna().sum()
# Only 2 missing values in Embarked column
train.loc[train['Embarked'].isna(),'Embarked'] = 'S'
train.isna().sum()
sex = pd.get_dummies(train['Sex'],drop_first=True)
sex.head()
embark = pd.get_dummies(train['Embarked'],drop_first=True)
embark.head(5)
train = pd.concat([train,sex,embark],axis=1)
train.head()
train.drop(['Sex','Embarked','Name','Ticket'],axis=1, inplace=True)
train.head()
train.drop('PassengerId',axis=1,inplace=True)
X = train.drop('Survived',axis=1)
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
print(classification_report(y_test, predictions))
print('\n')
confusion_matrix(y_test,predictions)
np.mean(cross_val_score(LogisticRegression(),X,y,cv=5))
np.mean(cross_val_score(SVC(),X,y,cv=5))
np.mean(cross_val_score(RandomForestClassifier(n_estimators=50),X,y,cv=5))
rf = RandomForestClassifier(n_estimators=50)
rf.fit(X_train,y_train)
pred = rf.predict(X_test)
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
test.isna().sum()
test['Fare'].mean()
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
test.drop('Cabin',axis=1,inplace=True)
test.loc[test['Fare'].isna(),'Fare'] = test['Fare'].mean()
Tsex = pd.get_dummies(test['Sex'],drop_first=True)
Tembark = pd.get_dummies(test['Embarked'],drop_first=True)
test = pd.concat([test,Tsex,Tembark],axis=1)
test.drop(['Sex','Embarked','Name','Ticket'],axis=1, inplace=True)
test.drop('PassengerId',axis=1,inplace=True)
test.shape
pred_test = rf.predict(test)
tt = pd.read_csv('../input/titanic/test.csv')
df_submit = pd.DataFrame(tt['PassengerId'])
df_submit['Survived']=pd.DataFrame(pred_test)
df_submit.head()
df_submit['Survived'].value_counts()
df_submit.to_csv("submission2.csv", index=False)