# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import warnings
warnings.filterwarnings('ignore')
train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
gender_submission=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')     
train.head()
test.head()
print(train.shape, test.shape)
train.info()
test.info()
train.isnull().sum()
test.isnull().sum()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
def bar_chart(feature):
    survived=train[train['Survived']==1][feature].value_counts()
    dead=train[train['Survived']==0][feature].value_counts()
    df=pd.DataFrame([survived,dead])
    df.index=['survived','dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
bar_chart('Sex')
bar_chart('Pclass')
bar_chart('Embarked')
bar_chart('SibSp')
bar_chart('Parch')
train['Age'].fillna(train['Age'].mean(),inplace=True)
test['Age'].fillna(test['Age'].mean(),inplace=True)
test['Fare'].fillna(test['Fare'].mean(),inplace=True)
train['Embarked'].fillna(value='S',inplace=True)
train['family']=train['SibSp']+train['Parch']+1
test['family']=test['SibSp']+train['Parch']+1
train['Sex'] = train['Sex'].replace(['female','male'],[0,1])
train['Embarked'] = train['Embarked'].replace(['S','Q','C'],[1,2,3])
test['Sex'] = test['Sex'].replace(['female','male'],[0,1])
test['Embarked'] = test['Embarked'].replace(['S','Q','C'],[1,2,3])
train_clean=train.drop(columns=['PassengerId','Name','SibSp','Parch','Ticket','Cabin'])
test_clean=test.drop(columns=['PassengerId','Name','SibSp','Parch','Ticket','Cabin'])
train_clean
X_train=train_clean.drop(columns=['Survived'])
y_train=train_clean[['Survived']]
from sklearn.preprocessing import StandardScaler
X_train_scale=StandardScaler().fit_transform(X_train)
pd.DataFrame(X_train_scale).head()
from sklearn.linear_model import LogisticRegression

LR=LogisticRegression().fit(X_train_scale, y_train)
y_pred=LR.predict(test_clean)
from sklearn.metrics import classification_report

print(classification_report(y_pred, gender_submission['Survived']))
from sklearn.model_selection import cross_val_score

scores=cross_val_score(LogisticRegression(),X_train_scale,y_train,cv=5)
print(scores)
print(scores.mean()) # generalization score
from sklearn.model_selection import GridSearchCV

grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]} # C = 1/alpha
grid
score=GridSearchCV(LogisticRegression(),grid).fit(X_train_scale, y_train)

print(score.best_params_)
print(score.best_score_)
data = {'PassengerId':gender_submission['PassengerId'],
        'Survived':y_pred}
result=pd.DataFrame(data)
result.to_csv('/kaggle/working/result_lr.csv', index=False)
from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier().fit(X_train_scale, y_train)
y_pred_rf=RF.predict(test_clean)
from sklearn.metrics import classification_report
#print(classification_report(y_pred_rf, gender_submission['Survived']))
#print(y_pred_rf)
scores=cross_val_score(RandomForestClassifier(), X_train_scale, y_train, cv=5)
print(scores)
print(scores.mean())
data={'PassengerId': gender_submission['PassengerId'],'Survived':y_pred_rf}
result_rf=pd.DataFrame(data)
result_rf.to_csv('/kaggle/working/result_rf.csv', index=False)
# result_rf1=pd.read_csv('/kaggle/working/result_rf.csv')
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
svc=SVC(kernel='linear', C=1)
scores=cross_val_score(svc, X_train_scale, y_train, cv=5)
print(scores)
print(scores.mean())
y_pred_svc=SVC(kernel='linear', C=1).fit(X_train_scale, y_train).predict(test_clean)
data={'PassengerId': gender_submission['PassengerId'],'Survived':y_pred_svc}
result_svc=pd.DataFrame(data)
result_svc.to_csv('/kaggle/working/result_svc.csv', index=False)
result_svc=pd.read_csv('/kaggle/working/result_svc.csv')
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
dtc=DecisionTreeClassifier()
scores=cross_val_score(dtc, X_train, y_train, cv=5)
print(scores)
print(scores.mean())

y_pred_dtc=DecisionTreeClassifier().fit(X_train, y_train).predict(test_clean)
data={'PassengerId': gender_submission['PassengerId'],'Survived':y_pred_dtc}
result_dtc=pd.DataFrame(data)
result_dtc.to_csv('/kaggle/working/result_dtc.csv', index=False)
result_dtc=pd.read_csv('/kaggle/working/result_dtc.csv')
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
nb=GaussianNB()
scores=cross_val_score(nb, X_train, y_train, cv=5)
print(scores)
print(scores.mean())

y_pred_nb=GaussianNB().fit(X_train, y_train).predict(test_clean)
data={'PassengerId': gender_submission['PassengerId'],'Survived':y_pred_nb}
result_nb=pd.DataFrame(data)
result_nb.to_csv('/kaggle/working/result_nb.csv', index=False)
result_nb=pd.read_csv('/kaggle/working/result_nb.csv')
