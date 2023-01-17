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
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

import seaborn as sns

import matplotlib.pyplot as plt
sns.distplot(a=train[train['Survived']==1]['Pclass'],label='Alive',bins=5)

sns.distplot(a=train[train['Survived']==0]['Pclass'],label='Dead',bins=5)

plt.legend()

plt.xticks([1,2,3])
test['Age'].replace(np.nan,np.median(test['Age'].dropna()),inplace=True)

train['Age'].replace(np.nan,np.median(train['Age'].dropna()),inplace=True)
train.head()

sns.distplot(a=train[train['Survived']==1]['Age'],label='Alive',bins=5)

sns.distplot(a=train[train['Survived']==0]['Age'],label='Dead',bins=5)

plt.legend()
train['Sex'].replace({'male':1,'female':0},inplace=True)

test['Sex'].replace({'male':1,'female':0},inplace=True)


sns.distplot(a=train[train['Survived']==1]['Sex'],kde=False,bins=3,label='Alive')

sns.distplot(a=train[train['Survived']==0]['Sex'],kde=False,bins=3,label='Dead')

plt.legend()

plt.xticks([0,1])

plt.show()

train['Sex'].value_counts()
sns.scatterplot(x=train['Survived'],y=train['SibSp']+train['Parch'])
sns.distplot(a=train[train['Embarked']=='C']['Survived'],bins=3,kde=False)

plt.title("Cherbourg")

plt.xticks([0,1])

plt.show()

plt.title("QueensTown")

sns.distplot(a=train[train['Embarked']=='Q']['Survived'],bins=3,kde=False)

plt.xticks([0,1])



plt.show()

plt.title("Southampton")

sns.distplot(a=train[train['Embarked']=='S']['Survived'],bins=3,kde=False)

plt.xticks([0,1])



plt.show()
sns.distplot(a=train[train['Embarked']=='S']['Pclass'],kde=False,bins=5)

plt.title('Passenger Class of people embarking from Southampton')

plt.show()

sns.distplot(a=train[train['Embarked']=='Q']['Pclass'],kde=False,bins=5)

plt.title('Passenger Class of people embarking from Queenstown')

plt.show()

sns.distplot(a=train[train['Embarked']=='C']['Pclass'],kde=False,bins=5)

plt.title('Passenger Class of people embarking from Cherbourg')

plt.show()

train['Embarked'].value_counts()
embarked=pd.get_dummies(train['Embarked'])

t_embarked=pd.get_dummies(test['Embarked'])

train=train.join(embarked)

test=test.join(t_embarked)
train.drop(['Name','SibSp','Parch','Ticket','Cabin','Embarked'],inplace=True,axis=1)

test.drop(['Name','SibSp','Parch','Ticket','Cabin','Embarked'],inplace=True,axis=1)
sns.heatmap(train.corr(),cmap='CMRmap')

plt.legend()
test['Fare'].replace({np.nan:np.median(test['Fare'].dropna())},inplace=True)
test.set_index('PassengerId',inplace=True)

train.set_index('PassengerId',inplace=True)
from sklearn.ensemble import RandomForestClassifier

X_train=train.drop('Survived',axis=1)

y_train=train['Survived']
forest=RandomForestClassifier(max_depth=4,n_estimators=2000)

forest.fit(X_train,y_train)

y_pred=forest.predict_proba(test)

y_pred=(y_pred[:,1]>0.45).astype(int)

final_df=pd.DataFrame({'PassengerId':test.index,'Survived':y_pred})

final_df.set_index('PassengerId',inplace=True)
final_df.to_csv('submit.csv')#the one with 0.784 score
from sklearn.svm import SVC

svc=SVC(C=1,gamma=10)

svc.fit(X_train,y_train)

y_pred2=svc.predict(test)

final_df2=pd.DataFrame({'PassengerId':test.index,'Survived':y_pred2})

final_df2.set_index('PassengerId',inplace=True)
final_df2.to_csv('submit.csv')#the one with 0.5 score