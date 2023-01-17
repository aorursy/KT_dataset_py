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
train_df=pd.read_csv('/kaggle/input/titanic/train.csv')
test_df=pd.read_csv('/kaggle/input/titanic/test.csv')
train_df
test_df
Id=test_df['PassengerId']
train_df.isnull().sum()
train_df['Age']=train_df['Age'].fillna(train_df['Age'].mean())
import seaborn as sns

sns.heatmap(train_df.isnull())
sns.heatmap(test_df.isnull())
test_df['Age']=test_df['Age'].fillna(test_df['Age'].mean())
train_df=train_df.drop('Cabin',axis=1)

test_df=test_df.drop('Cabin',axis=1)
sns.heatmap(train_df.isnull())
train_df['Embarked']=train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
train_df.isnull().sum()
train_df=train_df.drop(['PassengerId','Ticket','Fare'],axis=1)
Id=test_df['PassengerId']
test_df=test_df.drop(['PassengerId','Ticket','Fare'],axis=1)
train_df=train_df.drop(['Name'],axis=1)

test_df=test_df.drop(['Name'],axis=1)
train_df
train_df['Age']=train_df['Age'].astype('int64')
train_df
train_df[train_df['Survived']==1]['SibSp']
sns.countplot(x=train_df['Survived'],hue='SibSp',data=train_df)
sns.countplot(x=train_df['Survived'],hue='Embarked',data=train_df)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

train_df['Sex']=le.fit_transform(train_df['Sex'])

train_df['Embarked']=le.fit_transform(train_df['Embarked'])
test_df['Sex']=le.fit_transform(test_df['Sex'])

test_df['Embarked']=le.fit_transform(test_df['Embarked'])
train_df
test_df['Age']=test_df['Age'].astype('int64')
y=train_df['Survived']
train_df=train_df.drop(['Survived'],axis=1)
train_df.isnull().sum()
test_df.isnull().sum()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(train_df,y,test_size=0.2)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

lr=LogisticRegression()

lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)

score_1=accuracy_score(y_test,pred_1)
score_1
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

list_1=[]

for i in range(1,11):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    pred_y=knn.predict(x_test)

    score_1=accuracy_score(y_test,pred_y)

    list_1.append(score_1)

    
plt.plot(range(1,11),list_1)

plt.show()
print(max(list_1))
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

rfc=RandomForestClassifier()

params={'n_estimators': [200,300,400,500,600, 700],

    'max_features': ['auto', 'sqrt', 'log2']}
clf=GridSearchCV(rfc,params,cv=5,verbose=0)

clf.fit(x_train,y_train)

print(clf.best_params_)
rfc=RandomForestClassifier(max_features='log2',n_estimators=500)
rfc.fit(x_train,y_train)

pred_1=rfc.predict(x_test)

score_2=accuracy_score(y_test,pred_1)
score_2
from sklearn.svm import SVC

model=SVC()

params = {'C': [0.1, 1, 10, 100, 1000],  

              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 

              'kernel': ['rbf']} 
clf_1=GridSearchCV(model,params,cv=5,verbose=0)
clf_1.fit(x_train,y_train)

print(clf_1.best_params_)
model=SVC(C=1000,gamma=0.001,kernel='rbf')

model.fit(x_train,y_train)

pred_2=model.predict(x_test)

score_3=accuracy_score(y_test,pred_2)
score_3
from sklearn.naive_bayes import GaussianNB 

gnb = GaussianNB() 
gnb.fit(x_train,y_train)

pred_4=gnb.predict(x_test)
score_4=accuracy_score(y_test,pred_4)
score_4
from sklearn.linear_model import SGDClassifier
sdgc= SGDClassifier()
params={

    'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], # learning rate

    'penalty': ['l2'],

    'n_jobs': [-1,-2,-3,-4,-5,-6,1,2]

}
clf_3=GridSearchCV(sdgc,params,cv=5,verbose=0)
clf_3.fit(x_train,y_train)
print(clf_3.best_params_)
sdgc=SGDClassifier(alpha=0.0001,n_jobs=-1,penalty='l2')
sdgc.fit(x_train,y_train)
pred_5=sdgc.predict(x_test)

score_5=accuracy_score(y_test,pred_5)
score_5
predictions=rfc.predict(test_df)
submission=pd.DataFrame({'PassengerId':Id,

                        'Survived':predictions})
submission=submission.to_csv('submission.csv',index=False)