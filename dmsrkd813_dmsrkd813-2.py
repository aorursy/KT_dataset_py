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

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV

import warnings
train_data=pd.read_csv("/kaggle/input/titanic/train.csv")

test_data=pd.read_csv("/kaggle/input/titanic/test.csv")

test_data2=pd.read_csv("/kaggle/input/titanic/test.csv")

titanic=pd.concat([train_data, test_data], sort=False)

len_train_data=train_data.shape[0]
train_data.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())
titanic.select_dtypes(include='int').head()
titanic.select_dtypes(include='object').head()
titanic.select_dtypes(include='float').head()
titanic.isnull().sum()[titanic.isnull().sum()>0]
train_data.Fare=train_data.Fare.fillna(train_data.Fare.mean())

test_data.Fare=test_data.Fare.fillna(train_data.Fare.mean())
train_data.Cabin=train_data.Cabin.fillna("unknow")

test_data.Cabin=test_data.Cabin.fillna("unknow")
train_data.Embarked=train_data.Embarked.fillna(train_data.Embarked.mode()[0])

test_data.Embarked=test_data.Embarked.fillna(train_data.Embarked.mode()[0])
train_data['title']=train_data.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())

test_data['title']=test_data.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())
newtitles={

    "Capt":       "Officer",

    "Col":        "Officer",

    "Major":      "Officer",

    "Jonkheer":   "Royalty",

    "Don":        "Royalty",

    "Sir" :       "Royalty",

    "Dr":         "Officer",

    "Rev":        "Officer",

    "the Countess":"Royalty",

    "Dona":       "Royalty",

    "Mme":        "Mrs",

    "Mlle":       "Miss",

    "Ms":         "Mrs",

    "Mr" :        "Mr",

    "Mrs" :       "Mrs",

    "Miss" :      "Miss",

    "Master" :    "Master",

    "Lady" :      "Royalty"}
train_data['title']=train_data.title.map(newtitles)

test_data['title']=test_data.title.map(newtitles)
train_data.groupby(['title','Sex']).Age.mean()
def newage (cols):

    title=cols[0]

    Sex=cols[1]

    Age=cols[2]

    if pd.isnull(Age):

        if title=='Master' and Sex=="male":

            return 4.57

        elif title=='Miss' and Sex=='female':

            return 21.8

        elif title=='Mr' and Sex=='male': 

            return 32.37

        elif title=='Mrs' and Sex=='female':

            return 35.72

        elif title=='Officer' and Sex=='female':

            return 49

        elif title=='Officer' and Sex=='male':

            return 46.56

        elif title=='Royalty' and Sex=='female':

            return 40.50

        else:

            return 42.33

    else:

        return Age
train_data.Age=train_data[['title','Sex','Age']].apply(newage, axis=1)

test_data.Age=test_data[['title','Sex','Age']].apply(newage, axis=1)
warnings.filterwarnings(action="ignore")
train_data['Relatives']=train_data.SibSp+train_data.Parch

test_data['Relatives']=test_data.SibSp+test_data.Parch



train_data['Ticket2']=train_data.Ticket.apply(lambda x : len(x))

test_data['Ticket2']=test_data.Ticket.apply(lambda x : len(x))



train_data['Cabin2']=train_data.Cabin.apply(lambda x : len(x))

test_data['Cabin2']=test_data.Cabin.apply(lambda x : len(x))



train_data['Name2']=train_data.Name.apply(lambda x: x.split(',')[0].strip())

test_data['Name2']=test_data.Name.apply(lambda x: x.split(',')[0].strip())
warnings.filterwarnings(action="ignore")
train_data.drop(['PassengerId','Name','Ticket','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)

test_data.drop(['PassengerId','Name','Ticket','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)
titanic=pd.concat([train_data, test_data], sort=False)
titanic=pd.get_dummies(titanic)
train_data=titanic[:len_train_data]

test_data=titanic[len_train_data:]
train_data.Survived=train_data.Survived.astype('int')

train_data.Survived.dtype
xtrain=train_data.drop("Survived",axis=1)

ytrain=train_data['Survived']

xtest=test_data.drop("Survived", axis=1)
RF=RandomForestClassifier(random_state=1)

PRF=[{'n_estimators':[10,100],'max_depth':[3,6],'criterion':['gini','entropy']}]

GSRF=GridSearchCV(estimator=RF, param_grid=PRF, scoring='accuracy',cv=2)

scores_rf=cross_val_score(GSRF,xtrain,ytrain,scoring='accuracy',cv=5)
np.mean(scores_rf)
svc=make_pipeline(StandardScaler(),SVC(random_state=1))

r=[0.0001,0.001,0.1,1,10,50,100]

PSVM=[{'svc__C':r, 'svc__kernel':['linear']},

      {'svc__C':r, 'svc__gamma':r, 'svc__kernel':['rbf']}]

GSSVM=GridSearchCV(estimator=svc, param_grid=PSVM, scoring='accuracy', cv=2)

scores_svm=cross_val_score(GSSVM, xtrain.astype(float), ytrain,scoring='accuracy', cv=5)
np.mean(scores_svm)
model=GSSVM.fit(xtrain, ytrain)
pred=model.predict(xtest)
output=pd.DataFrame({'PassengerId':test_data2['PassengerId'],'Survived':pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")