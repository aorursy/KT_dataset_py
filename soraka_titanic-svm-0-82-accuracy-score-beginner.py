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


import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

print('Train Data İnfo')

train.info()

print('=='*50)

print('Test Data İnfo')

test.info()
train.head()
test.head()
train.drop('Cabin',axis=1,inplace= True)

test.drop('Cabin',axis=1,inplace= True)
import missingno as msno 

msno.matrix(train)
msno.matrix(test)
sns.distplot(train['Age'])
sns.distplot(test['Age'])
sns.distplot(test['Fare'])
train.head()
def box(arr):

    q1 = arr.quantile(0.25)

    q3 = arr.quantile(0.75)

    ıqr = q3-q1

    mini = q1-(1.5*ıqr)

    maxi = q3+(1.5*ıqr)

    arr = arr[(arr>mini)&(arr<maxi)]

    

def standart(arr):

    plus = arr.mean()+3*arr.std()

    minus = arr.mean()-3*arr.std()

    arr = arr[(arr>minus)&(arr<plus)]
standart(train['Age'])

box(test['Fare'])

standart(test['Age'])
sns.distplot(train['Age'])
train['Age']=train['Age'].fillna(train['Age'].median())

test['Age']=test['Age'].fillna(test['Age'].median())

test['Fare'] = test['Fare'].fillna(test['Fare'].median())

train = train.dropna()
print(train.isnull().sum())

print('**'*40)

print(test.isnull().sum())
features = {'Pclass','Sex','SibSp','Parch','Embarked'}

for graph in features:

    sns.countplot(x=train['Survived'],hue=train[graph],data=train)

    plt.ylabel(graph)

    plt.show()
feature = {'Pclass','Sex','SibSp','Parch','Embarked'}

for feat in feature:

    print('\n'*3,feat,'Features',train[feat].value_counts(),'\n','*'*50)
baby = train[(train['Age']<6)]

child = train[(train['Age']>=6)&(train['Age']<12)]

teen = train[(train['Age']>=12)&(train['Age']<18)]

middle_aged = train[(train['Age']>=18)&(train['Age']<35)]

adult = train[(train['Age']>=35)&(train['Age']<50)]

old = train[(train['Age']>=50)]

''''''''''''''''''''''''''''''''''''''''''''''''''''''''

baby_test = test[(test['Age']<6)]

child_test = test[(test['Age']>=6)&(test['Age']<12)]

teen_test = test[(test['Age']>=12)&(test['Age']<18)]

middle_aged_test = test[(test['Age']>=18)&(test['Age']<35)]

adult_test = test[(test['Age']>=35)&(test['Age']<50)]

old_test = test[(test['Age']>=50)]
baby['Age'] = 'Baby'

child['Age'] = 'Child'

teen['Age'] = 'Teen'

middle_aged['Age'] = 'Middle'

adult['Age'] = 'Adult'

old['Age'] = 'Old'

''''''''''''''''''''''''''''''''''''

baby_test['Age'] = 'Baby'

child_test['Age'] = 'Child'

teen_test['Age'] = 'Teen'

middle_aged_test['Age'] = 'Middle'

adult_test['Age'] = 'Adult'

old_test['Age'] = 'Old'
agednn = pd.concat([baby['Age'],child['Age'],teen['Age'],middle_aged['Age'],adult['Age'],old['Age']],axis = 0)

agednn_test = pd.concat([baby_test['Age'],child_test['Age'],teen_test['Age'],middle_aged_test['Age'],adult_test['Age'],old_test['Age']],axis = 0)
train['Age_n'] = train['Age']

test['Age_n'] = test['Age']

train.drop('Age',inplace = True,axis = 1)

test.drop('Age',inplace = True,axis = 1)
train = pd.concat([train,agednn],axis = 1)

test = pd.concat([test,agednn_test],axis=1)
test
train['Title'] = train['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

test['Title'] = test['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

train['Title'].value_counts()

def name_title_mapping(title):    

    

    if title == 'Mr':

        return 0

    elif title == 'Mrs':

        return 1

    elif title == 'Miss':

        return 2

    else:

        return 3
train['Title'] = train['Title'].apply(name_title_mapping)

test['Title'] = test['Title'].apply(name_title_mapping)

train.drop('Name',axis=1,inplace=True)

test.drop('Name',axis=1,inplace=True)
train.drop('Ticket',axis=1,inplace=True)

test.drop('Ticket',axis=1,inplace=True)
train['TotalFamilySize'] = train['SibSp']+train['Parch']

test['TotalFamilySize'] = test['SibSp']+test['Parch']
fam = train[(train['TotalFamilySize'] == 0)]

alo = train[(train['TotalFamilySize'] != 0)]

alo['TotalFamilySize']  = 'False'

fam['TotalFamilySize'] = 'True'

alone = pd.concat([fam['TotalFamilySize'],alo['TotalFamilySize']],axis = 0)

train['TotalFamilySize_'] = train['TotalFamilySize']

train= pd.concat([alone,train],axis=1)



fam_t = test[(test['TotalFamilySize'] == 0)]

alo_t = test[(test['TotalFamilySize'] != 0)]

alo_t['TotalFamilySize']  = 0

fam_t['TotalFamilySize'] = 1

alone_t = pd.concat([fam_t['TotalFamilySize'],alo_t['TotalFamilySize']],axis = 0)

test['TotalFamilySize_'] = test['TotalFamilySize']

test= pd.concat([alone_t,test],axis=1)
sex_map = {'male':1,'female':0}

train['Sex'] = train['Sex'].map(sex_map)

test['Sex'] = test['Sex'].map(sex_map)

embarked_map = {'S':1,'C':2,'Q':3}

train['Embarked'] = train['Embarked'].map(embarked_map)

test['Embarked'] = test['Embarked'].map(embarked_map)

age_map = {'Baby':0,'Child':1,'Teen':2,'Middle':3,'Adult':4,'Old':5}

train['Age'] = train['Age'].map(age_map)

test['Age'] = test['Age'].map(age_map)

one = pd.get_dummies(train['TotalFamilySize'])

one_t = pd.get_dummies(test['TotalFamilySize'])

train = pd.concat([train,one['TotalFamilySize_False']],axis=1)

one_t.columns = ['TotalFamilySize_False','T']

test = pd.concat([test,one_t[one_t.columns[0]]],axis=1)
train.drop('TotalFamilySize',axis = 1,inplace=True)

test.drop('TotalFamilySize',axis = 1,inplace=True)
X_train = train.drop('Survived',axis = 1)

y_train = train['Survived']

X_test = test

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

kfold = KFold(n_splits=10,shuffle=True,random_state=0)

test
svc = SVC(kernel='rbf',random_state=0)

score = cross_val_score(svc,X_train,y_train,cv=kfold,scoring='accuracy')

print(score)

print(score.mean())
svc_model = svc.fit(X_train,y_train)

svc_pred =svc_model.predict(test)
from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train, y_train)

Y_pred = classifier.predict(test)
from sklearn.metrics import accuracy_score

classifier.score(X_train, y_train)

classifier = round(classifier.score(X_train,y_train ) * 100, 2)

classifier
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission1.csv', index=False)
