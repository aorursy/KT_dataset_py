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
from sklearn.ensemble import RandomForestClassifier

train=pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')

test=pd.read_csv('/kaggle/input/titanic/test.csv', index_col='PassengerId')

train.head()

test.head()
train['Cabin']=train['Cabin'].str.get(0)

train.Cabin.fillna("N", inplace=True)



test['Cabin']=test['Cabin'].str.get(0)

test.Cabin.fillna("N", inplace=True)
X_train=train.iloc[:,[1,3,4,5,6,8,9,10]]

y_train=train.iloc[:,0]

y_train=pd.DataFrame(y_train)



X_test=test.iloc[:,[0,2,3,4,5,7,8,9]]

X_test.head()
X_train.isnull().any()
X_train=pd.get_dummies( X_train, columns = ['Sex', 'Cabin'], drop_first=True)

X_train=pd.get_dummies( X_train, columns = ['Embarked'])

X_train=X_train.drop(columns=['Cabin_T'])



X_test=pd.get_dummies( X_test, columns = ['Sex', 'Cabin'], drop_first=True)

X_test=pd.get_dummies( X_test, columns = ['Embarked'])

X_test
X_train
X_train.Age.fillna(X_train.Age.mean(), inplace=True)

X_test.Age.fillna(X_test.Age.mean(), inplace=True)

X_test.Fare.fillna(X_test.Fare.mean(), inplace=True)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(X_train)

X_test=sc.transform(X_test)
X_train=pd.DataFrame(X_train)

X_test=pd.DataFrame(X_test)

X_test.isnull().any()
from sklearn.model_selection import GridSearchCV



RFC=RandomForestClassifier(random_state=27)



'''

***first iteration***

parameters={

    'n_estimators':[10,100, 500, 1000],

    'criterion':['gini', 'entropy'],

    'max_depth':[2,4,6,8,10]

}

results:{'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100}

'''



'''

***second iteration***

parameters={

    'n_estimators':[80, 90,100, 110, 120],

    'criterion':['entropy'],

    'max_depth':[5,6,7]

}

result: {'criterion': 'entropy', 'max_depth': 5, 'n_estimators': 80}

'''



'''

***third iteration***

parameters={

    'n_estimators':[60,70,80,85],

    'criterion':['entropy'],

    'max_depth':[4,5,6]

}

result:{'criterion': 'entropy', 'max_depth': 5, 'n_estimators': 80}

'''



parameters={

    'n_estimators':[78,79,80,81,82],

    'criterion':['entropy'],

    'max_depth':[5]

}



gs=GridSearchCV(estimator=RFC,

                param_grid=parameters,

                scoring='accuracy',

                cv=9,

                n_jobs=-1)

gs=gs.fit(X_train, y_train.values.ravel())

np=gs.best_params_

print(np)
classifier=RandomForestClassifier(n_estimators=80,

                                  criterion='entropy',

                                  max_depth=5,

                                  random_state=27)

classifier.fit(X_train, y_train.values.ravel())
y_pred = classifier.predict(X_test)

y_pred
test
submission = pd.DataFrame({'PassengerId':test.index,'Survived':y_pred})

submission.head()
submission.to_csv('RFwDS.csv',index=False)