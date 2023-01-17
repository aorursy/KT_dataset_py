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
import pandas as pd
train_data = pd.read_csv('/kaggle/input/train.csv')
train_data.describe()

train_data.head(3)

train_data.isnull().sum()
train_data['Embarked'].describe()
train_data.groupby(['Pclass','Survived']).count()['PassengerId']
train_data.groupby(['Sex','Survived']).count()['PassengerId']
train_data.shape
train_data.groupby(['SibSp','Survived']).count()['PassengerId']
train_data.loc[train_data['Fare']>300,'Fare'] = 300
train_data.boxplot(column = ['Fare'],by = ['Survived'])
train_data.groupby(['Embarked','Survived']).count()['PassengerId']
train_data['Survived'].value_counts()
#label encoding categorical values 

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le_sex = le.fit(train_data['Sex'])

le_sex.classes_

train_data = train_data.assign(le_Sex = le_sex.transform(train_data['Sex']))

'''

le_emb = le.fit(train_data['Embarked'].astype(str))

train_data['le_Embarked'] = le_emb.transform(train_data['Embarked'])'''
train_data = train_data.assign(Embarked = train_data['Embarked'].fillna('S'))



le_emb = preprocessing.LabelEncoder()

le_emb = le_emb.fit(train_data['Embarked'])

le_emb.classes_

train_data = train_data.assign(le_Embarked = le_emb.transform(train_data['Embarked']))
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators=100)

req_features = ['Pclass','SibSp','Parch','Fare','le_Sex','le_Embarked']

clf.fit(train_data[req_features],train_data['Survived'])

'''PassengerId      0

Survived         0

Pclass           0

Name             0

Sex              0

Age            177

SibSp            0

Parch            0

Ticket           0

Fare             0

Cabin          687

Embarked         2'''
test_data = pd.read_csv('/kaggle/input/test.csv')



test_data = test_data.assign(Fare = test_data['Fare'].fillna(train_data['Fare'].mean()))



test_data['le_Sex'] = le_sex.transform(test_data['Sex']) 

test_data['le_Embarked'] = le_emb.transform(test_data['Embarked']) 



test_data.isnull().sum()
test_data.loc[test_data['Fare']>300,'Fare'] = 300

test_data.boxplot(column='Fare')
train_data.shape
test_data.boxplot(column='Fare')
train_data.boxplot(column='Fare')
train_data['Fare'].hist()
from sklearn import preprocessing

from sklearn import svm



X_train = train_data[req_features]



scaler = preprocessing.StandardScaler().fit(X_train)

X_train_scale = scaler.transform(X_train)





X_test = test_data[req_features]



X_test_scale = scaler.transform(X_test)

    

svm_clf = svm.SVC(kernel='poly')

svm_clf.fit(X_train_scale, train_data['Survived'])

    

y_predict = svm_clf.predict(X_test_scale)

test_data = test_data.assign(Survived = svm_clf.predict(X_test_scale))
test_data['Survived'].value_counts()

solution_cols = ['PassengerId','Survived']

test_data.loc[:,solution_cols].to_csv('svm_pred_poly_kernel.csv',index=False)
test_data = test_data.assign(Survived = clf.predict(test_data[req_features]))

test_data.loc[:,solution_cols].to_csv('rf_pred.csv',index=False)
test_data.isnull().sum()
test_data['Survived'].value_counts()