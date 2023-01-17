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



from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC
m_df = pd.read_csv('/kaggle/input/titanic/train.csv')

m_df['Age'] = m_df['Age'].apply(lambda x: x//10)

m_df = m_df.drop(columns=['PassengerId', 'Name', 'SibSp', 'Parch','Ticket', 'Fare', 'Cabin'])

m_df.replace('male', 0, inplace=True)

m_df.replace('female', 1, inplace=True)

m_df.replace('C', 0, inplace=True)

m_df.replace('S', 1, inplace=True)

m_df.replace('Q', 2, inplace=True)



t_df = pd.read_csv('/kaggle/input/titanic/test.csv')

t_df.dropna(inplace=True)

t_df['Age'] = t_df['Age'].apply(lambda x: x//10)

for_file = t_df['PassengerId']

t_df = t_df.drop(columns=['PassengerId', 'Name', 'SibSp', 'Parch','Ticket', 'Fare', 'Cabin'])

t_df.replace('male', 0, inplace=True)

t_df.replace('female', 1, inplace=True)

t_df.replace('C', 0, inplace=True)

t_df.replace('S', 1, inplace=True)

t_df.replace('Q', 2, inplace=True)

m_df.dropna(inplace=True)
X = m_df.drop(columns='Survived')

y = m_df[['Survived']]

test_x = t_df
KNN = KNeighborsClassifier()

KNN.fit(X_train,y_train.ravel())

test_y = KNN.predict(X_test)

print('Accuracy -', accuracy_score(test_y, y_test))
DTC = DecisionTreeClassifier(random_state=1)

DTC.fit(X_train,y_train)

test_y = DTC.predict(X_test)

print('Accuracy -', accuracy_score(test_y, y_test))
MLP = MLPClassifier(random_state=0)

MLP.fit(X_train,y_train.ravel())

test_y = MLP.predict(X_test)

print('Accuracy -', accuracy_score(test_y, y_test))
RF = RandomForestClassifier(random_state=1, n_estimators=10)



RF.fit(X_train,y_train.ravel())

test_y = RF.predict(X_test)

print('Accuracy -', accuracy_score(test_y, y_test))
svc = SVC(gamma='scale')



svc.fit(X_train,y_train.ravel())

test_y = svc.predict(X_test)

print('Accuracy -', accuracy_score(test_y, y_test))
MLP.fit(X,np.array(y).ravel())

test_y = MLP.predict(test_x)

test_y = [int(i) for i in test_y]

file = pd.DataFrame({'PassengerId':for_file, 'Survived':test_y})

file.to_csv('submission_MLP.csv', index = False)
KNN.fit(X,np.array(y).ravel())

test_y = KNN.predict(test_x)

test_y = [int(i) for i in test_y]

file = pd.DataFrame({'PassengerId':for_file, 'Survived':test_y})

file.to_csv('submission_KNN.csv', index = False)
DTC.fit(X,y)

test_y = DTC.predict(test_x)

test_y = [int(i) for i in test_y]

file = pd.DataFrame({'PassengerId':for_file, 'Survived':test_y})

file.to_csv('submission_DTC.csv', index = False)
RF.fit(X,np.array(y).ravel())

test_y = RF.predict(test_x)

test_y = [int(i) for i in test_y]

file = pd.DataFrame({'PassengerId':for_file, 'Survived':test_y})

file.to_csv('submission_RF.csv', index = False)
svc.fit(X,np.array(y).ravel())

test_y = svc.predict(test_x)

test_y = [int(i) for i in test_y]

file = pd.DataFrame({'PassengerId':for_file, 'Survived':test_y})

file.to_csv('submission_svc.csv', index = False)