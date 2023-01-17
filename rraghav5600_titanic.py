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
df = pd.read_csv('/kaggle/input/titanic/train.csv')

t_df = pd.read_csv('/kaggle/input/titanic/test.csv')
df.head()
# df['Age'] = df['Age'].apply(lambda x: x//10)

df = df.drop(columns=['PassengerId', 'Name', 'SibSp', 'Parch','Ticket', 'Fare', 'Cabin', 'Age'])

df.replace('male', 0, inplace=True)

df.replace('female', 1, inplace=True)

df.replace('C', 0, inplace=True)

df.replace('S', 1, inplace=True)

df.replace('Q', 2, inplace=True)

df.dropna(inplace=True)

df.head()
# my_imputer = SimpleImputer()

# t_df['Age'] = my_imputer.fit_transform(np.array(t_df['Age']).reshape(-1,1))

# t_df['Age'] = t_df['Age'].apply(lambda x: x//10)

ffile = t_df['PassengerId']

t_df = t_df.drop(columns=['PassengerId', 'Name', 'SibSp', 'Parch','Ticket', 'Fare', 'Cabin', 'Age'])

t_df.replace('male', 0, inplace=True)

t_df.replace('female', 1, inplace=True)

t_df.replace('C', 0, inplace=True)

t_df.replace('S', 1, inplace=True)

t_df.replace('Q', 2, inplace=True)

t_df.head()
X = df.drop(columns='Survived')

y = df['Survived']

test_X = t_df

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.25, random_state=1)
rf = RandomForestClassifier(random_state=0, n_estimators=1)

rf.fit(X_train, y_train.ravel())

pred = rf.predict(X_test)

accuracy_score(pred, y_test)
knn = KNeighborsClassifier()

knn.fit(X_train, y_train.ravel())

pred = knn.predict(X_test)

accuracy_score(pred, y_test)
mlp = MLPClassifier(random_state=0)

mlp.fit(X_train, y_train.ravel())

pred = mlp.predict(X_test)

accuracy_score(pred, y_test)
rf.fit(X,np.array(y).ravel())

vals = rf.predict(test_X)

file = pd.DataFrame({'PassengerId':ffile, 'Survived':vals})

file.to_csv('submission_rf.csv', index = False)

file.head()
mlp.fit(X,np.array(y).ravel())

vals_m = mlp.predict(test_X)

file = pd.DataFrame({'PassengerId':ffile, 'Survived':vals_m})

file.to_csv('submission_mlp.csv', index = False)

file.head()
knn.fit(X,np.array(y).ravel())

vals_k = knn.predict(test_X)

file = pd.DataFrame({'PassengerId':ffile, 'Survived':vals_k})

file.to_csv('submission_knn.csv', index = False)

file.head()