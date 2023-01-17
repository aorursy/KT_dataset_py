import numpy as np

import pandas as pd
df = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')

df.head()
df.shape
df.info()
df.describe(include='all')
dummies = pd.get_dummies(df['Sex'])

df['Male'] = dummies['male']

df.drop('Sex', axis=1, inplace=True)
dummies = pd.get_dummies(df['Embarked'])

df[['C', 'S']] = dummies[['C','S']]

df.drop('Embarked', axis=1, inplace=True)
df.drop(['Name', 'Ticket', 'Cabin'], inplace=True, axis=1)

df.head()
df.corr()
y = df['Survived']

X = df.drop(['Survived'], axis=1)
X.fillna(X['Age'].quantile(.5), inplace=True)
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
X = StandardScaler().fit(X).transform(X.astype(float))
knn = KNeighborsClassifier(n_neighbors=14, n_jobs=-1)

knn_score = cross_val_score(knn, X, y, cv=4, scoring='accuracy')

print('KNN(14) average:', knn_score.mean())

print(knn_score)
svc = SVC()

svc_score = cross_val_score(svc, X, y, cv=4, scoring='accuracy')

print('SVC(rbf):', svc_score.mean())

print(svc_score)
test = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')

test.head()
# Get dummies for Sex

dummies = pd.get_dummies(test['Sex'])

test['Male'] = dummies['male']

test.drop('Sex', axis=1, inplace=True)



# Get dummies for Embarked

dummies = pd.get_dummies(test['Embarked'])

test[['C', 'S']] = dummies[['C','S']]

test.drop('Embarked', axis=1, inplace=True)



# Drop Name, Ticket and Cabin

test.drop(['Name', 'Ticket', 'Cabin'], inplace=True, axis=1)

test.head()
# fill null values



test.Age.fillna(test['Age'].quantile(.5), inplace=True)

test.Fare.fillna(test['Fare'].mean(), inplace=True)

test.describe()
knn.fit(X, y)

result = knn.predict(test)

result[:10]
result_series = pd.Series(result, index=test.index, name='Survived')

result_series.head()
result_series.to_csv('../working/titanic_result.csv', header='Survived')
svc.fit(X,y)

svc_result = svc.predict(test)

svc_result[:10]
result_series = pd.Series(svc_result, index=test.index, name='Survived')

result_series.head()
result_series.to_csv('../working/titanic_result2.csv', header='Survived')
'''

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: 

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

'''