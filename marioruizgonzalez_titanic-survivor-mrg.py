# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Understanding the data

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

df_test.head()
# Understanding the data

df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_train.head()
# Verify quantity of rows in the dataset

print('Quantity data in -test-:')

print(df_test.shape)

print('Quantity data in -train-:')

print(df_train.shape)
# Verify data types

print('Test data types:')

print(df_test.info())

print('Train data types:')

print(df_train.info())
# Found data missing

print(pd.isnull(df_test).sum())

print(pd.isnull(df_train).sum())
# Review stadistics 

print(df_train.describe())

print(df_test.describe())
# Changing alphanumerics for numerics

# Change in sex column

df_train['Sex'].replace(['female','male'],[0,1], inplace=True)

df_test['Sex'].replace(['female','male'],[0,1], inplace=True)

# Change in embarked column

df_train['Embarked'].replace(['Q','S','C'],[0,1,2], inplace=True)

df_test['Embarked'].replace(['Q','S','C'],[0,1,2], inplace=True)
#verify change

print('replace columns')

df_test.head()
print('replace columns')

df_train.head()
# Calculate mean for replace missing data

print(df_train['Age'].mean())

print(df_test['Age'].mean())

# Replace age missing for mean

promedio = 30

df_train['Age'] = df_train['Age'].replace(np.nan, promedio)

df_test['Age'] = df_test['Age'].replace(np.nan, promedio)
# Build various groups accord to binds of ages

bins = [0,8,15,18,25,40,60,100]

names = ['1','2','3','4','5','6','7']

df_train['Age'] = pd.cut(df_train['Age'], bins, labels = names)

df_test['Age'] = pd.cut(df_test['Age'], bins, labels = names)
# Drop column cabin for a lot missing data

df_train.drop(['Cabin'], axis = 1, inplace = True)

df_test.drop(['Cabin'], axis = 1, inplace = True)
# Drop unnecessary columns

df_train = df_train.drop(['PassengerId', 'Name', 'Ticket'], axis=1 )

df_test = df_test.drop(['Name', 'Ticket'], axis=1 )
# Drop rows with missing data

df_train.dropna(axis=0, how='any', inplace=True)

df_test.dropna(axis=0, how='any', inplace=True)
# Verify data

print(pd.isnull(df_train).sum())
print(pd.isnull(df_test).sum())
print(df_train.shape)

print(df_test.shape)
df_train.head()
df_test.head()
# Separated column with survivor info

X = np.array(df_train.drop(['Survived'],1))

y = np.array(df_train['Survived'])
# Separated data of "Train" in training and test for test algorithms

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)

print('Logistic regression precision:')

print(logreg.score(X_train, y_train))

# Support Vector Machines

svc = SVC()

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

print('Support Vector Machines precision:')

print(svc.score(X_train, y_train))
# K neighbors

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

Y_pred = knn.predict(X_test)

print('K neighbors precision:')

print(knn.score(X_train, y_train))
ids = df_test['PassengerId']
### Logistic Regression

prediccion_logreg = logreg.predict(df_test.drop('PassengerId', axis=1))

out_logreg = pd.DataFrame({'PassengerId': ids, 'Survived': prediccion_logreg})

print('Logistic Regression Prediction:')

print(out_logreg.head())
## Support Vector Machines

prediccion_svc= svc.predict(df_test.drop('PassengerId', axis=1))

out_svc = pd.DataFrame({'PassengerId': ids, 'Survived': prediccion_svc})

print('Support Vector Machine Prediction:')

print(out_svc.head())
## Support Vector Machines

prediccion_knn= knn.predict(df_test.drop('PassengerId', axis=1))

out_knn = pd.DataFrame({'PassengerId': ids, 'Survived': prediccion_knn})

print('KNN Prediction:')

print(out_knn.head())