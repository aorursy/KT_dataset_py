# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rdn

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')

combine = train_df, test_df

test_df.head()
train_df.head()
train_df.info()
train_df.describe()
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

train_df.head()

test_df = test_df.drop(['Name', 'Ticket'], axis=1)
train_df['Embarked'] = train_df['Embarked'].fillna('S')
sns.factorplot('Embarked', 'Survived', data=train_df, size=4 , aspect=3)

fig,(axis1, axis2) = plt.subplots(1,2,figsize=(15,5))



sns.countplot(x='Embarked', data=train_df, ax=axis1)

sns.countplot(x='Survived', hue='Embarked', data=train_df, ax=axis2)

sns.countplot(x='Embarked', data=train_df, ax=axis1)
sns.countplot(x='Embarked', data = train_df,)
fig, (axis1, axis2) = plt.subplots(1,2, figsize=(15,4))

axis1.set_title('Valores de idade original')

axis2.set_title('Novos valores idade')



# pegando média, desvio padrão e NaN no dataset de treino



average_age_titanic = train_df['Age'].mean()

std_age_titanic = train_df['Age'].std()

count_nan_age_titanic = train_df['Age'].isnull().sum()



# mesma coisa com dataset de teste



average_age_test = test_df['Age'].mean()

std_age_test = test_df['Age'].std()

count_nan_age_test = test_df['Age'].isnull().sum()



# Gerando números aleatórios entre média e desvio padrão.



rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)

rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)



# desenhando plot original



train_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)



# Completando valores nulo com os valores aleatórios



train_df['Age'][np.isnan(train_df["Age"])] = rand_1

test_df['Age'][np.isnan(test_df["Age"])] = rand_2

train_df['Age'] = train_df['Age'].astype(int)

test_df['Age'] = test_df['Age'].astype(int)



train_df['Age'].hist(bins=70, ax = axis2)
train_df.drop('Cabin', axis=1, inplace=True)

test_df.drop('Cabin', axis=1, inplace=True)
train_df.head()
test_df.head()
train_df["Family"] = train_df['Parch'] + train_df['SibSp']

test_df["Family"] = test_df['Parch'] + test_df['SibSp']

train_df['Family'].loc[train_df['Family'] > 1] = 1

train_df['Family'].loc[train_df['Family'] == 0] = 0

train_df.drop(['Parch', 'SibSp'], axis=1, inplace=True)
test_df['Family'].loc[test_df['Family'] > 1] = 1

test_df['Family'].loc[test_df['Family'] == 0] = 0

test_df.drop(['Parch', 'SibSp'], axis=1, inplace=True)
sns.countplot(x='Family', data=train_df)



train_df['Sex'].loc[train_df['Sex'] == 'male'] = 1

train_df['Sex'].loc[train_df['Sex'] == 'female'] = 0

test_df['Sex'].loc[test_df['Sex'] == 'male'] = 1

test_df['Sex'].loc[test_df['Sex'] == 'female'] = 0
train_df['Embarked'].loc[train_df['Embarked'] == 'S'] = 0

train_df['Embarked'].loc[train_df['Embarked'] == 'Q'] = 1

train_df['Embarked'].loc[train_df['Embarked'] == 'C'] = 2

test_df['Embarked'].loc[test_df['Embarked'] == 'S'] = 0

test_df['Embarked'].loc[test_df['Embarked'] == 'Q'] = 1

test_df['Embarked'].loc[test_df['Embarked'] == 'C'] = 2
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
test_df["Fare"][np.isnan(test_df["Fare"])] = 7
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df['Survived']

X_test = test_df.drop('PassengerId', axis=1). copy()
X_test.isnull().sum()
X_train.dropna(inplace=True)
# logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)



logreg.score(X_train, Y_train)
#Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

svc.score(X_train, Y_train)
random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
train_df
# correlation

coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df['Coefficiente Estimate'] = pd.Series(logreg.coef_[0])



coeff_df
submission = pd.DataFrame({

    "PassengerId": test_df['PassengerId'],

    "Survived" : Y_pred

})

submission.to_csv('titanic.csv', index=False)