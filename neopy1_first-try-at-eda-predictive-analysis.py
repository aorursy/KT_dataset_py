# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

train_df.head()
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

train_df.info()
train_df['Family'] = train_df['SibSp']+train_df['Parch']

train_df['Family'] = train_df['Family'].apply(lambda x: 1 if x>0 else 0)

train_df = train_df.drop(['SibSp', 'Parch'], axis=1)

#train_df.info()
train_df['Embarked'] = train_df['Embarked'].fillna('S')
mean1 = train_df['Age'].mean()

std = train_df['Age'].std()

std 

mean1
#train_df['Age'] = train_df['Age'].apply(lambda x: np.nan if x==mean1 else x)

count = train_df['Age'].isnull().sum()

rand = np.random.randint(mean1-std,mean1+std,size=count)

train_df['Age']
import matplotlib.pyplot as plt

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

train_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

train_df['Age'][np.isnan(train_df['Age'])] = rand

train_df['Age'].astype(int).hist(bins=70, ax=axis2)
train_df.info()


test_df = pd.read_csv('../input/test.csv')

test_df['Family'] = test_df['SibSp'] + test_df['Parch']

test_df['Family'] = test_df['Family'].apply(lambda x: 1 if x>0 else 0)

test_df = test_df.drop(['SibSp', 'Parch'], axis=1)
test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
f = test_df['Fare'].mean()

fm = test_df['Fare'].std()

f,fm
test_df['Fare'] = test_df['Fare'].fillna(f)

test_df.info()
count = test_df['Age'].isnull().sum()

rand = np.random.randint(f-fm, f+fm, size=count)

test_df['Age'][np.isnan(test_df['Age'])] = rand
test_df.info()
test_df = pd.get_dummies(test_df, ['Sex', 'Embarked'])

train_df = pd.get_dummies(train_df, ['Sex', 'Embarked'])
Y_train = train_df['Survived']

X_train = train_df.drop('Survived', axis=1)

X_test = test_df

# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
test = pd.read_csv('../input/test.csv')



submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)