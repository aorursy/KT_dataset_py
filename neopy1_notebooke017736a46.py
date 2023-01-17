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
train_df.info()
#train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean())
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

train_df.info()
train_df['Family'] = train_df['SibSp']+train_df['Parch']

train_df['Family'] = train_df['Family'].apply(lambda x: 1 if x>0 else 0)

train_df = train_df.drop(['SibSp', 'Parch'], axis=1)

#train_df.info()
train_df['Family']
train_df['Embarked'].value_counts()
train_df['Embarked'] = train_df['Embarked'].fillna('S')
by_age = train_df.groupby('Age')['Survived'].count()

by_age.plot(kind='bar')
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
X_train = train_df.drop("Survived",axis=1)

Y_train = train_df['Survived']
test_df = pd.read_csv('../input/test.csv')

test_df['Family'] = test_df['SibSp'] + test_df['Parch']

test_df['Family'] = test_df['Family'].apply(lambda x: 1 if x>0 else 0)

test_df = test_df.drop(['SibSp', 'Parch'], axis=1)
ohc.n_values_
train_df1 = train_df

Y = train_df1['Survived'].values

train_df1 = train_df1.drop(['Survived'],axis=1)

x = train_df1.values
test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test_df.info()
test_df[np.isnan(test_df['Fare'])]
fare = test_df[(test_df['Pclass'] == 3) & (test_df['Embarked'] == 'S') & (test_df['Family'] == 0)]

std = test_df[(test_df['Pclass'] == 3) & (test_df['Embarked'] == 'S')]['Fare'].std()

fare
f = fare['Fare'].mean()

fm = fare['Fare'].std()

f,fm
test_df['Fare'] = test_df['Fare'].fillna(f)

test_df.info()
count = test_df['Age'].isnull().sum()

rand = np.random.randint(f-fm, f+fm, size=count)

test_df['Age'][np.isnan(test_df['Age'])] = rand
test_df.info()
from sklearn.decomposition import PCA

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import RFECV, SelectKBest

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression, Lasso

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB, BernoulliNB

from sklearn.neighbors import KNeighborsClassifier

classifiers = [('RandomForestClassifierG', RandomForestClassifier(n_jobs=-1, criterion='gini')),

               ('RandomForestClassifierE', RandomForestClassifier(n_jobs=-1, criterion='entropy')),

               ('AdaBoostClassifier', AdaBoostClassifier()),

               ('ExtraTreesClassifier', ExtraTreesClassifier(n_jobs=-1)),

               ('KNeighborsClassifier', KNeighborsClassifier(n_jobs=-1)),

               ('DecisionTreeClassifier', DecisionTreeClassifier()),

               ('ExtraTreeClassifier', ExtraTreeClassifier()),

               ('LogisticRegression', LogisticRegression()),

               ('GaussianNB', GaussianNB()),

               ('BernoulliNB', BernoulliNB())

              ]

allscores = []

#x, Y = mod_df.drop('ParentschoolSatisfaction', axis=1), np.asarray(mod_df['ParentschoolSatisfaction'], dtype="|S6")

for name, classifier in classifiers:

    scores = []

    for i in range(20): # 20 runs

        roc = cross_val_score(classifier, x, Y)

        scores.extend(list(roc))

    scores = np.array(scores)

    print(name, scores.mean())

    new_data = [(name, score) for score in scores]

    allscores.extend(new_data)


train_df = pd.get_dummies(train_df, ['Sex', 'Embarked'])
train_df.head()
train_df1 = train_df[:446, :]

test_df1 = train_df[447:, :]
