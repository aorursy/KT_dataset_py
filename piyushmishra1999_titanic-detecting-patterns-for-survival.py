# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting

import seaborn as sns # even more plotting

from sklearn.preprocessing import LabelEncoder, OneHotEncoder # preprocessing

from sklearn.preprocessing import StandardScaler # more preprocessing

from sklearn.ensemble import RandomForestClassifier





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
training_set = pd.read_csv('../input/train.csv')

training_set.head()
training_set.info()
training_set[training_set.Age.isnull()]
age = training_set.Age

sns.distplot(age.dropna())
print(age.mean())

print(age.median())

print(age.mode())

print(age.std())
age = age.fillna(int(training_set.Age.median()))
sns.distplot(age)
print(age.mean())

print(age.median())

print(age.mode())

print(age.std())
age = training_set.Age.interpolate()
sns.distplot(age)
print(age.mean())

print(age.median())

print(age.mode())

print(age.std())
age = training_set.Age.interpolate(method = 'polynomial', order = 2)
sns.distplot(age)
print(age.mean())

print(age.median())

print(age.mode())

print(age.std())
age[age < 0]
training_set.Age = training_set.Age.interpolate()
training_set.info()
training_set[training_set.Embarked.isnull()]
training_set.Embarked = training_set.Embarked.fillna('S')

training_set.info()
training_set[training_set.Cabin.isnull()].head()
(1 - 204/891)*100
training_set[training_set.Cabin.isnull() == False].head()
training_set.Cabin.value_counts().head(10)
training_set = training_set.drop('Cabin', 1)

training_set.info()
corr = training_set.corr()

cmap = sns.diverging_palette(220, 10, as_cmap=True)

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
corr
test_set = pd.read_csv('../input/test.csv')

test_set.head()
test_set.info()
test_set.Age = test_set.Age.interpolate()

test_set = test_set.drop('Cabin', 1)

test_set.info()
test_set[test_set.Fare.isnull()]
sns.distplot(test_set.Fare[test_set.Pclass == 3].dropna())

print(test_set.Fare[test_set.Pclass == 3].mode())
test_set.Fare = test_set.Fare.fillna(7.75)

test_set.info()
sns.distplot(training_set.Survived)
corr.Survived
sns.countplot(x = 'Pclass', hue = 'Survived', data = training_set)

plt.ylabel('Passenger Count')
age = pd.qcut(training_set.Age, 8)

plt.figure(figsize = (8, 5))

sns.countplot(x = age, hue = training_set.Survived)
sns.countplot(x = 'Sex', hue = 'Survived', data = training_set)

plt.ylabel('Passenger Count')
plt.figure(figsize = (10, 5))

plt.subplot(1, 2, 1)

sns.countplot(x = 'Parch', hue = 'Survived', data = training_set)

plt.ylabel('Passenger Count')

plt.xlabel('No. of parents/children')

plt.subplot(1, 2, 2)

sns.countplot(x = 'SibSp', hue = 'Survived', data = training_set)

plt.ylabel('Passenger Count')

plt.xlabel('Siblings/Spouse')

plt.tight_layout()
training_set.Ticket.value_counts().head(10)
training_set['Ticket_Frequency'] = training_set.groupby('Ticket')['Ticket'].transform('count')

sns.countplot(x = 'Ticket_Frequency', hue = 'Survived', data = training_set)
test_set['Ticket_Frequency'] = test_set.groupby('Ticket')['Ticket'].transform('count')
training_set = training_set.drop(['Name', 'PassengerId'], 1)

training_set.info()
test_set = test_set.drop(['Name', 'PassengerId'], 1)

test_set.info()
training_set = training_set.drop('Ticket', 1)

test_set = test_set.drop('Ticket', 1)
label_encoder = LabelEncoder()

training_set.Sex = label_encoder.fit_transform(training_set.Sex)

test_set.Sex = label_encoder.transform(test_set.Sex)
label_encoder = LabelEncoder()

training_set.Embarked = label_encoder.fit_transform(training_set.Embarked)

test_set.Embarked = label_encoder.transform(test_set.Embarked)
training_set['Pclass1'] = pd.get_dummies(training_set.Pclass)[1]

training_set['Pclass2'] = pd.get_dummies(training_set.Pclass)[2]



training_set['Embarked0'] = pd.get_dummies(training_set.Embarked)[0]

training_set['Embarked1'] = pd.get_dummies(training_set.Embarked)[1]



training_set.head()
test_set['Pclass1'] = pd.get_dummies(test_set.Pclass)[1]

test_set['Pclass2'] = pd.get_dummies(test_set.Pclass)[2]



test_set['Embarked0'] = pd.get_dummies(test_set.Embarked)[0]

test_set['Embarked1'] = pd.get_dummies(test_set.Embarked)[1]



test_set.head()
training_set = training_set.drop(columns = ['Pclass', 'Embarked'], axis = 1)
test_set = test_set.drop(columns = ['Pclass', 'Embarked'], axis = 1)
X_train = training_set.iloc[:, 1:]

y_train = training_set.Survived

X_test = test_set
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
X_train
model = RandomForestClassifier(criterion='gini', 

                            n_estimators=1100,

                            max_depth=5,

                            min_samples_split=4,

                            min_samples_leaf=5,

                            max_features='auto',

                            n_jobs=-1,

                            verbose=1)
model.fit(X_train, y_train)
model.score(X_train, y_train)