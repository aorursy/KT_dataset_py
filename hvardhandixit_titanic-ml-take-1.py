# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.info()

test.info()
c = train.columns

c = list(c)

c.remove('PassengerId')

c.remove('Ticket')

c.remove('Cabin')

c.remove('Name')

c.remove('Embarked')

print(c)

for i in c:

    u = train[i].unique()

    u.sort()

    print("{} = {}".format(i, u))

print('Embarked = ',train['Embarked'].unique())
c = test.columns

c = list(c)

c.remove('PassengerId')

c.remove('Ticket')

c.remove('Cabin')

c.remove('Name')

#c.remove('Embarked')

print(c)

for i in c:

    u = test[i].unique()

    u.sort()

    print("{} = {}".format(i, u))

#print('Embarked = ',train['Embarked'].unique())
missing = train.isnull().sum()

missing_percentage = missing/891*100

pd.concat([missing, missing_percentage], 1, keys = ['Total Missing', '% Missing'])
missing2 = test.isnull().sum()

missing_p2 = missing2/418*100

pd.concat([missing2, missing_p2], 1, keys = ['Total Missing', '% Missing'])

train.head()
test[test['Fare'

].isnull()]
mean_fare = test[(test['Sex'] == 'male') & (test['Embarked'] == 'S')]['Fare'].mean()
test[test['Fare'].isnull()]['Fare']
test['Fare'].loc[test['Fare'].isnull()] = mean_fare
test[test['PassengerId'] == 1044]
train[train['Embarked'].isnull()]
sns.set_context(rc={"lines.linewidth": 0.001})

sns.set(style = 'whitegrid')

fig, ax = plt.subplots(figsize = (15,15), nrows = 2, ncols = 1)

sns.boxplot(x = 'Fare', y = 'Embarked', data = train, palette = 'Set3', order = ['Q', 'C', 'S'], hue = 'Pclass')

sns.boxplot(x = 'Fare', y = 'Embarked', data = test, ax = ax[0], palette = 'Set3', order = ['Q', 'C', 'S'], hue= 'Pclass')
test[(test['Embarked'] == 'Q') & (test['Fare'] > 70)]
train[(train['Embarked'] == 'Q') & (train['Fare'] > 70)]
test[(test['Embarked'] == 'C') & (test['Fare'] > 70)]
train[(train['Embarked'] == 'C') & (train['Fare'] > 70)]
train['Embarked'].ix[train['Embarked'].isnull()] = 'C'
print(train[(train['PassengerId'] == 62)]['Embarked'])

print(train[(train['PassengerId'] == 830)][

    'Embarked'

])
train = train.drop('Cabin', 1)

test = test.drop('Cabin', 1)
sns.barplot(train['Survived'], orient = 'v', palette = 'Set3')
sns.barplot(train['Sex'], train['Survived'], palette = 'Set3')
sns.barplot(train['Pclass'], train['Survived'], palette = 'Set3')
fig = plt.figure(figsize = (10, 10))

sns.barplot(x = 'Embarked', y = 'Survived', hue = 'Pclass', data = train, palette = 'Set3')
sns.barplot(x = 'Embarked', y = 'Survived', data = train, palette = 'Set3', order = [

    'C', 'Q', 'S'

])
fig, ax = plt.subplots(figsize = (8, 5), nrows = 1, ncols = 2)

sns.pointplot(x = 'SibSp', y  = 'Survived', data = train)

sns.pointplot(x = 'Parch', y = 'Survived', data = train, ax = ax[0])
fig, ax = plt.subplots(figsize = (12, 5), nrows = 1, ncols = 2)

sns.countplot(x = 'Parch', data = train, ax = ax[0])

sns.countplot(x = 'SibSp', data = train)
train['FamilySize'] = train['SibSp'] + train['Parch']

train.drop(['SibSp', 

           'Parch'], axis = 1, inplace = True)

test['FamilySize'] = test['SibSp'] + test['Parch']

test.drop(['SibSp', 

           'Parch'], axis = 1, inplace = True)
sns.pointplot(x = 'FamilySize', y = 'Survived', data = train)
print ('Age values missing from train = {}%'.format( train['Age'].isnull().sum()/891*100))

print ('Age values missing from test = {}%'.format( test['Age'].isnull().sum()/418*100))
train.drop(['PassengerId', 'Name', 'Ticket'], axis = 1, inplace = True)

test.drop(['PassengerId', 'Name', 'Ticket'], axis = 1, inplace = True)
Male = []

Female = []



for i in train['Sex']:

    if (i == 'male'):

        Male.append(1)

        Female.append(0)

    else:

        Male.append(0)

        Female.append(1)

Male = pd.Series(Male)

Female = pd.Series(Female)

train['Male'] = Male

train['Female'] = Female
S = []

Q = []

C = []

for i in train['Embarked']:

    if (i == 'S'):

        S.append(1)

        Q.append(0)

        C.append(0)

    elif (i == 'C'):

        S.append(0)

        Q.append(0)

        C.append(1)

    else:

        S.append(0)

        Q.append(1)

        C.append(0)

S = pd.Series(S)

Q = pd.Series(Q)

C = pd.Series(C)

train['S'] = S

train['Q'] = Q

train['C'] = C
train.drop(['Sex', 'Embarked'], axis= 1, inplace = True)
train.head()
Male = []

Female = []



for i in test['Sex']:

    if (i == 'male'):

        Male.append(1)

        Female.append(0)

    else:

        Male.append(0)

        Female.append(1)

Male = pd.Series(Male)

Female = pd.Series(Female)

test['Male'] = Male

test['Female'] = Female



S = []

Q = []

C = []

for i in test['Embarked']:

    if (i == 'S'):

        S.append(1)

        Q.append(0)

        C.append(0)

    elif (i == 'C'):

        S.append(0)

        Q.append(0)

        C.append(1)

    else:

        S.append(0)

        Q.append(1)

        C.append(0)

S = pd.Series(S)

Q = pd.Series(Q)

C = pd.Series(C)

test['S'] = S

test['Q'] = Q

test['C'] = C



test.drop(['Sex', 'Embarked'], axis = 1, inplace = True)
test.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split

from sklearn import metrics
data2 = train[train['Age'].isnull()] #All null values

data1 = train[train['Age'].notnull()] #No null values

accuracy = 0

for i in range(20):

    train1, test1 = train_test_split(data1, test_size = 0.25)

    train1_X = train1.drop('Age', axis = 1)

    train1_y = train1['Age']

    test1_X = test1.drop('Age', axis = 1)

    test1_y = test1["Age"]



    model = RandomForestRegressor()

    model.fit(train1_X, train1_y)

    p1 = model.predict(test1_X)

    p1 = list(map(int, p1))

    accuracy += model.score(test1_X, p1)

print('Accuracy = ', accuracy/20)
data2 = train[train['Age'].isnull()] #All null values

data1 = train[train['Age'].notnull()] #No null values

accuracy = 0

for i in range(20):

    train1, test1 = train_test_split(data1, test_size = 0.25)

    train1_X = train1.drop('Age', axis = 1)

    train1_y = train1['Age']

    test1_X = test1.drop('Age', axis = 1)

    test1_y = test1["Age"]



    model = KNeighborsRegressor()

    model.fit(train1_X, train1_y)

    p1 = model.predict(test1_X)

    p1 = list(map(int, p1))

    accuracy += model.score(test1_X, p1)

print('Accuracy = ', accuracy/20)
training = train[train['Age'].notnull()]

testing = train[train['Age'].isnull()]

testing = testing.drop('Age', axis = 1)

X_train = training.drop('Age', axis =1)

y_train = training['Age']

model = RandomForestRegressor()

model.fit(X_train, y_train)

p1 = model.predict(testing)

p1 = list(map(int, p1))

len(p1)
imp = list(train['Age'].loc[train['Age'].isnull()].index)

for i in range(177):

    train['Age'].iloc[imp[i]] = p1[i]
train.info()
test.info()
training = test[test['Age'].notnull()]

testing = test[test['Age'].isnull()]

testing = testing.drop('Age', axis = 1)

X_train = training.drop('Age', axis =1)

y_train = training['Age']

model = RandomForestRegressor()

model.fit(X_train, y_train)

p1 = model.predict(testing)

p1 = list(map(int, p1))

len(p1)
imp2 = list(test['Age'].loc[test['Age'].isnull()].index)

for i in range(len(imp2)):

    test['Age'].iloc[imp2[i]] = p1[i]

test.info()
fig = plt.figure(figsize = (10, 8))

sns.kdeplot(train['Age'][train['Survived'] == 1], label = 'Survived')

sns.kdeplot(train['Age'][train['Survived'] == 0], label = 'Did not survive')
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB
Algo = {}
acc = 0

for i in range(100):    

    a, b = train_test_split(train, test_size = 0.25)

    aX = a.drop('Survived', 1)

    ay = a.Survived

    bX = b.drop('Survived', 1)

    by = b.Survived

    model = RandomForestClassifier()

    model.fit(aX, ay)

    p = model.predict(bX)

    acc += metrics.accuracy_score(p, by)

print('Accuracy = ', acc/100)

Algo['RandomForestClassifier'] = acc/100

acc = 0

for i in range(100):    

    a, b = train_test_split(train, test_size = 0.25)

    aX = a.drop('Survived', 1)

    ay = a.Survived

    bX = b.drop('Survived', 1)

    by = b.Survived

    model = SVC()

    model.fit(aX, ay)

    p = model.predict(bX)

    acc += metrics.accuracy_score(p, by)

print('Accuracy = ', acc/100)

Algo['SVC'] = acc/100

acc = 0

for i in range(100):    

    a, b = train_test_split(train, test_size = 0.25)

    aX = a.drop('Survived', 1)

    ay = a.Survived

    bX = b.drop('Survived', 1)

    by = b.Survived

    model = KNeighborsClassifier()

    model.fit(aX, ay)

    p = model.predict(bX)

    acc += metrics.accuracy_score(p, by)

print('Accuracy = ', acc/100)

Algo['KNeighborsClassifier'] = acc/100

acc = 0

for i in range(100):    

    a, b = train_test_split(train, test_size = 0.25)

    aX = a.drop('Survived', 1)

    ay = a.Survived

    bX = b.drop('Survived', 1)

    by = b.Survived

    model = GaussianProcessClassifier()

    model.fit(aX, ay)

    p = model.predict(bX)

    acc += metrics.accuracy_score(p, by)

print('Accuracy = ', acc/100)

Algo['GaussianProcessClassifier'] = acc/100

acc = 0

for i in range(100):    

    a, b = train_test_split(train, test_size = 0.25)

    aX = a.drop('Survived', 1)

    ay = a.Survived

    bX = b.drop('Survived', 1)

    by = b.Survived

    model = DecisionTreeClassifier()

    model.fit(aX, ay)

    p = model.predict(bX)

    acc += metrics.accuracy_score(p, by)

print('Accuracy = ', acc/100)

Algo['DecisionTreeClassifier'] = acc/100

acc = 0

for i in range(100):    

    a, b = train_test_split(train, test_size = 0.25)

    aX = a.drop('Survived', 1)

    ay = a.Survived

    bX = b.drop('Survived', 1)

    by = b.Survived

    model = AdaBoostClassifier()

    model.fit(aX, ay)

    p = model.predict(bX)

    acc += metrics.accuracy_score(p, by)

print('Accuracy = ', acc/100)

Algo['AdaBoostClassifier'] = acc/100

acc = 0

for i in range(100):    

    a, b = train_test_split(train, test_size = 0.25)

    aX = a.drop('Survived', 1)

    ay = a.Survived

    bX = b.drop('Survived', 1)

    by = b.Survived

    model = GaussianNB()

    model.fit(aX, ay)

    p = model.predict(bX)

    acc += metrics.accuracy_score(p, by)

print('Accuracy = ', acc/100)

Algo['GaussianNB'] = acc/100

Algo
model = RandomForestClassifier()

model.fit(train.drop('Survived', 1), train.Survived)

result = model.predict(test)

new_test = pd.read_csv('../input/test.csv')

submission = pd.DataFrame({

        "PassengerId": new_test.PassengerId,

        "Survived": result

    })

submission.PassengerId = submission.PassengerId.astype(int)

submission.Survived = submission.Survived.astype(int)



submission.to_csv("Titanic_Submission_Woohoo.csv", index=False)