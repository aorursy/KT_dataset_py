# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style = 'whitegrid')

sns.distributions._has_statsmodels = False # To handle RuntimeError: Selected KDE bandwidth is 0.
data_train = pd.read_csv('../input/titanic/train.csv')

data_test = pd.read_csv('../input/titanic/test.csv')
train = data_train.copy()

test = data_test.copy()
train.head()
test.head()
print(train.info())

print('\n')

print(test.info())
train['Cabin'] = train['Cabin'].str.get(0)

test['Cabin'] = test['Cabin'].str.get(0)
num_data = train[['Age', 'SibSp', 'Parch', 'Fare']]

cat_data = train[['Survived', 'Pclass', 'Sex', 'Cabin', 'Embarked']]
import statsmodels



fig, ax = plt.subplots(2, 2 ,figsize = (12,8))

fig.tight_layout(pad=5.0)

sns.distplot(ax = ax[0, 0], a = num_data['Age'].dropna())

ax[0, 0].set_title('Age', fontsize = 18)



sns.distplot(ax = ax[0, 1], a = num_data['SibSp'].dropna())

ax[0, 1].set_title('SibSp', fontsize = 18)



sns.distplot(ax = ax[1, 0], a = num_data['Parch'].dropna())

ax[1, 0].set_title('Parch', fontsize = 18)



sns.distplot(ax = ax[1, 1], a = num_data['Fare'].dropna())

ax[1, 1].set_title('Fare', fontsize = 18)



plt.show()
heatmapdata = train[['Survived', 'Age', 'SibSp', 'Parch', 'Fare']]



cormat = heatmapdata.corr()

fig, ax = plt.subplots(figsize = (8,4))

sns.heatmap(data = cormat)

plt.show()
fig, ax = plt.subplots(cat_data.shape[1], 1, figsize = (8,16))

fig.tight_layout(pad=5.0)



for i, n in enumerate(cat_data):

        sns.barplot(ax = ax[i], x = cat_data[n].fillna('NaN').value_counts().index, y = cat_data[n].fillna('NaN').value_counts())

        ax[i].set_title(n)

plt.show()
test.insert(1, 'Survived', -1)

test.info()
print('Train :\n',train.isnull().sum())

print('\n')

print('Test :\n', test.isnull().sum())
train['Age'].fillna(train['Age'].median(), inplace = True)

test['Age'].fillna(train['Age'].median(), inplace = True)



train['Fare'].fillna(train['Fare'].median(), inplace = True)

test['Fare'].fillna(train['Fare'].median(), inplace = True)



train.dropna(subset=['Embarked'] , inplace = True)
train.drop(['Cabin'], axis = 1, inplace = True)

test.drop(['Cabin'], axis = 1, inplace = True)
print('Train :\n',train.isnull().sum())

print('\n')

print('Test :\n', test.isnull().sum())
train['LastName'] = train['Name'].str.split(',', expand=True)[0]

test['LastName'] = test['Name'].str.split(',', expand=True)[0]
train.head()
train['Train'] = 1

test['Train'] = 0



alldata = pd.concat((train, test), sort = False).reset_index(drop = True)



sur_data = []

died_data = []

for index, row in alldata.iterrows():

    s = alldata[(alldata['LastName']==row['LastName']) & (alldata['Survived']==1)]

    d = alldata[(alldata['LastName']==row['LastName']) & (alldata['Survived']==0)]

    

    s=len(s)

    if row['Survived'] == 1:

        s-=1



    d=len(d)

    if row['Survived'] == 0:

        d-=1

        

    sur_data.append(s)

    died_data.append(d)

    

alldata['FamilySurvived'] = sur_data

alldata['FamilyDied'] = died_data
train = alldata[alldata['Train'] == 1]

test = alldata[alldata['Train'] == 0]
q1 = train['Age'].quantile(0.25)

q3 = train['Age'].quantile(0.75)

iqr = q3-q1

train = train[~((train['Age'] < (q1 - 1.5 * iqr)) | (train['Age'] > (q3+1.5*iqr)))]



q1=train['Fare'].quantile(0.25)

q3 = train['Fare'].quantile(0.75)

iqr = q3-q1

train = train[~ ((train['Fare'] < q1 - 1.5 * iqr) | (train['Fare'] > (q3 + 1.5 * iqr)))]
train['Fare'] = np.log1p(train['Fare'])

test['Fare'] = np.log1p(test['Fare'])
import seaborn as sns

fig, ax = plt.subplots(1, 2 ,figsize = (16,4))

sns.distplot(ax = ax[0], a = train['Age'])

sns.distplot(ax = ax[1], a = train['Fare'])

plt.show()
train.head()
test.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(train['Pclass'])

train['Pclass'] = le.transform(train['Pclass'])

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse = False, drop = 'first', categories = 'auto')

ohe.fit(train[['Sex', 'Embarked']])

ohecategory_train = ohe.transform(train[['Sex', 'Embarked']])

ohecategory_test = ohe.transform(test[['Sex', 'Embarked']])



for i in range(ohecategory_train.shape[1]):

    train['dummy_variable_' + str(i)] = ohecategory_train[:,i]



for i in range(ohecategory_test.shape[1]):

    test['dummy_variable_' + str(i)] = ohecategory_test[:,i]





print('Train shape :', train.shape)

print('Test shape :', test.shape)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(train[['Age', 'SibSp', 'Parch', 'Fare']])

train[['Age', 'SibSp', 'Parch', 'Fare']] = sc.transform(train[['Age', 'SibSp', 'Parch', 'Fare']])

test[['Age', 'SibSp', 'Parch', 'Fare']] = sc.transform(test[['Age', 'SibSp', 'Parch', 'Fare']])
train.head()
test.head()
print('Train columns :\n',train.columns)

print('Train shape : ', train.shape)

print('\n')

print('Test columns :\n',test.columns)

print('Test shape : ', test.shape)
train.head()
train.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Embarked', 'LastName', 'Train'], axis = 1, inplace = True)

test.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Embarked', 'LastName', 'Train'], axis = 1, inplace = True)
print('Train columns :\n',train.columns)

print('Train shape : ', train.shape)

print('\n')

print('Test columns :\n',test.columns)

print('Test shape : ', test.shape)
X_train = train.iloc[:, 1:].values

y_train = train.iloc[:, 0].values



X_test = test.iloc[:, 1:].values

y_test = test.iloc[:, 0].values



print('X_train :\n', X_train[0:5])

print('y_train :\n', y_train[0:5])
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



clf = KNeighborsClassifier(leaf_size = 1, metric = 'minkowski', n_neighbors = 12, p = 1, weights = 'distance')

accuracies = cross_val_score(clf, X_train, y_train, cv = 10)

print('Accuracies : ', accuracies)

print('AVG Accuracies : ', accuracies.mean())

print('STD:',accuracies.std())
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

y_pred = y_pred.astype('int64')



submission = pd.DataFrame()

submission['PassengerId'] = data_test['PassengerId']

submission['Survived'] = y_pred

submission['Survived'].value_counts()
submission.to_csv(r'Submission.csv', index = False, header = True)