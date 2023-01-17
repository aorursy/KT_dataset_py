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
# get letter on column 'cabin' value to categorize column 'cabin'

train['Cabin'] = train['Cabin'].str.get(0)

test['Cabin'] = test['Cabin'].str.get(0)
# split into numeric and categorical data for analysis purpose

num_data = train[['Age', 'SibSp', 'Parch', 'Fare']]

cat_data = train[['Survived', 'Pclass', 'Sex', 'Cabin', 'Embarked']]
import statsmodels



fig, ax = plt.subplots(2, 2 ,figsize = (12,8))

fig.tight_layout(pad=5.0)

# can use for loop, if to much columns

sns.distplot(ax = ax[0, 0], a = num_data['Age'].dropna())

ax[0, 0].set_title('Age', fontsize = 18)



sns.distplot(ax = ax[0, 1], a = num_data['SibSp'].dropna())

ax[0, 1].set_title('SibSp', fontsize = 18)



sns.distplot(ax = ax[1, 0], a = num_data['Parch'].dropna())

ax[1, 0].set_title('Parch', fontsize = 18)



sns.distplot(ax = ax[1, 1], a = num_data['Fare'].dropna())

ax[1, 1].set_title('Fare', fontsize = 18)



plt.show()
# heatmap data numeric

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
# create columns survived so that same shape with training data

test.insert(1, 'Survived', -1)

test.info()
print('Train :\n',train.isnull().sum())

print('\n')

print('Test :\n', test.isnull().sum())
# handle missing data on column age (do the same on data test, but with median of data train)

train['Age'].fillna(train['Age'].median(), inplace = True)

test['Age'].fillna(train['Age'].median(), inplace = True)



# we know test data have nan values on fare (do the same with train data, for better understanding)

train['Fare'].fillna(train['Fare'].median(), inplace = True)

test['Fare'].fillna(train['Fare'].median(), inplace = True)



# handle missing data on embarked columns

train.dropna(subset=['Embarked'] , inplace = True)
# Drop cabin because that's have many null/nan values

train.drop(['Cabin'], axis = 1, inplace = True)

test.drop(['Cabin'], axis = 1, inplace = True)
print('Train :\n',train.isnull().sum())

print('\n')

print('Test :\n', test.isnull().sum())
# Create column family survived & died from column 'Name' (LastName)

train['LastName'] = train['Name'].str.split(',', expand=True)[0]

test['LastName'] = test['Name'].str.split(',', expand=True)[0]
train.head()
train['Train'] = 1

test['Train'] = 0



alldata = pd.concat((train, test), sort = False).reset_index(drop = True)



# From Ken Jee (https://www.youtube.com/watch?v=I3FBJdiExcg&t=1477s)

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

# Remove outlier from data train

# https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba



q1 = train['Age'].quantile(0.25)

q3 = train['Age'].quantile(0.75)

iqr = q3-q1

train = train[~((train['Age'] < (q1 - 1.5 * iqr)) | (train['Age'] > (q3+1.5*iqr)))]



q1=train['Fare'].quantile(0.25)

q3 = train['Fare'].quantile(0.75)

iqr = q3-q1

train = train[~ ((train['Fare'] < q1 - 1.5 * iqr) | (train['Fare'] > (q3 + 1.5 * iqr)))]
# Do log transform for column fare to make data more close into normal distribution

train['Fare'] = np.log1p(train['Fare']) # the same as np.log(train['Fare'] + 1)

test['Fare'] = np.log1p(test['Fare']) # the same as np.log(test['Fare'] + 1)
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

# By dropping one of the one-hot encoded columns from each categorical feature, we ensure there are no "reference" columns—the remaining columns become linearly independent.

# https://kiwidamien.github.io/are-you-getting-burned-by-one-hot-encoding.html

# https://www.youtube.com/watch?v=g9aLvY8BfRM

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
# https://benalexkeen.com/feature-scaling-with-scikit-learn/

# https://stats.stackexchange.com/questions/463690/multiple-regression-with-mixed-continuous-categorical-variables-dummy-coding-s



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(train[['Age', 'SibSp', 'Parch', 'Fare']])

train[['Age', 'SibSp', 'Parch', 'Fare']] = sc.transform(train[['Age', 'SibSp', 'Parch', 'Fare']])

test[['Age', 'SibSp', 'Parch', 'Fare']] = sc.transform(test[['Age', 'SibSp', 'Parch', 'Fare']])

train.head()
test.head()
# See if train and test data have same shape and column position

print('Train columns :\n',train.columns)

print('Train shape : ', train.shape)

print('\n')

print('Test columns :\n',test.columns)

print('Test shape : ', test.shape)
# See & explore the data for dropping unused columns/features

train.head()
# Drop columns 'Sex' and 'Embarked' because we haved one hot encode them

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