import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import scipy.stats as st

import numpy as np



train = pd.read_csv('../input/train.csv', header=0, index_col='PassengerId')

test = pd.read_csv('../input/test.csv', header=0, index_col='PassengerId')



# take out the 'Survived' column from the training data

X_train = train.drop('Survived', axis=1)



# length of the training data; used to recombine the sets

tr_len = len(X_train)



# combined data

df = X_train.append(test)
plt.figure(1, figsize=(8, 4))

plt.subplot(121)

sns.barplot(x='Sex', y='Survived', data=train)

plt.subplot(122)

sns.barplot(x='Pclass', y='Survived', data=train)

plt.show()
df['Name'].head()
df['Title'] = df['Name'].str.extract('\,\s(.*?)[.]', expand=False)

train['Title'] = df.loc[:tr_len, 'Title']
print('Unique titles in the training set only:\n{}\n'.format(train['Title'].unique()))

print('Unique titles in both sets:\n{}'.format(df['Title'].unique()))
train_f = train[train['Sex'] == 'female']

train_m = train[train['Sex'] == 'male']

plt.figure(2, figsize=(16, 6))

plt.subplot(121)

sns.countplot(train_f['Title'])

plt.subplot(122)

sns.countplot(train_m['Title'])

plt.show()
df[df['Title'] == 'Ms']
df['Title'].replace('Mme', 'Mrs', inplace=True)

df['Title'].replace(['Mlle', 'Ms'], 'Miss', inplace=True)

df['Title'].replace(['Lady', 'the Countess', 'Dona'], 'FRare', inplace=True)

df['Title'].replace(['Sir', 'Jonkheer', 'Don'], 'MRare', inplace=True)

df['Title'].replace(['Col', 'Capt', 'Major'], 'Mil', inplace=True)



# female and male doctors

df.loc[(df['Title'] == 'Dr') & (df['Sex'] == 'female'), 'Title'] = 'FRare'

df.loc[(df['Title'] == 'Dr') & (df['Sex'] == 'male'), 'Title'] = 'MRare'



# mutate the training DataFrame, for exploration

train['Title'] = df.loc[:tr_len, 'Title']



# plot titles by gender and class

train_f = train[train['Sex'] == 'female']

train_m = train[train['Sex'] == 'male']

plt.figure(3, figsize=(12, 8))

plt.subplot(231)

sns.barplot(x='Title', y='Survived',data=train_f[train_f['Pclass'] == 1])

plt.xlabel('Females, 1st Class')

plt.subplot(234)

sns.barplot(x='Title', y='Survived', data=train_m[train_m['Pclass'] == 1])

plt.xlabel('Males, 1st Class')

plt.subplot(232)

sns.barplot(x='Title', y='Survived', data=train_f[train_f['Pclass'] == 2])

plt.xlabel('Females, 2nd Class')

plt.subplot(235)

sns.barplot(x='Title', y='Survived', data=train_m[train_m['Pclass'] == 2])

plt.xlabel('Males, 2nd Class')

plt.subplot(233)

sns.barplot(x='Title', y='Survived', data=train_f[train_f['Pclass'] == 3])

plt.xlabel('Females, 3rd Class')

plt.subplot(236)

sns.barplot(x='Title', y='Survived', data=train_m[train_m['Pclass'] == 3])

plt.xlabel('Males, 3rd Class')

plt.show()
df['Title'].replace('FRare', 'Mrs', inplace=True)

df['Title'].replace('MRare', 'Mr', inplace=True)

df['Title'].replace('Mil', 'Mr', inplace=True)

df['Title'].replace('Rev', 'Mr', inplace=True)

train['Title'] = df.loc[:tr_len, 'Title']
fs_ages = train.loc[(train['Survived'] == 1) & (train['Sex'] == 'female'), 'Age'].dropna()

fd_ages = train.loc[(train['Survived'] == 0) & (train['Sex'] == 'female'), 'Age'].dropna()

ms_ages = train.loc[(train['Survived'] == 1) & (train['Sex'] == 'male'), 'Age'].dropna()

md_ages = train.loc[(train['Survived'] == 0) & (train['Sex'] == 'male'), 'Age'].dropna()



plt.figure(4, figsize=(8, 8))

plt.subplot(211)

sns.distplot(fs_ages, bins=range(81), kde=False, color='C1', label='Survived')

sns.distplot(fd_ages, bins=range(81), kde=False, color='C0', label='Died', axlabel='Female Age')

plt.legend()

plt.subplot(212)

sns.distplot(ms_ages, bins=range(81), kde=False, color='C1', label='Survived')

sns.distplot(md_ages, bins=range(81), kde=False, color='C0', label='Died', axlabel='Male Age')

plt.legend()

plt.show()
rate_by_age_m = np.zeros(80)

rate_by_age_f = np.zeros(80)

for i in range(80):

    ages = train[(train['Age'] >= i - 2) & (train['Age'] <= i + 4)]

    rate_by_age_m[i] = ages.loc[ages['Sex'] == 'male', 'Survived'].mean()

    rate_by_age_f[i] = ages.loc[ages['Sex'] == 'female', 'Survived'].mean()

plt.figure(5)

plt.plot(rate_by_age_m, label='Males')

plt.plot(rate_by_age_f, label='Females')

plt.xlabel('6-Age Window')

plt.ylabel('Survival Rate')

plt.legend()

plt.show()
df.loc[df['Title'] == 'Master', 'Age'].describe()
df.loc[(df['Title'] != 'Master') & (df['Sex'] == 'male') & (df['Age'] <= 14.5)]
# boys below age 12 are 'Master'

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 12), 'Title'] = 'Master'

# teenage boys above age 12 are 'Mister'

df.loc[(df['Title'] == 'Master') & (df['Age'] > 12), 'Title'] = 'Mr'



train['Title'] = df.loc[:tr_len, 'Title']
print('Female Survival Rate: {}'.format(train_f['Survived'].mean()))

print('Miss Survival Rate: {}'.format(train.loc[train['Title'] == 'Miss', 'Survived'].mean()))

print('Girl Survival Rate: {}'.format(train_f.loc[train_f['Age'] <= 12, 'Survived'].mean()))
df['Age'] = df.groupby('Title')['Age'].apply(lambda x: x.fillna(x.median()))

train['Age'] = df.loc[:tr_len, 'Age']

df['Child'] = df['Age'] <= 12

train['Child'] = df.loc[:tr_len, 'Child']
df['Family'] = df['SibSp'] + df['Parch']

train['Family'] = df.loc[:tr_len, 'Family']



plt.figure(6, figsize=(12, 8))

train_f = train[train['Sex'] == 'female']

train_m = train[train['Sex'] == 'male']

plt.subplot(231)

sns.countplot(train_f.loc[train_f['Pclass'] == 1, 'Family'])

plt.xlabel('Females, 1st Class')

plt.subplot(234)

sns.countplot(train_m.loc[train_m['Pclass'] == 1, 'Family'])

plt.xlabel('Males, 1st Class')

plt.subplot(232)

sns.countplot(train_f.loc[train_f['Pclass'] == 2, 'Family'])

plt.xlabel('Females, 2nd Class')

plt.subplot(235)

sns.countplot(train_m.loc[train_m['Pclass'] == 2, 'Family'])

plt.xlabel('Males, 2nd Class')

plt.subplot(233)

sns.countplot(train_f.loc[train_f['Pclass'] == 3, 'Family'])

plt.xlabel('Females, 3rd Class')

plt.subplot(236)

sns.countplot(train_m.loc[train_m['Pclass'] == 3, 'Family'])

plt.xlabel('Males, 3rd Class')

plt.show()
plt.figure(7, figsize=(12, 8))

plt.subplot(231)

sns.barplot(x='Family', y='Survived',data=train_f[train_f['Pclass'] == 1])

plt.xlabel('Females, 1st Class')

plt.subplot(234)

sns.barplot(x='Family', y='Survived', data=train_m[train_m['Pclass'] == 1])

plt.xlabel('Males, 1st Class')

plt.subplot(232)

sns.barplot(x='Family', y='Survived', data=train_f[train_f['Pclass'] == 2])

plt.xlabel('Females, 2nd Class')

plt.subplot(235)

sns.barplot(x='Family', y='Survived', data=train_m[train_m['Pclass'] == 2])

plt.xlabel('Males, 2nd Class')

plt.subplot(233)

sns.barplot(x='Family', y='Survived', data=train_f[train_f['Pclass'] == 3])

plt.xlabel('Females, 3rd Class')

plt.subplot(236)

sns.barplot(x='Family', y='Survived', data=train_m[train_m['Pclass'] == 3])

plt.xlabel('Males, 3rd Class')

plt.show()
df['FamSize'] = (df['Family'] >= 4).astype(int) + (df['Family'] > 0).astype(int)

train['FamSize'] = df.loc[:tr_len, 'FamSize']
train['Cabin'].dropna().head()
df['Deck'] = df['Cabin'].str[0]

df['Deck'].fillna('U', inplace=True)

train['Deck'] = df.loc[:tr_len, 'Deck']



train_f = train[train['Sex'] == 'female']

train_m = train[train['Sex'] == 'male']

plt.figure(8, figsize=(8, 8))

plt.subplot(221)

sns.barplot(x='Deck', y='Survived', data=train_f[train_f['Pclass'] == 1])

plt.xlabel('Females, 1st Class')

plt.ylabel('Survival Rate')

plt.subplot(222)

sns.barplot(x='Deck', y='Survived', data=train_m[train_m['Pclass'] == 1])

plt.xlabel('Males, 1st Class')

plt.ylabel('Survival Rate')

plt.subplot(223)

sns.barplot(x='Deck', y='Survived', data=train_f[train_f['Pclass'] == 2])

plt.xlabel('Females, 2nd Class')

plt.ylabel('Survival Rate')

plt.subplot(224)

sns.barplot(x='Deck', y='Survived', data=train_m[train_m['Pclass'] == 2])

plt.xlabel('Males, 2nd Class')

plt.ylabel('Survival Rate')

plt.show()
df['GoodDeck'] = df['Deck'].isin(['A', 'B', 'C', 'D', 'E', 'F'])

train['GoodDeck'] = df.loc[:tr_len, 'GoodDeck']
df['TicketSize'] = df['Ticket'].value_counts()[df['Ticket']].values

df['AdjFare'] = df['Fare'].div(df['TicketSize'])

train['AdjFare'] = df.loc[:tr_len, 'AdjFare']

plt.figure(9)

sns.boxplot(x='Pclass', y='AdjFare', data=df[df['AdjFare'] > 0])

plt.show()
df['ScFare'] = df.groupby('Pclass')['AdjFare'].apply(lambda x: x.sub(x.median()).div(x.std())).fillna(0)

train['ScFare'] = df.loc[:tr_len, 'ScFare']
plt.figure(10)

sns.distplot(df['ScFare'])

plt.show()
center = np.zeros(136)

rate_by_fare_m = np.zeros(136)

rate_by_fare_f = np.zeros(136)

for i in range(136):

    center[i] = 0.1 * i - 5.5

    fares = train[(train['ScFare'] >= 0.1 * i - 5.7) & (train['ScFare'] <= 0.1 * i - 5.3)]

    rate_by_fare_m[i] = fares.loc[fares['Sex'] == 'male', 'Survived'].mean()

    rate_by_fare_f[i] = fares.loc[fares['Sex'] == 'female', 'Survived'].mean()

plt.figure(11)

plt.plot(center, rate_by_fare_m, label='Males')

plt.plot(center, rate_by_fare_f, label='Females')

plt.xlabel('0.4-Z-Score Window')

plt.ylabel('Survival Rate')

plt.legend()

plt.show()
df[df['Embarked'].isnull()]
df.loc[df['Ticket'].str.startswith('113'), 'Embarked'].value_counts()
df['Embarked'].fillna('S', inplace=True)

train['Embarked'] = df.loc[:tr_len, 'Embarked']
df.info()
dft = df.drop(['Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Family', 'Cabin', 'Deck', 'Fare',

               'AdjFare', 'ScFare', 'Ticket', 'TicketSize', 'Embarked'], axis=1)
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier



# encode categorical variables

le = LabelEncoder()

dft['Title'] = le.fit_transform(dft['Title'])



# recreate the training and test sets

Xp_train = dft[:tr_len]

p_test = dft[tr_len:]

p_train = Xp_train.join(train[['Survived']])



# split the new training set into X and y.

X = p_train.drop('Survived', axis=1)

y = p_train['Survived']



# select parameters

rf = RandomForestClassifier(n_estimators=500)

depths = [4, 5, 6, 7]

features = [2, 3]

rf_params = {'max_depth': depths, 'max_features': features}

grid = GridSearchCV(rf, param_grid=rf_params, cv=5).fit(X, y)

print('Parameter Scores:\n{}\n'.format(pd.DataFrame(

    grid.cv_results_['mean_test_score'].reshape(len(depths), len(features)),

    index=depths, columns=features)))

print('Feature Importances:\n{}'.format(pd.Series(

    grid.best_estimator_.feature_importances_, index=X.columns)))
predicted = np.column_stack((p_test.index.values, grid.best_estimator_.predict(p_test)))

np.savetxt('prediction.csv', predicted.astype(int), fmt='%d', delimiter=',',

           header='PassengerId,Survived', comments='')