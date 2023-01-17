import numpy as np

import pandas as pd

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
train = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')

test_X = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')
print(train.columns); print(train.shape)

print(test_X.columns); print(test_X.shape)
features = list(set(train.columns) - {'Survived'})

target = 'Survived'

train_X, train_y = train[features], train[target]
train_X.head()
print(train_X.info())
print(test_X.info())
plt.figure()

plt.title('Visualizing null values in train (nulls in white)')

sns.heatmap(train_X.isnull(), cbar=False)
null_percent = (train_X.isnull().sum() / train_X.shape[0]).values * 100

plt.figure()

plt.xlabel('Percentage of nulls')

sns.barplot(x=null_percent, y=features)
null_threshold = 0.5    # Features with more than null_threshold fraction of nulls will be dropped.

for feature in features:

    null_fraction = train_X[feature].isnull().sum() / train_X.shape[0]

    if(null_fraction > null_threshold):

        print(f'Dropped \'{feature}\' having {round(null_fraction * 100, 2)}% nulls.')

        train_X = train_X.drop(feature, axis=1)

        test_X = test_X.drop(feature, axis=1)

        features.remove(feature)
print('Number of unique values for each feature')

train_X.nunique()
numeric_features = ['Age', 'Fare', 'Parch', 'SibSp']

ordinal_features = ['Pclass']    # Label encoded

nominal_features = ['Sex', 'Ticket', 'Embarked']    # To one hot encode
fig, ax = plt.subplots(1, 2, figsize=(8, 3))

plt.tight_layout()

for i, feature in enumerate(['Age', 'Fare']):

    ax[i].set_title(f'Histogram of {feature}')

    sns.distplot(train_X[feature], ax=ax[i], kde=False)
fig, ax = plt.subplots(1, 5, figsize=(18, 3))

plt.tight_layout()

for i, feature in enumerate(['Parch', 'SibSp', 'Pclass', 'Sex', 'Embarked']):

    sns.countplot(train_X[feature], ax=ax[i])
plt.figure()

sns.pairplot(pd.concat([train_y, train_X[numeric_features]], axis=1), 

             hue='Survived', 

             vars=numeric_features,

             markers=['+', 'x'],

             diag_kind=None, 

             dropna=True,

             plot_kws={'alpha': 0.4})
plt.title('Effect of age on survival')

sns.swarmplot(x=train_y, y=train_X['Age'])
plt.title('Effect of fare on survival')

sns.swarmplot(x=train_y, y=train_X['Fare'])
survival_percent = dict(round(train.groupby(by='Sex').mean()['Survived'] * 100, 2))

print(survival_percent)
survived_males_age = train_X['Age'].where((train_X['Sex'] == 'male') & (train_y == 1))

not_survived_males_age = train_X['Age'].where((train_X['Sex'] == 'male') & (train_y == 0))

survived_females_age = train_X['Age'].where((train_X['Sex'] == 'female') & (train_y == 1))

not_survived_females_age = train_X['Age'].where((train_X['Sex'] == 'female') & (train_y == 0))
fig, ax = plt.subplots(1, 2, figsize=(9, 4))

plt.tight_layout()



ax[0].set_title('Male age hist (by survival)')

ax[0].set_xlim([0, 80])

sns.distplot(survived_males_age, ax=ax[0], kde=False, label='Survived')

sns.distplot(not_survived_males_age, ax=ax[0], kde=False, label='Not survived')

ax[0].legend()



ax[1].set_title('Female age hist (by survival)')

ax[1].set_xlim([0, 80])

sns.distplot(survived_females_age, ax=ax[1], kde=False, label='Survived')

sns.distplot(not_survived_females_age, ax=ax[1], kde=False, label='Not survived')

ax[1].legend()
survival_percent = dict(round(train.groupby(by='Parch').mean()['Survived'] * 100, 2))

print(survival_percent)
plt.figure()

plt.xlabel('Number of parents/children')

plt.ylabel('% survived')

items = survival_percent.items()

parch = [item[0] for item in items]

survival_rates = [item[1] for item in items]

sns.barplot(x=parch, y=survival_rates)
survival_percent = dict(round(train.groupby(by='SibSp').mean()['Survived'] * 100, 2))

print(survival_percent)
plt.figure()

plt.xlabel('Number of siblings/spouses')

plt.ylabel('% survived')

items = survival_percent.items()

parch = [item[0] for item in items]

survival_rates = [item[1] for item in items]

sns.barplot(x=parch, y=survival_rates)
train_X['FamilyMembers'] = train_X['Parch'] + train_X['SibSp']

test_X['FamilyMembers'] = test_X['Parch'] + test_X['SibSp']

numeric_features.append('FamilyMembers')
survival_percent = dict(round((pd.concat([train_X, train_y], axis=1)).groupby(by='FamilyMembers').mean()['Survived'] * 100, 2))

print(survival_percent)
plt.figure()

plt.xlabel('Number of family members')

plt.ylabel('% survived')

items = survival_percent.items()

parch = [item[0] for item in items]

survival_rates = [item[1] for item in items]

sns.barplot(x=parch, y=survival_rates)
survival_percent = dict(round(train.groupby(by='Pclass').mean()['Survived'] * 100, 2))

print(survival_percent)
plt.figure()

plt.xlabel('Class')

plt.ylabel('% survived')

items = survival_percent.items()

parch = [item[0] for item in items]

survival_rates = [item[1] for item in items]

sns.barplot(x=parch, y=survival_rates)
survival_percent = dict(round(train.groupby(by='Embarked').mean()['Survived'] * 100, 2))

print(survival_percent)
plt.figure()

plt.xlabel('Port of embarking')

plt.ylabel('% survived')

items = survival_percent.items()

parch = [item[0] for item in items]

survival_rates = [item[1] for item in items]

sns.barplot(x=parch, y=survival_rates)
train_titles = [name.split(',')[1].strip().split(' ')[0] for name in train_X['Name']]

test_titles = [name.split(',')[1].strip().split(' ')[0] for name in test_X['Name']]
train_X['Title'] = train_titles

test_X['Title'] = test_titles
print('Title occurrences in training set:')

print(train_X['Title'].value_counts())
print(train_X['Name'][train_X['Title'] == 'the'])    # Using logical indexing on train_X['Name']
print('Title occurrences in test set:')

print(test_X['Title'].value_counts())
def group_titles(title):

    """ Function to group low occurrence titles using the scheme above. """

    

    if(title in {'Major.', 'Col.', 'Capt.'}):

        return 'Military'

    elif(title in {'Jonkheer.', 'Lady.', 'Sir.', 'Don.', 'Dona.', 'the'}):

        return 'Royalty'

    elif(title in {'Mlle.', 'Ms.'}):

        return 'Miss.'

    elif(title == 'Mme.'):

        return 'Mrs.'

    else:

        return title
train_X['Title'] = train_X['Title'].apply(group_titles)

test_X['Title'] = test_X['Title'].apply(group_titles)
print(f'Unique titles in training set:\t{train_X.Title.unique()}')

print(f'Unique titles in test set:\t{test_X.Title.unique()}')
nominal_features.append('Title')
corr_with_fare = train_X['Age'].corr(train_X['Fare'])

corr_with_family_members = train_X['Age'].corr(train_X['FamilyMembers'])

corr_with_pclass = train_X['Age'].corr(train_X['Pclass'])



print(f'Correlation of Age with Fare = {round(corr_with_fare, 3)}')

print(f'Correlation of Age with FamilyMembers = {round(corr_with_family_members, 3)}')

print(f'Correlation of Age with Pclass = {round(corr_with_pclass, 3)}')
null_ids_train = train_X['Age'].isnull()

null_ids_train = null_ids_train[null_ids_train != False].index.tolist()    # Passenger indices where age is null in training set

len(null_ids_train)
fill_values_train = train_X.groupby(by=['Title', 'Pclass']).mean()['Age'].astype(int)

print('Age values to fill classwise, training set:')

print(fill_values_train)
for index in null_ids_train:

    title = train_X.loc[index, 'Title']

    pclass = train_X.loc[index, 'Pclass']

    train_X.loc[index, 'Age'] = fill_values_train[(title, pclass)]
print(train_X.info())
null_ids_test = test_X['Age'].isnull()

null_ids_test = null_ids_test[null_ids_test != False].index.tolist()    # Passenger indices where age is null in test set

len(null_ids_test)
fill_values_test = test_X.groupby(by=['Title', 'Pclass']).mean()['Age'].astype(int)

print('Age values to fill classwise, test set:')

print(fill_values_test)
for index in null_ids_test:

    title = test_X.loc[index, 'Title']

    pclass = test_X.loc[index, 'Pclass']

    test_X.loc[index, 'Age'] = fill_values_test[(title, pclass)]
print(test_X.info())
imputer = SimpleImputer(strategy='mean')

train_X[numeric_features] = imputer.fit_transform(train_X[numeric_features])

test_X[numeric_features] = imputer.transform(test_X[numeric_features])
imputer = SimpleImputer(strategy='most_frequent')



train_X[ordinal_features] = imputer.fit_transform(train_X[ordinal_features])

test_X[ordinal_features] = imputer.transform(test_X[ordinal_features])



train_X[nominal_features] = imputer.fit_transform(train_X[nominal_features])

test_X[nominal_features] = imputer.transform(test_X[nominal_features])
print(f'train_X null count = {train_X.isnull().sum().sum()}')

print(f'test_X null count = {test_X.isnull().sum().sum()}')
print(f'Ordinal categorical features: {ordinal_features}')

print(f'Nominal categorical features: {nominal_features}')
print(train_X['Pclass'].unique())
one_hot_encoder = OneHotEncoder(handle_unknown='error', sparse=False, drop='if_binary')



train_oe_matrix = one_hot_encoder.fit_transform(train_X[['Sex', 'Embarked', 'Title']]).astype('int')

test_oe_matrix = one_hot_encoder.transform(test_X[['Sex', 'Embarked', 'Title']]).astype('int')



print(train_oe_matrix.shape)

print(test_oe_matrix.shape)
one_hot_encoder.categories_
features = ['Sex', 'Embarked', 'Title']

ohe_column_names = list()

for i, categories in enumerate(one_hot_encoder.categories_):

    for category in categories:

        ohe_column_names.append(features[i] + '_' + category)

# Dropped first column in 'Sex', so dropping it from ohe_column_names as well

ohe_column_names.pop(0)

print(ohe_column_names)
# Converting one hot encoded matrices to dataframe

train_oe = pd.DataFrame(train_oe_matrix, index=train_X.index, columns=ohe_column_names)

test_oe = pd.DataFrame(test_oe_matrix, index=test_X.index, columns=ohe_column_names)



# Dropping columns 'Sex', 'Embarked' and 'Title' as they have been one hot encoded

train_X = train_X.drop(columns=['Sex', 'Embarked', 'Title'])

test_X = test_X.drop(columns=['Sex', 'Embarked', 'Title'])



# Adding the one hot encoded columns to train_X and test_X

train_X = pd.concat([train_X, train_oe], axis=1)

test_X = pd.concat([test_X, test_oe], axis=1)
train_X = train_X.drop(columns=['Ticket', 'Name'])

test_X = test_X.drop(columns=['Ticket', 'Name'])
fig, ax = plt.subplots(1, 2, figsize=(7, 3))

fig.suptitle('Before scaling', y=1.1)

plt.tight_layout()

ax[0].set_title('Training')

ax[1].set_title('Test')

sns.scatterplot(train_X['Age'], train_X['Fare'], ax=ax[0], hue=train_y)

sns.scatterplot(test_X['Age'], test_X['Fare'], ax=ax[1])
mean = train_X[numeric_features].mean(axis=0)

stddev = train_X[numeric_features].std(axis=0)



for numeric_feature in numeric_features:

    train_X[numeric_feature] = (train_X[numeric_feature] - mean[numeric_feature]) / stddev[numeric_feature]

    test_X[numeric_feature] = (test_X[numeric_feature] - mean[numeric_feature]) / stddev[numeric_feature]
fig, ax = plt.subplots(1, 2, figsize=(7, 3))

fig.suptitle('After scaling', y=1.1)

plt.tight_layout()

ax[0].set_title('Training')

ax[1].set_title('Test')

sns.scatterplot(train_X['Age'], train_X['Fare'], ax=ax[0], hue=train_y)

sns.scatterplot(test_X['Age'], test_X['Fare'], ax=ax[1])