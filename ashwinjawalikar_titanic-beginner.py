import pandas as pd

pd.options.display.max_columns=None

import numpy as np

from matplotlib import pyplot as plt

from matplotlib import style

import matplotlib.gridspec as gridspec

%matplotlib inline

style.use('bmh')

import seaborn as sns

sns.set_style('dark')

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder, RobustScaler, MinMaxScaler, StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

import csv
with open('../input/titanic/train.csv', 'r', encoding='utf-8') as f:

    train = pd.read_csv(f)

    train.name = 'Train'

    

with open('../input/titanic/test.csv', 'r', encoding='utf-8') as f:

    test = pd.read_csv(f)

    test.name = 'Test'

    

train.head(10) # train set
test.head(10) # test set
train.info()

print('\n', '=' * 40, '\n')

test.info()
print('For Train: ')

train.describe(include='all') # looking at all columns
print('For Test; ')

test.describe(include='all')
train[['Pclass', 'Survived']].groupby('Pclass').mean().sort_values(by='Survived', ascending=False) # with Pclass
train[['Sex', 'Survived']].groupby('Sex').mean().sort_values(by='Survived', ascending=False) # with gender
train[['Parch', 'Survived']].groupby('Parch').mean().sort_values(by='Survived', ascending=False) # with parents or children
train[['SibSp', 'Survived']].groupby('SibSp').mean().sort_values(by='Survived', ascending=False) # with siblings or spouses
train[['Embarked', 'Survived']].groupby('Embarked').mean().sort_values(by='Survived', ascending=False) # with embarked location
age = train.groupby('Survived')['Age']



fig = plt.figure(figsize=(12, 7))

sns.histplot(age.get_group(0), color='red', alpha=0.7, label='Dead')

sns.histplot(age.get_group(1), color='blue', alpha=0.7, label='Survived')

plt.title('Age of the Survived')

plt.legend(loc='best', fontsize=14)

plt.tight_layout()
pclass_age = train.groupby(['Pclass', 'Survived', 'Sex'])['Age'] # age of passengers based on survivability, sex and pclass





fig = plt.figure(figsize=(13, 8.5))

grid = gridspec.GridSpec(nrows=6, ncols= 4, figure=fig)



for x in np.arange(2, 7, 2):

    ax1 = fig.add_subplot(grid[x-2:x, :2])

    sns.histplot(pclass_age.get_group((x/2, 1, 'male')), color='blue', alpha=0.6, bins=20, ax=ax1)

    sns.histplot(pclass_age.get_group((x/2, 0, 'male')), color='red', alpha=0.7, bins=20, ax=ax1)

    plt.title(f'Pclass = {int(x/2)} | Survived = 1/0 for Males')

    plt.tight_layout()

    if x != 6:

        ax1.xaxis.set_visible(False)



    ax2 = fig.add_subplot(grid[x-2:x, 2:], sharey=ax1)

    sns.histplot(pclass_age.get_group((x/2, 1, 'female')), color='blue', alpha=0.6, bins=20, ax=ax2, label='Survived')

    sns.histplot(pclass_age.get_group((x/2, 0, 'female')), color='red', alpha=0.6, bins=20, ax=ax2, label='Dead')

    plt.title(f'Pclass = {int(x/2)} | Survived = 1/0 for Females')

    ax2.yaxis.set_visible(False)

    plt.tight_layout()

    if x != 6:

        ax2.xaxis.set_visible(False)

    if x == 2:

        plt.legend(loc='best', fontsize=15)





plt.show()
embarked_age = train.groupby(['Embarked', 'Survived', 'Sex'])['Age'] # age of passengers based on sex, survivablity and departure location

embarked_loc = list(train.Embarked.unique())[:3]

embarked_loc.insert(0, 'dummy value')



fig = plt.figure(figsize=(13, 8.5))

grid = gridspec.GridSpec(nrows=6, ncols= 4, figure=fig)





for x in np.arange(2, 7, 2):

    ax1 = fig.add_subplot(grid[x-2:x, :2])

    sns.histplot(embarked_age.get_group((embarked_loc[int(x/2)], 1, 'male')), color='blue', alpha=0.6, bins=20, ax=ax1)

    sns.histplot(embarked_age.get_group((embarked_loc[int(x/2)], 0, 'male')), color='red', alpha=0.6, bins=20, ax=ax1)

    plt.title(f'Embarked = {embarked_loc[int(x/2)]} Survived = 1/0 for Males')

    plt.tight_layout()

    if x != 6:

        ax1.xaxis.set_visible(False)

        

    ax2 = fig.add_subplot(grid[x-2:x, 2:], sharey=ax1)

    sns.histplot(embarked_age.get_group((embarked_loc[int(x/2)], 1, 'female')), color='blue', alpha=0.6, bins=20, ax=ax2, label='Survived')

    sns.histplot(embarked_age.get_group((embarked_loc[int(x/2)], 0, 'female')), color='red', alpha=0.6, bins=20, ax=ax2, label='Dead')

    plt.title(f'Embarked = {embarked_loc[int(x/2)]} Survived = 1/0 for Females')

    plt.tight_layout()

    ax2.yaxis.set_visible(False)

    if x != 6:

        ax2.xaxis.set_visible(False)

    if x == 2:

        plt.legend(loc='best', fontsize=15)

        

plt.show()
fare_survived = train.groupby(['Embarked', 'Survived'])[['Fare', 'Sex']]

embarked_loc = list(train.Embarked.unique())[:3]

embarked_loc.insert(0, 'dummy value')



fig = plt.figure(figsize=(13, 8.5))

grid = gridspec.GridSpec(nrows=6, ncols= 4, figure=fig)





for x in np.arange(2, 7, 2):

    ax1 = fig.add_subplot(grid[x-2:x, :2])

    sns.barplot(x=fare_survived.get_group((embarked_loc[int(x/2)], 0)).Sex, 

                        y=fare_survived.get_group((embarked_loc[int(x/2)], 0)).Fare, ci=None, palette=['red', 'blue'], alpha=0.7, ax=ax1)

    plt.title(f'Embarked = {embarked_loc[int(x/2)]} | Dead')

    plt.tight_layout()

    plt.ylim([0, 100])

    if x != 6:

        ax1.xaxis.set_visible(False)

        

    ax2 = fig.add_subplot(grid[x-2:x, 2:])

    sns.barplot(x=fare_survived.get_group((embarked_loc[int(x/2)], 1)).Sex, 

                        y=fare_survived.get_group((embarked_loc[int(x/2)], 1)).Fare, ci=None, palette=['red', 'blue'], alpha=0.7, ax=ax2)

    plt.title(f'Embarked = {embarked_loc[int(x/2)]} | Survived')

    plt.tight_layout()

    plt.ylim([0, 100])

    ax2.yaxis.set_visible(False)

    if x != 6:

        ax2.xaxis.set_visible(False)

    

    

plt.show()
plt.figure(figsize=(12, 8))

sns.barplot(y='Fare', x='Pclass', data=train, hue='Survived', ci=None, palette=['red', 'blue'], alpha=0.6)

plt.title('Fare price between classes and embarkment location')

plt.tight_layout()

plt.legend(fontsize=30)

plt.show()
print('Before:', train.shape, test.shape)

train, test = train.drop(['Cabin', 'Ticket'], axis=1), test.drop(['Cabin', 'Ticket'], axis=1) # dropped Cabin and Ticket features

print('After:', train.shape, test.shape)
datasets = [train, test] # combining so changes apply to both tables

for dataset in datasets:

    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False) # extracting titles from names

    

pd.crosstab(train.Title, train.Sex)
# replacing specifc mistyped titles with correct ones

for dataset in datasets:

    dataset.Title = dataset.Title.replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'})



# grouping names with a very low count

rare_names = ['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir', 'Dona']

for dataset in datasets:

    dataset.Title = dataset.Title.replace(rare_names, 'Rare')

pd.crosstab(train.Title, train.Sex)
train[['Title', 'Survived']].groupby('Title').mean() # we can observe the survivability rate for each title
title_conv = LabelEncoder().fit(train.Title)# preparing to attach numbers to each unique title

sex_conv = LabelEncoder().fit(train.Sex) # converting to numerical gender values



for dataset in datasets:

    dataset.Title = title_conv.transform(dataset.Title) + 1

    dataset.Sex = sex_conv.transform(dataset.Sex) + 1

    

train.head(10)
train = train.drop(['PassengerId', 'Name'], axis=1)

test = test.drop(['PassengerId', 'Name'], axis=1)

    

print(train.shape, test.shape)
grid = sns.FacetGrid(train, col='Pclass', row='Sex', aspect=1.5, height=4.5)

grid.map(sns.histplot, 'Age', color='black', bins=20, alpha=0.7)

plt.tight_layout()

plt.show()
# taking the median of every sub group of Pclass and Sex and getting their age to the nearest 0.5 and converting the column to int datatype

train.Age = train.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(int(x.median() / 0.5 + 0.5) * 0.5)).astype(int)

test.Age = test.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(int(x.median() / 0.5 + 0.5) * 0.5)).astype(int)

train.head()
train['AgeBand'] = pd.qcut(train.Age, 5)

test['AgeBand'] = pd.qcut(test.Age, 5)

    

train[['AgeBand', 'Survived']].groupby('AgeBand').mean() # survival rate per age band
train.Age = LabelEncoder().fit_transform(train.AgeBand) + 1

test.Age = LabelEncoder().fit_transform(test.AgeBand) + 1



# once we are done creating categorical age features, we can drop the AgeBand feature for both datasets

train.drop('AgeBand', axis=1, inplace=True)

test.drop('AgeBand', axis=1, inplace=True)

train.head()
train['FamilySize'] = train.SibSp + train.Parch

test['FamilySize'] = test.SibSp + test.Parch



train[['FamilySize', 'Survived']].groupby('FamilySize').mean().sort_values(by='Survived', ascending=False)
train.head(10)
train['IsAlone'] = train.FamilySize.transform(lambda x: 1 if x == 0 else 0)

test['IsAlone'] = test.FamilySize.transform(lambda x: 1 if x == 0 else 0)



train.sample(10)
train[['IsAlone', 'Survived']].groupby('IsAlone').mean() # survival rate for passengers who were and weren't alone
print('Train before:', train.shape, '\nTest before', test.shape, '\n')

train.drop(['SibSp', 'Parch', 'FamilySize'], axis=1, inplace=True)

test.drop(['SibSp', 'Parch', 'FamilySize'], axis=1, inplace=True)

print('Train after:', train.shape, '\nTest after', test.shape)
train.head()
train[['Survived', 'Embarked', 'Pclass']].groupby(['Pclass', 'Embarked']).mean()
fig = plt.figure(figsize=(9, 7))

sns.barplot(x='Pclass', y='Survived', data=train, hue='Embarked', ci=None)

plt.legend(loc='best', title='Embarked', fontsize=15)

plt.title('Survival rate between Embarked location and Pclass')

plt.tight_layout()

plt.show()
train.Embarked = train.groupby('Pclass')['Embarked'].transform(lambda x: x.fillna(x.mode()[0]))

train.Embarked.isnull().sum() #  embarked feature having no more empty values
train[['Embarked', 'Survived']].groupby('Embarked').mean()
embarked_conv = LabelEncoder().fit(train.Embarked)

test.Embarked = embarked_conv.transform(test.Embarked) + 1

train.Embarked = embarked_conv.transform(train.Embarked) + 1
train.head(10)
test.Fare = test[['Pclass', 'Fare']].groupby('Pclass').transform(lambda x: x.fillna(x.median()))

test.Fare.isnull().sum()
train['FareBand'] = pd.qcut(train.Fare, 5)

test['FareBand'] = pd.qcut(test.Fare, 5)

train[['FareBand', 'Survived']].groupby('FareBand').mean()
#### LabelEncoding any FareBand features we make for both datasets

train['Fare'] =  LabelEncoder().fit_transform(train.FareBand) + 1

test['Fare'] = LabelEncoder().fit_transform(test.FareBand) + 1



train.drop('FareBand', axis=1, inplace=True) # dropping the FareBand feature as we have no use for it anymore

test.drop('FareBand', axis=1, inplace=True)                                



train.head()
train.info()

print('\n', '=' * 40, '\n')

test.info()
train['AgePclass'] = train.Age * train.Pclass # Age and Pclass

test['AgePclass'] = test.Age * test.Pclass



train['FareEmbarked'] = train.Fare * train.Embarked # Fare and Embarked

test['FareEmbarked'] = test.Fare * test.Embarked



test.head()
X_train = train.drop('Survived', axis=1).copy()

y_train = train['Survived']

X_test = test.copy()



X_train[:5]
stan_scaler = StandardScaler().fit_transform(X_train) # standard scaler

rob_scaler = RobustScaler().fit_transform(X_train) # robust scaler

minmax_scaler = MinMaxScaler().fit_transform(X_train) # min max scaler
stan_log = np.round(LogisticRegression().fit(stan_scaler, y_train).score(stan_scaler, y_train) * 100, 2)

rob_log = np.round(LogisticRegression().fit(rob_scaler, y_train).score(rob_scaler, y_train) * 100, 2)

minmax_log = np.round(LogisticRegression().fit(minmax_scaler, y_train).score(minmax_scaler, y_train) * 100, 2)



print(f'With StandardScaler the accruacy is = {stan_log}%\nWith RobustScaler the accuracy is ={rob_log}%\nWith MinMaxScaler the accuracy is = {minmax_log}%')
log_score = 0

best_c = 0



for x in np.arange(0.01, 5, 0.01):

    clf = LogisticRegression(C=x).fit(minmax_scaler, y_train)

    clf_score = np.round(clf.score(minmax_scaler, y_train) * 100, 3)

    if clf_score > log_score:

        best_c = x

        log_score = clf_score

        

print('Highest score for LogisticRegression is {0}% with C of {1}'.format(log_score, best_c))
k_score = 0

best_k = 0



for x in np.arange(1, 51, 1):

    clf = KNeighborsClassifier(n_neighbors=x).fit(minmax_scaler, y_train)

    clf_score = np.round(clf.score(minmax_scaler, y_train) * 100, 3)

    if clf_score > k_score:

        k_score = clf_score

        best_k = x

        

print('Highest score for KNeighborsClassifier is {0}% with K of {1}'.format(k_score, best_k))
svc_score = 0

best_c = 0



for x in np.arange(0.01, 5, 0.01):

    clf = SVC(C=x).fit(minmax_scaler, y_train)

    clf_score = np.round(clf.score(minmax_scaler, y_train) * 100, 3)

    if clf_score > svc_score:

        svc_score = clf_score

        best_c = x

        

print('Highest score for SVM is {0}% with C of {1}'.format(svc_score, best_c))
dec_score = np.round(DecisionTreeClassifier().fit(minmax_scaler, y_train).score(minmax_scaler, y_train) * 100, 3)

print('Highst score for DecisionTreeClassifier is {}%'.format(dec_score))
forest_score = 0

forest_trees = 0



for x in np.arange(100, 500, 100):

    clf = RandomForestClassifier(n_estimators=x).fit(minmax_scaler, y_train)

    clf_score = np.round(clf.score(minmax_scaler, y_train) * 100, 3)

    if clf_score > forest_score:

        forest_score = clf_score

        forest_trees = x

        

print('Highest score for RandomForestClassifier is {0}% with {1} trees'.format(forest_score, forest_trees))
models = pd.DataFrame({'Models':['LogisticRegression', 'KNeighborsClassifier', 'Support Vector Machine', 

                                                  'DecisionTreeClassifier', 'RandomForestClassifier'], 

                       'Score':[log_score, k_score, svc_score, dec_score, forest_score]}).sort_values(by='Score', ascending=False)



models
test_scaled = MinMaxScaler().fit_transform(X_test)



predicts = RandomForestClassifier(n_estimators=100).fit(minmax_scaler, y_train).predict(test_scaled)

predicts[:5]
i = list(range(892, 942, 1))

rows = list(zip(i, predicts))

rows[:5]
with open('submission.csv', 'w', newline='') as csvfile:

    writer = csv.writer(csvfile)

    writer.writerows(rows)