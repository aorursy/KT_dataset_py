import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



sns.set()

pd.set_option('display.expand_frame_repr', False)
# loading the datasets

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
train.head()
train.info()
train.isnull().sum().sort_values(ascending=False)
def bar_chart(feature):

    survived = train[train['Survived'] == 1][feature].value_counts()

    dead = train[train['Survived'] == 0][feature].value_counts()

    df = pd.DataFrame([survived, dead])

    df.index = ['Survived', 'Dead']

    df.plot(kind='bar', stacked=True, figsize=(10, 5))
# Pclass

bar_chart('Pclass')
fig, ax = plt.subplots(nrows= 2, ncols=2, figsize=(16, 10))



plots = []

plots.append(train.groupby('Survived')['Survived'].count().plot(kind='bar', ax=ax[0][0], title='Total Survied/Dead', color=['r', 'g']))

plots.append(train.groupby('Sex')['Sex'].count().plot(kind='bar', ax=ax[0][1], title='Total Male/Female'))

plots.append(train.loc[train['Survived'] == 1].groupby('Sex')['Sex'].count().plot(kind='bar', ax=ax[1][0], title='Male/Female Survived'))

plots.append(train.loc[train['Survived'] == 0].groupby('Sex')['Sex'].count().plot(kind='bar', ax=ax[1][1], title='Male/Female Died'))





for plot in plots:

    total = 0



    for index, i in enumerate(plot.patches):

        total += i.get_height()

        

    plot.text(0, 10, "Total Passengers: " + str(total), bbox=dict(color='white'))

    

    for index, i in enumerate(plot.patches):

        perc = str(round(i.get_height()/total * 100, 2)) + '%'

        plot.annotate(perc, xy=(0, 0.5), xytext=(index, i.get_height()))
facet = sns.FacetGrid(train, hue="Survived", aspect=5, margin_titles=True)

facet.map(sns.kdeplot, 'Age', shade=True)

facet.set(xlim=(0, train['Age'].max()), xlabel='Age (years)', 

          title='Kernel Density Estimation (Age and Survival Rate)', ylabel='Density')

facet.add_legend()



plt.xticks(range(0, int(train['Age'].max()), 5))

plt.show()
facet = sns.FacetGrid(train, hue="Survived", aspect=5, margin_titles=True)

facet.map(sns.kdeplot, 'Fare', shade=True)

facet.set(xlim=(0, train['Fare'].max()), xlabel='Fare (pounds)', 

          title='Kernel Density Estimation (Fare and Survival Rate)', ylabel='Density')

facet.add_legend()
facet = sns.FacetGrid(train, hue="Survived", aspect=4)

facet.map(sns.kdeplot, 'Fare', shade=True)

facet.set(xlim=(0, 100))

facet.add_legend()



plt.xticks(range(0, 100, 10))

plt.show()
# SibSp

facet = sns.FacetGrid(train, hue="Survived",aspect=5)

facet.map(sns.kdeplot,'SibSp',shade= True)

facet.set(xlim=(0, train['SibSp'].max()))

facet.add_legend()
# Parch

facet = sns.FacetGrid(train, hue="Survived",aspect=5)

facet.map(sns.kdeplot,'Parch',shade= True, bw=1.5)

facet.set(xlim=(0, train['Parch'].max()))

facet.add_legend()
fig, ax = plt.subplots(nrows= 2, ncols=2, figsize=(20, 12))

plots = []



Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()

Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()

Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

df.index = ['1st class','2nd class', '3rd class']



train.groupby('Embarked')['Embarked'].count().plot(kind='bar', ax=ax[0][0], title='Total Embarked')

train.loc[train['Survived'] == 0].groupby('Embarked')['Embarked'].count().plot(kind='bar', ax=ax[0][1], title='Total Embarked | Died')

train.loc[train['Survived'] == 1].groupby('Embarked')['Embarked'].count().plot(kind='bar', ax=ax[1][0], title='Total Embarked | Survived')

df.plot(kind='bar',stacked=True, ax=ax[1][1], title='Embarked/Pclass Relation')

name = train.groupby('Name')['Name'].count().sort_values(ascending=False)

cabin = train.groupby('Cabin')['Cabin'].count().sort_values(ascending=False)

ticket = train.groupby('Ticket')['Ticket'].count().sort_values(ascending=False)



print('--Name has {} unique values'.format(len(name)))

print('--Ticket has {} unique values'.format(len(ticket)))

print('--Cabin has total of {} values out of which {} are unique'.format(891-687, len(cabin)))
c = train[~train['Cabin'].isnull()]['Survived'].value_counts()

cn = train[train['Cabin'].isnull()]['Survived'].value_counts()



total_died = len(train.loc[train['Survived'] == 0])



df = pd.DataFrame([c, cn])

df.index = ['With Cabin', 'Missing Cabin']

df.columns = ['Dead', 'Survived']



print('With Cabin % on death rate: {}%'.format(round(df.iloc[0, 0]/total_died * 100, 2)))

print('Missing Cabin % on death rate: {}%'.format(round(df.iloc[1, 0]/total_died * 100, 2)))



df.plot(kind='bar', stacked=True, figsize=(10, 5))
train_test_data = [train, test]

train.head()
for dataset in train_test_data:

    

    dataset.loc[ dataset['Age'] <= 3, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 3) & (dataset['Age'] <= 10), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 10) & (dataset['Age'] <= 17), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 17) & (dataset['Age'] <= 30), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 45), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 45) & (dataset['Age'] <= 52), 'Age'] = 5

    dataset.loc[ dataset['Age'] > 52, 'Age'] = 6
for dataset in train_test_data:

    

    dataset.loc[ dataset['Fare'] <= 7, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7) & (dataset['Fare'] <= 20), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 20) & (dataset['Fare'] <= 27), 'Fare'] = 2

    dataset.loc[(dataset['Fare'] > 27) & (dataset['Fare'] <= 40), 'Fare'] = 3

    dataset.loc[(dataset['Fare'] > 40) & (dataset['Fare'] <= 50), 'Fare'] = 4

    dataset.loc[(dataset['Fare'] > 50) & (dataset['Fare'] <= 65), 'Fare'] = 5

    dataset.loc[(dataset['Fare'] > 65) & (dataset['Fare'] <= 80), 'Fare'] = 6

    dataset.loc[(dataset['Fare'] > 80) & (dataset['Fare'] <= 110), 'Fare'] = 7

    dataset.loc[ dataset['Fare'] > 110, 'Fare'] = 7
train.head()
train["Family"] = train["SibSp"] + train["Parch"]

test["Family"] = test["SibSp"] + test["Parch"]
facet = sns.FacetGrid(train, hue="Survived", aspect=5)

facet.map(sns.kdeplot,'Family',shade= True)

facet.set(xlim=(0, train['Family'].max()))

facet.add_legend()

plt.xticks(range(0, train['Family'].max(), 1))

plt.show()
for dataset in train_test_data:

    dataset.loc[ dataset['Family'] == 0, 'Family'] = 0

    dataset.loc[ dataset['Family'] == 1, 'Family'] = 1

    dataset.loc[(dataset['Family'] > 1) & (dataset['Family'] <= 3), 'Family'] = 2

    dataset.loc[(dataset['Family'] > 3) & (dataset['Family'] <= 6), 'Family'] = 3

    dataset.loc[ dataset['Family'] > 6, 'Family'] = 4
train['Alone'] = 1

train.loc[train['Family'] > 0, 'Alone'] = 0



test['Alone'] = 1

test.loc[train['Family'] > 0, 'Alone'] = 0
for dataset in train_test_data:

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'].value_counts()
bar_chart('Title')
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2,

                 "Master": 3, "Dr": 4, "Rev": 4, "Col": 4, "Major": 4, "Mlle": 4,"Countess": 4,

                 "Ms": 4, "Lady": 4, "Jonkheer": 4, "Don": 4, "Dona" : 4, "Mme": 4,"Capt": 4,"Sir": 4}

for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].map(title_mapping)
facet = sns.FacetGrid(train, hue="Title",aspect=5)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()
train['Age'].fillna(train.groupby('Title')['Age'].transform('median'), inplace=True)

test['Age'].fillna(test.groupby('Title')['Age'].transform('median'), inplace=True)



train['Age'] = train['Age'].astype(int)

test['Age'] = test['Age'].astype(int)
# filling missing values for Embarked based on the most frequent value from our EDA

for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')
train.groupby('Pclass')['Fare'].median().sort_values(ascending=False)
# fill missing Fare with median fare for each Pclass

train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)

test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)



train['Fare'] = train['Fare'].astype(int)

test['Fare'] = test['Fare'].astype(int)
for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].str[:1]



train['Cabin'].value_counts()
df = round((train.groupby('Cabin')['Survived'].sum() / len(train.loc[(train['Survived'] == 1) & (~train['Cabin'].isnull())])) * 100, 2)

df.sort_values(ascending=False)
bar_chart('Cabin')
P1 = train[train['Pclass'] == 1]['Cabin'].value_counts()

P2 = train[train['Pclass'] == 2]['Cabin'].value_counts()

P3 = train[train['Pclass'] == 3]['Cabin'].value_counts()



df = pd.DataFrame([P1, P2, P3])

df.index = ['Pclass 1', 'Pclass 2', 'Pclass 3']

df.plot(kind='bar', stacked=True, figsize=(10, 5))
ES = train[train['Embarked'] == 'S']['Cabin'].value_counts()

EC = train[train['Embarked'] == 'C']['Cabin'].value_counts()

EQ = train[train['Embarked'] == 'Q']['Cabin'].value_counts()



df = pd.DataFrame([ES, EC, EQ])

df.index = ['S', 'C', 'Q']

df.plot(kind='bar', stacked=True, figsize=(10, 5))
facet = sns.FacetGrid(train, hue="Cabin",aspect=5)

facet.map(sns.kdeplot,'Fare',shade= True)

facet.set(xlim=(0, train['Fare'].max()))

facet.add_legend()
cabin_mapping = {"A": 5, "B":2, "C": 1, "D": 3, "E": 4, "F": 5, "G": 5, "T": 5}

for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

    

train.head()
sex_params = {'male': 0, 'female': 1}

for dataset in train_test_data:

    dataset['Sex'] = dataset['Sex'].map(sex_params)
embarked_mapping = {"S": 0, "C": 1, "Q": 2}

for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
# features_drop = ['Name', 'Ticket']

features_drop = ['Name', 'Ticket']

train = train.drop(features_drop, axis=1)

test = test.drop(features_drop, axis=1)

train = train.drop(['PassengerId'], axis=1)



train.head()
test.head()
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier



survived = train['Survived']

cabin_train = train.loc[~train['Cabin'].isna(), train.columns].drop('Survived', axis=1)

cabin_null_train = train.loc[train['Cabin'].isna(), train.columns].drop(['Cabin', 'Survived'], axis=1)



passengerId = test['PassengerId']

cabin_test = test.loc[~test['Cabin'].isna(), test.columns].drop('PassengerId', axis=1)

cabin_null_test = test.loc[test['Cabin'].isna(), test.columns].drop(['Cabin', 'PassengerId'], axis=1)



X = pd.concat([cabin_train, cabin_test], axis=0)



y = X['Cabin']

X.drop(['Cabin'], axis=1, inplace=True)



# Random Forest Classifier

rfc = RandomForestClassifier(n_estimators=500,max_depth=16)

predicted_values = rfc.fit(X, y).predict(X)



f1 = round(f1_score(y, predicted_values, average='micro'), 2)

accuracy = round(accuracy_score(y, predicted_values), 2)

precision = round(precision_score(y, predicted_values, average='micro'), 2)

recall = round(recall_score(y, predicted_values, average='micro'), 2)

print("Accuracy: {} | Precision: {} | Recall: {} | F1: {} ".format(accuracy, precision, recall, f1))



preds = rfc.predict(cabin_null_train)

cabin_null_train['Cabin'] = preds

train = pd.concat([cabin_train, cabin_null_train], axis=0)

train = pd.concat([survived, train], axis=1).reset_index(drop=True)

train.head()
# prepare Test Data

preds = rfc.predict(cabin_null_test)

cabin_null_test['Cabin'] = preds

test = pd.concat([cabin_test, cabin_null_test], axis=0)

test = pd.concat([passengerId, test], axis=1).reset_index(drop=True)

test.head()
# from sklearn.feature_selection import chi2, SelectKBest

# from sklearn.preprocessing import MinMaxScaler

# import numpy as np



# X = train.drop('Survived', axis=1)

# y = train['Survived']

# x_cols = X.columns



# _scaler = MinMaxScaler()

# X = _scaler.fit_transform(X)





# _selector_kbest = SelectKBest(chi2, k='all')

# selected_features_kbest = []



# _selector_kbest.fit(X, y)

# selected_features_kbest.append(list(_selector_kbest.scores_))



# selected_features_kbest = np.mean(selected_features_kbest, axis=0)

# thresh = np.quantile(selected_features_kbest, 0.25)

# selected_features_kbest = [x_cols[i] for i, score in enumerate(selected_features_kbest) if score > thresh]



# print('Total Features Selected: {}/{}'.format(len(selected_features_kbest), len(x_cols)))

# print(selected_features_kbest)
# sur = train['Survived']

# pas = test['PassengerId']



# train = train.loc[:, selected_features_kbest]

# train['Survived'] = sur



# test = test.loc[:, selected_features_kbest]

# test['PassengerId'] = pas
train.head()
test.head()
# def onehotencode(df):

#     pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')

#     cabin = pd.get_dummies(df['Cabin'], prefix='Cabin')

#     title = pd.get_dummies(df['Title'], prefix='Title')

#     embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')

    

#     df2 = pd.concat([pclass, cabin, title, embarked], axis=1)

#     return df2



# train_encodings = onehotencode(train)

# test_encodings = onehotencode(test)



# train = pd.concat([train, train_encodings], axis=1)

# test = pd.concat([test, test_encodings], axis=1)



# features_drop = ['Pclass', 'Cabin', 'Title', 'Embarked']

# train.drop(features_drop, axis=1, inplace=True)

# test.drop(features_drop, axis=1, inplace=True)
train.head()
test.head()
train.to_csv('./train_cleaned.csv', index=False)

test.to_csv('./test_cleaned.csv', index=False)