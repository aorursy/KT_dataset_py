import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from collections import Counter

import string

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression





pd.options.mode.chained_assignment = None





def get_title(name):

    name = name.split(',')[1]

    name = name.split('.')[0]

    return name.strip()





def get_title_grouped(name):

    title = get_title(name)

    if title in ['Rev', 'Dr', 'Col', 'Major', 'the Countess', 'Sir', 'Lady', 'Jonkheer', 'Capt', 'Dona', 'Don']:

        title = 'Rare'

    elif title in ['Ms', 'Mlle']:

        title = 'Miss'

    elif title == 'Mme':

        title = 'Mrs'

    return title





def get_deck(cabin):

    if isinstance(cabin, str):

        if cabin[0] == 'T':

            return np.nan

        return cabin[0]

    return cabin





train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

full = pd.concat([train, test])



# feature engineering described in previous notebooks

full['Embarked'].fillna('C', inplace=True)

full['Fare'].fillna(8.05, inplace=True)

full['Title'] = full['Name'].apply(get_title_grouped)

full['Deck'] = full['Cabin'].apply(get_deck)

full['Family size'] = full['Parch'] + full['SibSp']
ticket_nums = [int(n.split()[-1]) for n in full['Ticket'].values if n.split()[-1].isdigit()]

plt.hist(ticket_nums, 50)

plt.xlabel('Ticket number')

plt.ylabel('Count')

plt.show()
ticket_nums = [num for num in ticket_nums if num < 2000000]

plt.hist(ticket_nums, 50)

plt.xlabel('Ticket number')

plt.ylabel('Count')

plt.show()
def get_ticket_num(ticket):

    ticket_num = ticket.split()

    ticket_num = ''.join(char for char in ticket_num[-1].strip() if char not in string.punctuation)

    if not ticket_num.isdigit():

        return np.nan

    return int(ticket_num)



full['Ticket number'] = full['Ticket'].apply(get_ticket_num)

full['Ticket number'].fillna(np.nanmedian(full['Ticket number'].values), inplace=True)



full.drop(['Name', 'Ticket', 'Cabin', 'Parch', 'SibSp'], axis=1, inplace=True)
encoders = {}

to_encode = ['Embarked', 'Sex', 'Title']

for col in to_encode:

    encoders[col] = LabelEncoder()

    encoders[col].fit(full[col])

    full[col] = encoders[col].transform(full[col])



age_train = full[full['Age'].notnull()]

age_predict = full[~full['Age'].notnull()]

lr = LinearRegression()

lr.fit(age_train.drop(['Deck', 'Survived', 'PassengerId', 'Age'], axis=1), age_train['Age'])

predicted_ages = lr.predict(age_predict.drop(['Deck', 'Survived', 'PassengerId', 'Age'], axis=1))

age_predict['Age'] = [max(0., age) for age in predicted_ages]



full = pd.concat([age_train, age_predict]).sort_values('PassengerId')
ages = age_train.Age

ages.plot.kde(label='Original')

ages = full.Age

ages.plot.kde(label='With predicted missing values')

plt.xlabel('Age')

plt.legend(prop={'size': 9})

plt.show()
Counter(full['Deck'].values)
full_with_deck = full[full['Deck'].notnull()]

full_without_deck = full[~full['Deck'].notnull()]



full_with_deck_means, full_without_deck_means = [], []

for col in full_with_deck:

    if col not in ['Deck', 'PassengerId']:

        sum_means = np.nanmean(full_with_deck[col].values) + np.nanmean(full_without_deck[col].values)

        full_with_deck_means.append(np.nanmean(full_with_deck[col].values)/sum_means)

        full_without_deck_means.append(np.nanmean(full_without_deck[col].values)/sum_means)



bar_width = 0.35

opacity = 0.4

x_index = np.arange(len(full_with_deck_means))



plt.bar(x_index, full_with_deck_means, bar_width, alpha=opacity, color='b', label='With deck value')

plt.bar(x_index + bar_width, full_without_deck_means, bar_width, alpha=opacity, color='r', label='Missing deck value')

plt.legend(loc='upper center', prop={'size': 9})

plt.ylabel('Ratio of means')

plt.xticks(x_index + bar_width, [col for col in full_with_deck if col not in ['PassengerId', 'Deck']], rotation='vertical')

plt.show()
full['Deck'].fillna('N', inplace=True)



encoders['Deck'] = LabelEncoder()

encoders['Deck'].fit(full['Deck'])

full['Deck'] = encoders['Deck'].transform(full['Deck'])
train = full[full.PassengerId < 892]

test = full[full.PassengerId >= 892]



rf = RandomForestClassifier(n_estimators=100, oob_score=True)

rf.fit(train.drop(['Survived', 'PassengerId'], axis=1), train['Survived'])



rf.score(train.drop(['Survived', 'PassengerId'], axis=1), train['Survived'])
rf.oob_score_
features = list(zip(train.drop(['Survived', 'PassengerId'], axis=1).columns.values, rf.feature_importances_))

features.sort(key=lambda f: f[1])

names = [f[0] for f in features]

lengths = [f[1] for f in features]



pos = np.arange(len(features)) + .5

plt.barh(pos, lengths, align='center', color='r', alpha=opacity)

plt.yticks(pos, names)

plt.xlabel('Gini importance')

plt.show()