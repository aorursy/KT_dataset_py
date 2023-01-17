import re

import math

import pandas as pd

import numpy as np

import seaborn as sb

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectKBest

from sklearn.cross_validation import StratifiedKFold

from sklearn.model_selection import GridSearchCV
X_train = pd.read_csv('../input/train.csv')

X_test = pd.read_csv('../input/test.csv')

X_all = pd.concat([X_train, X_test])
def get_honorific(str):

    h = re.search('[A-Za-z]+\.', str).group(0)

    return h if h else 'missed'



def deck_parser(cabin):

    not_found = 'missed'

    if isinstance(cabin, (str)):

        d = re.search('^[A-Z]', cabin).group(0)

        return d if d else not_found

    else:

        return not_found



class DataDigest:

    def __init__(self):

        self.honorific = None



digest = DataDigest()

digest.honorific = X_test['Name'].apply(get_honorific).unique()

digest.age = X_all.groupby('Sex')['Age'].median()

digest.fare = X_all.groupby('Pclass')['Fare'].median()

digest.embarked = X_test['Embarked'].unique()

digest.deck = X_test['Cabin'].apply(deck_parser).unique()

digest.cabin = X_test['Cabin'].fillna('missed').unique()

digest.ticket = X_test['Ticket'].fillna('missed').unique()
def munge_data(df, digest):

    age_mapper = lambda i: digest.age[i['Sex']] if pd.isnull(i['Age']) else i['Age']

    df['Age'] = df.apply(age_mapper, axis=1)

    fare_mapper = lambda i: digest.fare[i['Pclass']] if pd.isnull(i['Fare']) else i['Fare']

    df['Fare'] = df.apply(fare_mapper, axis=1)

    

    sex_d = pd.get_dummies(df['Sex'], prefix='SexD')

    df = pd.concat([df, sex_d], axis=1)

    df['Sex'] = df['Sex'].apply(lambda s: 1 if s == 'male' else 0)

    

    def honorific_mapper(name):

        honorifics = {'Mr.': 1, 'Mrs.': 2, 'Miss.': 3, 'Master.': 4,

                      'Ms.': 5,'Col.': 6, 'Rev.': 7, 'Dr.': 8, 'Dona.': 9}

        honorific = get_honorific(name)

        return honorifics.get(honorific, 0)

    

    df['Honorific'] = df['Name'].apply(honorific_mapper)

    

    df.drop('Name', axis=1, inplace=True)

    

    def embarked_mapper(key):

        embarked = {'Q': 1, 'S': 2, 'C': 3}

        return embarked.get(key, 0)

    

    df['Embarked'] = df['Embarked'].apply(embarked_mapper)

    

    def deck_mapper(deck):

        deck = deck_parser(deck)

        decks = {'B': 1, 'E': 2, 'A': 3, 'C': 4, 'D': 5, 'F': 6, 'G': 7}

        return decks.get(deck, 0)

    

    df['Deck'] = df['Cabin'].apply(deck_mapper)

    

    def cabin_mapper(cabin):

        index = np.where(digest.cabin==cabin)[0]

        return index[0] if len(index) > 0 else -1

    

    df['Cabin'] = df['Cabin'].apply(cabin_mapper)

    

    embarked_d = pd.get_dummies(df['Embarked'], prefix='EmbarkedD')

    df = pd.concat([df, embarked_d], axis=1)

    

    honorific_d = pd.get_dummies(df['Honorific'], prefix='HonorificD')

    df = pd.concat([df, honorific_d], axis=1)

    

    deck_d = pd.get_dummies(df['Deck'], prefix='DeckD')

    df = pd.concat([df, deck_d], axis=1)

    

    df.drop(['PassengerId'], axis=1, inplace=True)

    

    def ticket_mapper(ticket):

        index = np.where(digest.ticket==ticket)[0]

        return index[0] if len(index) > 0 else -1

    

    df['Ticket'] = df['Ticket'].apply(ticket_mapper)

    

    return df



X_train_munged = munge_data(X_train, digest)

X_train_munged['HonorificD_9'] = pd.Series(np.zeros(891))

X_test_munged = munge_data(X_test, digest)

X_test_munged['EmbarkedD_0'] = pd.Series(np.zeros(891))

X_test_munged['HonorificD_0'] = pd.Series(np.zeros(891))

X_all_munged = pd.concat([X_train_munged.drop('Survived', axis=1), X_test_munged])



X_train_munged.info()
features = [

    'Pclass',

    'Sex',

    #'SexD_female', 'SexD_male',

    'Age',

    'SibSp',

    'Parch',

    'Fare',

    'Cabin',

    'Deck',

    #'Ticket',

    'Honorific',

    'HonorificD_0', 'HonorificD_1', 'HonorificD_2', 'HonorificD_3', 'HonorificD_4', 'HonorificD_5',

    'HonorificD_6', 'HonorificD_7', 'HonorificD_8',

    'Embarked',

    'EmbarkedD_0' , 'EmbarkedD_1', 'EmbarkedD_2', 'EmbarkedD_3',

    'DeckD_0', 'DeckD_1', 'DeckD_2', 'DeckD_3', 'DeckD_4', 'DeckD_5', 'DeckD_6', 'DeckD_7'

]





#scaler = StandardScaler()

#scaler.fit(X_all_munged)



#X_train_scaled = scaler.transform(X_train_munged[features])

#X_test_scaled = scaler.transform(X_test_munged[features])
sb.pairplot(X_train_munged, vars=["Age", "Pclass", "Sex"], hue="Survived", dropna=True)

sb.plt.show()
feature_selector = SelectKBest()

feature_selector.fit(X_train_munged[features], X_train_munged['Survived'])

scores = -np.log10(feature_selector.pvalues_)

plt.bar(range(len(features)), scores)

plt.xticks(range(len(features)), features, rotation='vertical')

plt.show()
cv = StratifiedKFold(X_train['Survived'], random_state=1)



model = RandomForestClassifier(random_state=1, n_jobs=-1)

params = [{

    'n_estimators': [300],

    'min_samples_split': [8],

    'min_samples_leaf': [2]

}]



model_selector = GridSearchCV(model, params, cv=cv, n_jobs=-1)

model_selector.fit(X_train_munged[features], X_train['Survived'])
best_model = model_selector.best_estimator_

print('Accuracy (random forest auto): {} with params {}'

       .format(model_selector.best_score_, model_selector.best_params_))
best_model.fit(X_train_munged[features], X_train['Survived'])

predictions = best_model.predict(X_test_munged[features])
submission = pd.DataFrame({'PassengerId': X_test['PassengerId'], 'Survived': predictions})

#submission.to_csv('submission.csv', index=False)