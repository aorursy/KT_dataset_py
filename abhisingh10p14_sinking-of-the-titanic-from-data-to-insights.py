import numpy as np

import pandas as pd

from sklearn.model_selection import cross_val_score

from sklearn import tree

from sklearn.linear_model import LinearRegression

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import graphviz



sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})

sns.set_palette(sns.color_palette("husl", 10))

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.info()
test.info()
titanic = pd.concat([train, test], axis=0).reset_index()

titanic.info()
f, (g1, g2) = plt.subplots(1, 2, figsize=(15,5))

sns.barplot(x="Sex", y="Survived", hue="Pclass", data=train, ax=g1).set_title(

    "Average Survival per Gender and Social Class");

sns.pointplot(x="Sex", y="Survived", hue="Pclass", data=train, ax=g2).set_title(

    "Average Survival per Gender and Social Class");
f, (g1, g2, g3) = plt.subplots(1, 3, figsize=(15,5))

sns.distplot(titanic.Fare.dropna(), ax=g1, hist=False, color='g').set_title(

    "How Fare is Distributed");

sns.boxplot(x='Fare', y='Pclass', data=titanic, orient='h', ax=g2).set_title(

    "Boxplot of Fare per Social Class");

sns.boxplot(x='Fare', y='Survived', data=titanic, orient='h', ax=g3).set_title(

    "Boxplot of Fare per Survival Status");

titanic.groupby('Pclass')['Fare'].describe()
titanic.query("Fare > 300")
f, (g1, g2) = plt.subplots(1, 2, figsize=(15,5))

sns.pointplot(x='Pclass', y='Survived', hue='Sex', data=train, ax=g1).set_title(

    "Average Survival per Social Class and Gender");

sns.countplot(x='Pclass', hue='Survived', data=train, ax=g2).set_title(

    "Nr of Survivals/Deaths per Social Class");
titanic[['Age']].describe().T
f, (g1, g2, g3) = plt.subplots(1, 3, figsize=(15, 5))

sns.distplot(

    titanic['Age'].dropna(), bins=int(titanic['Age'].max()), 

    color='g', ax=g1).set_title("Age Distribution");



sns.distplot(

    titanic.query("Sex == 'female'")['Age'].dropna(), bins=int(titanic['Age'].max()),

    color='red', hist=False, ax=g2, label='Women').set_title(

    "Age Distribution for Women and for Men");

sns.distplot(

    titanic.query("Sex == 'male'")['Age'].dropna(), bins=int(titanic['Age'].max()),

    color='b', hist=False, ax=g2, label='Men');



sns.distplot(

    titanic.query("Survived == 0")['Age'].dropna(), bins=int(titanic['Age'].max()),

    hist=False, label="Died", color='black', ax=g3).set_title(

    "Age Distribution per Survival Status");

sns.distplot(

    titanic.query("Survived == 1")['Age'].dropna(), bins=int(titanic['Age'].max()),

    hist=False, label="Survived", color='g', ax=g3);
f, (g1, g2, g3) = plt.subplots(1, 3, figsize=(15,5))

sns.boxplot(titanic['Age'], orient='h', ax=g1).set_title(

    "Boxplot of Age");

sns.boxplot(x='Age', y='Embarked', data=titanic, orient='h', ax=g2).set_title(

    "Boxplot of Age per Port");

sns.boxplot(x='Age', y='Pclass', data=titanic, orient='h', ax=g3).set_title(

    "Boxplot of Age per Social Class");
cabin_nnull = titanic[~titanic.Cabin.isnull()].copy()

cabin_nnull['deck'] = cabin_nnull['Cabin'].str[0]

f, (g1, g2) = plt.subplots(1, 2, figsize=(15, 5))

sns.barplot(

    x='deck', y='Survived', data=cabin_nnull, order=sorted(cabin_nnull.deck.unique()),

    ax=g1).set_title("Average Survival per Deck");

sns.countplot(

    x="deck", hue="Pclass", data=cabin_nnull, palette="Greens_d", 

    order=sorted(cabin_nnull.deck.unique()), ax=g2).set_title(

    "Nr of Passengers per Deck and Social Class");
counts= train['Survived'].value_counts()

print("Overall Survival Rate:\t\t {}".format((counts[1] / counts.sum())))

print("Cabin-Not-Null Survival Rate:\t {}".format(cabin_nnull.Survived.sum() / cabin_nnull.shape[0]))
titanic.groupby('Embarked')['Fare'].describe()
titanic[titanic.Embarked.isnull()]
f, ((g1, g2), (g3, g4)) = plt.subplots(2, 2, figsize=(12,12))

sns.barplot(x='Embarked', y='Survived', hue='Pclass', data=titanic, ax=g1).set_title(

    "Average Survival per Port and Social Class");

sns.countplot(x='Embarked', hue='Survived', data=titanic, ax=g2).set_title(

    "Nr of Passengers per Port and Survival Status");

sns.barplot(x='Embarked', y='Fare', hue='Survived', data=titanic, ax=g3).set_title(

    "Average Fare per Port and Survival Status");

sns.barplot(x='Embarked', y='Fare', hue='Pclass', data=titanic, ax=g4).set_title(

    "Average Fare per Port and Social Class");
titanic.Name.head()
titles = titanic.Name.apply(lambda s: s.split(',')[1].split('.')[0])

titles.value_counts()
df = titanic[['Survived']].copy()

df['family_size'] = titanic['Parch'] + titanic['SibSp'] + 1

f, (g1, g2) = plt.subplots(1, 2, figsize=(15, 5))

sns.barplot(x="family_size", y="Survived", data=df, ax=g1).set_title(

    "Average Survival per Family Size");

sns.countplot(x="family_size", data=df, ax=g2, palette='Greens_d').set_title(

    "Nr of Passengers For Each Group In Family Size");
print(titanic.Ticket.count())

print(titanic.Ticket.nunique())

titanic.Ticket.unique()[1:50]
feats = titanic.copy()

feats[feats.Embarked.isnull()]
feats['Embarked'].fillna('C', inplace=True)

feats['Embarked'].isnull().sum()
feats[feats.Fare.isnull()]
desc = feats.query("Embarked == 'S' and Pclass == 3")['Fare'].describe()

feats['Fare'].fillna(desc['mean'], inplace=True)

feats['Fare'].isnull().sum()
predictors = ['Fare', 'Parch', 'Pclass', 'SibSp', 'Age']

age_train = feats.loc[~feats.Age.isnull(), predictors]

age_test = feats.loc[feats.Age.isnull(), predictors]

lm = LinearRegression()

lm.fit(age_train.drop('Age', axis=1), age_train['Age'], )

predicted_age = lm.predict(age_test.drop('Age', axis=1))

feats.loc[feats.Age.isnull(), 'Age'] = predicted_age

feats.info()
sns.distplot(titanic.Age.dropna(), hist=False, color='r', label='Before Imputation');

sns.distplot(feats.Age, hist=False, color='g', label='After Imputation').set_title(

    "KDEs of Age Distributed Before and After Imputation");
sex = pd.get_dummies(feats['Sex'])

sex.head()
dic = {'C': 'cherbourg', 'Q': 'queenstown', 'S': 'southampton'}

embarked = pd.get_dummies(feats['Embarked'])

embarked.columns = [dic[i] for i in embarked.columns if i in dic.keys()]

embarked.head()
bins = [-1, 2, 6, 12, 16, 40, 60, 100]

group_names = ['baby', 'small_child', 'child', 'teenager', 'young_adult', 'adult', 'senior']

categories = pd.cut(feats['Age'], bins, labels=group_names)

feats['age_bins'] = categories
f, (g1, g2) = plt.subplots(1, 2, figsize=(15, 5))

sns.countplot(

    x='age_bins', hue='Survived', data=feats, order=group_names, 

    palette="Greens_d", ax=g1).set_title(

    "Nr of Survivals and Deaths per Age Group");

sns.barplot(

    x='age_bins', y='Survived', hue='Sex', data=feats, order=group_names, 

    palette="Greens_d", ax=g2).set_title(

    "Average Survival per Age Group and Gender");

g2.legend(loc='upper left');
age = pd.get_dummies(feats['age_bins'])

age.head()
cabin_not_null = pd.Series([0] * feats.shape[0]).rename('cabin_not_null')

cabin_not_null[~feats.Cabin.isnull()] = 1

cabin_not_null.head()
surname = pd.DataFrame(

    feats.Name.apply(lambda s: s.split(',')[0].split('.')[0]).rename('surname'))

feats['surname'] = surname

surname.head()
feats['title'] = pd.DataFrame(

    feats.Name.apply(lambda s: s.split(',')[1].split('.')[0].strip(' ')).rename('title'))

changes = {

    'Mlle': 'Miss', 'Mme': 'Mrs', 'Ms': 'Miss',

    'Sir': 'Noble', 'Lady': 'Noble', 'the Countess': 'Noble', 'Jonkheer': 'Noble', 'Don': 'Noble', 'Dona': 'Noble',

    'Major': 'Militar', 'Capt': 'Militar', 'Col': 'Militar'}

feats['title'].replace(changes, inplace=True)

print(feats.title.value_counts())



title = pd.get_dummies(feats['title'])

title.columns = [i.lower() for i in title.columns]

title.head()
f, (g1, g2) = plt.subplots(1, 2, figsize=(15, 5))

sns.countplot(x='title', hue='Survived', data=feats, palette="Greens_d", ax=g1).set_title(

    'Nr of Survivals and Deaths per Title');

sns.barplot(x='title', y='Survived', data=feats, palette="Greens_d", ax=g2).set_title(

    "Average Survival per Title ");
feats['family_size'] = feats['Parch'] + feats['SibSp'] + 1

bins = [0, 1, 4, 11]

feats['family_type'] = pd.cut(

    feats['family_size'], bins, labels=['single', 'small_family', 'large_family'])

feats['family_size'].head()
f, (g1, g2) = plt.subplots(1, 2, figsize=(15, 5))

sns.barplot(x='family_type', y='Survived', data=feats, ax=g1).set_title(

    "Average Survival per Family Type");

sns.countplot(x='family_type', hue='Survived', data=feats, palette="Greens_d", ax=g2).set_title(

    'Nr of Survivals and Deaths per Family Type');
family_type = pd.get_dummies(feats['family_type'])

family_type.head()
tickets = feats.query("family_size > 1")['Ticket'].copy(

    ).str.replace('.', '').rename('ticket').to_frame()

split = tickets.ticket.str.split(' ')

tickets['ticket_nr'] = split.apply(lambda s: s.pop())

def get_element(s):

    '''Get the element of a list.'''

    try:

        return s[0]

    except Exception as e: 

        return None

tickets['ticket_prefix'] = split.apply(lambda s: get_element(s))

tickets[['ticket', 'ticket_prefix', 'ticket_nr']].head()
pars = []

for t in tickets.ticket_nr.unique():

    dat = feats.iloc[tickets[tickets.ticket_nr == t].index.tolist()] 

    if dat.shape[0] == 1 or not any(dat.Age <= 6):

        continue  # skips if there is only 1 passenger per ticket number or if there is no baby

    family = pd.concat([dat.query("Age <= 6"), dat.query("Parch > 0 and Age > 15")])

    pars.append(family)

parents = pd.concat(pars).query("Parch > 0 and Age > 15")

parents['parents'] = 1

parents[['surname', 'Age', 'Parch', 'Pclass', 'Sex', 

         'SibSp', 'Survived', 'age_bins', 'family_size']].head()

feats = feats.join(parents['parents'])

feats['parents'].fillna(0, inplace=True)
feats['parents'].sum()
tickets.ticket_prefix.value_counts().rename('nr_tickets').to_frame()
dfs = [feats.iloc[:train.shape[0]], sex[:train.shape[0]],

       embarked[:train.shape[0]], cabin_not_null[:train.shape[0]],

       age[:train.shape[0]], title[:train.shape[0]],

       family_type[:train.shape[0]]

      ]

new_feats = pd.concat(dfs, axis=1, ignore_index=False).drop('index', axis=1)

new_feats.columns = map(str.lower, new_feats.columns.tolist())
new_feats.to_csv("titanic_feats.csv", index=False)  # save feats to file
new_feats.columns.unique()
predictors = ['fare', 'pclass', 'male', 'cabin_not_null', 

              'master', 'small_family', 'age',

             ]

x = new_feats[predictors]

y = new_feats['survived']

x.head()
dt = tree.DecisionTreeClassifier(max_depth=5, min_samples_split=5)

dt = dt.fit(x, y)
pd.Series(dict(zip(x.columns, dt.feature_importances_))).rename(

    'feat_importance').sort_values(ascending=True).plot(

    kind='barh', title='Feature Importance Ranking', color='g');
scores = cross_val_score(dt, x, y, cv=5, scoring='accuracy', n_jobs=-1)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
dot_data = tree.export_graphviz(

    dt, out_file=None, feature_names=predictors,

    filled=True, rounded=True, special_characters=True)

graphviz.Source(dot_data)