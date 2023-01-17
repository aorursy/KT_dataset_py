# import the usual libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train_df = pd.read_csv('../input/train.csv', index_col='PassengerId')
test_df = pd.read_csv('../input/test.csv', index_col='PassengerId')
df = pd.concat([train_df, test_df])

df.sample(10)
# Have a look at other numerical features
df.describe()
# Basic statistics for non-numerical cases
df.describe(include=['O'])
df.isnull().sum()
is_estimated_or_null = lambda x: pd.isnull(x) or (x>1 and divmod(x, 1)[1] == 0.5)
df['estimated_age'] = df.Age.apply(lambda age: 1 if is_estimated_or_null(age) else 0)
# Let's verify the guess grouping
age_grouped = df[['Pclass','Sex','Embarked','Age']].groupby(['Pclass','Sex','Embarked']).median()
age_grouped
real_age = lambda row: row.Age if not pd.isnull(row.Age) else age_grouped.loc[row.Pclass].loc[row.Sex].loc[row.Embarked].Age
df['Age'] = df[['Pclass','Sex','Embarked','Age']].apply(real_age, axis=1)
df['cabin_letter'] = df.Cabin.apply(lambda c: c[0] if not pd.isnull(c) else 'N') # N=none

df.sample(5)
# Grouping by cabin letter should show us some insights...
survival_ratio = df[['cabin_letter','Pclass','Survived']].groupby(['cabin_letter']).mean()
people_count = df[['cabin_letter','Name']].groupby(['cabin_letter']).count().rename(columns={'Name': 'passenger_count'})

pd.concat([survival_ratio,people_count], axis=1)
df['surname'] = df.Name.apply(lambda n: n.split(',')[0])
df.sample(10)
#Group by surname and class, in order to find people that could be a family
surnames = df[['surname','Cabin','Pclass','Name']].groupby(['surname','Pclass']).count()
surnames.head()
# Find cases with more people than assigned cabin
missing = surnames[(surnames.Cabin>0) & (surnames.Cabin<surnames.Name)] # Notice the element-wise binary logical operator '&'
missing.rename(columns={'Name': 'passenger_count'})
df[df.surname=='Brown']
df[df.surname=='Hoyt']
tickets_grouped = df[['Ticket','Cabin','Name']].groupby('Ticket').count()

# Filter: With at least a Cabin, with at least 2 people, and more people than cabins
candidate_tickets = tickets_grouped[(tickets_grouped['Cabin']>=1) & (tickets_grouped['Name']>=2) & (tickets_grouped['Cabin']<tickets_grouped['Name'])]
candidate_tickets
df[df.Ticket=='113781']
shared_tickets = candidate_tickets.index.tolist()

find_cabin_given_ticket = lambda ticket: df[(df.Ticket==ticket) & (pd.notnull(df.Cabin))].Cabin.values[0]
def assign_cabin(row):
    if pd.isnull(row.Cabin) and row.Ticket in shared_tickets: 
        return find_cabin_given_ticket(row.Ticket) 
    return row.Cabin

df['Cabin'] = df[['Cabin', 'Ticket']].apply(assign_cabin, axis=1)
df['cabin_letter'] = df['Cabin'].apply(lambda c: c[0] if not pd.isnull(c) else 'N') # N=none

df[df.Ticket=='113781']
df.Cabin.isnull().sum()
df[['Embarked', 'Survived', 'Name', 'Pclass']].groupby('Embarked').agg(
    {'Name': ['count'], 'Pclass': ['mean'], 'Survived': ['mean']})
df['Embarked'].fillna('S', inplace=True)
df[pd.isnull(df.Fare)]
estimated_fare = df[(df.Embarked=='S') & (df.Pclass==3) & (df.Sex=='male')].Fare.mean()
df['Fare'].fillna(estimated_fare, inplace=True)
grouped_ages = df[['Age','Survived']].groupby(by=lambda index: int(df.loc[index]['Age']/10)).mean()
grouped_ages.plot(x='Age', y='Survived')
#Why the line goes up in 80 years? An outlier?
df[df['Age']>=80]
df = df[df['Age']<80]
df['decade'] = df['Age'].apply(lambda age: int(age/10))

# We will save useful features (column names) for later.
useful = ['Age', 'decade']
# Grouping by 100s
fare_grouped = df[['Fare', 'Survived']].groupby(by=lambda i: int(df.loc[i]['Fare']/100)).mean()
fare_grouped.plot(x='Fare', y='Survived')
useful.append('Fare')
sibblings_grouped = df[['SibSp', 'Survived']].groupby('SibSp').mean()
sibblings_grouped.plot()
generations_grouped = df[['Parch', 'Survived']].groupby('Parch').mean()
generations_grouped.plot()
df['family_size'] = df['SibSp'] + df['Parch'] + 1

# Let's see if there is a clear limit
df[['family_size', 'Survived']].groupby('family_size').mean().plot()
df['small_family'] = df['family_size'].apply(lambda size: 1 if size<=4 else 0)
df['big_family'] = df['family_size'].apply(lambda size: 1 if size>=7 else 0)
df['no_family'] = df['family_size'].apply(lambda s: 1 if s==1 else 0)

useful.extend(['SibSp', 'Parch', 'family_size', 'small_family', 'big_family', 'no_family'])
survived_sex = df[df['Survived']==1]['Sex'].value_counts()
survived_sex.name='Survived'
dead_sex = df[df['Survived']==0]['Sex'].value_counts()
dead_sex.name='Dead'

table = pd.DataFrame([survived_sex,dead_sex])

table.T.plot(kind='bar', stacked=True, color='gr')
df['male'] = df['Sex'].map({'male': 1, 'female': 0})

useful.append('male')
embarked_grouped = df[['Embarked', 'Pclass', 'Survived']].groupby(['Embarked','Pclass']).mean()
embarked_grouped.plot(kind='barh')
df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='embarked')], axis=1)

useful.extend(['embarked_{}'.format(x) for x in ['C', 'S', 'Q']])

#Let's see how these multiple columns look like
df.sample(5)
useful.append('Pclass')
df = pd.concat([df, pd.get_dummies(df['cabin_letter'], prefix='deck')], axis=1)

letters = df['cabin_letter'].unique()
useful.extend(['deck_{}'.format(x) for x in letters])
ticket_count = df[['Ticket', 'Name']].groupby('Ticket').count().rename(columns={'Name':'count'}).sort_values(by='count', ascending=False)
ticket_count.head()
df[df['Ticket']=='CA. 2343']
df[df['Ticket']=='1601']
df['ticket_owners'] = df['Ticket'].apply(lambda x: ticket_count.loc[x])
df['shared_fare'] = df['Fare'] / df['ticket_owners']

df['alone'] = df[['ticket_owners','no_family']].apply(lambda row: 1 if row.ticket_owners==1 and row.no_family==1 else 0 , axis=1)

useful.extend(['ticket_owners', 'shared_fare', 'alone'])
df['ticket_owners'].describe()
older_age = df[['Ticket', 'Age']].groupby('Ticket').max()
df['older_relative_age'] = df['Ticket'].apply(lambda ticket: older_age.loc[ticket])

useful.extend(['older_relative_age'])
import re


def ticket_type(t):
    if re.match('^\d+$', t):
        return 'len' + str(len(t))
    else:
        return re.sub('[^A-Z]', '', t)


df['ticket_type'] = df['Ticket'].apply(ticket_type)

df[['ticket_type', 'Survived']].groupby(
    'ticket_type').agg({'Survived': ['mean', 'std','count']}).sort_values(('Survived','count'), ascending=False)
def useful_ticket_type(ticket_type):
    useful_types = ['A', 'SOTONOQ', 'WC']
    if ticket_type in useful_types:
        return ticket_type
    else:
        return 'other'


df['useful_ticket_type'] = df['ticket_type'].apply(useful_ticket_type)

df = pd.concat(
    [df, pd.get_dummies(df['useful_ticket_type'], prefix='ticket_type')], axis=1)

letters = df['useful_ticket_type'].unique()
useful.extend(['ticket_type_{}'.format(x) for x in letters])

df.sample(10)
df['name_length'] = df.Name.apply(len)
df[['name_length', 'Survived']].groupby('name_length').mean().plot()
df['name_length_short'] = df['name_length'].apply(lambda s: 1 if s <= 35 else 0)
df['name_length_mid'] = df['name_length'].apply(lambda s: 1 if 35 < s <=58 else 0)
df['name_length_long'] = df['name_length'].apply(lambda s: 1 if s > 58 else 0)

useful.extend(['name_length', 'name_length_short', 'name_length_mid', 'name_length_long'])
import langid

df['lang'] = df['Name'].apply(lambda n: langid.classify(n)[0])
df[['Name','lang']].sample(10)
lang_count = df[['lang','Name']].groupby('lang').count().rename(columns={'Name':'count'})
lang_class = df[['lang','Pclass']].groupby('lang').mean()
lang_survived = df[['lang','Survived']].groupby('lang').mean()
pd.concat([lang_count, lang_class, lang_survived], axis=1).sort_values(by='count', ascending=False).head(15)
language_groups = {
    'uk': ('cy', 'en'),
    'germanic': ('da', 'de', 'nl'),
    'latin': ('es', 'fr', 'it', 'la', 'pt', 'br', 'ro'),
    'african': ('af', 'rw', 'xh'),
    'asian': ('id', 'tl', 'tr')
}
language_map = { y:x for x in language_groups for y in language_groups[x]}    

df['lang_group'] = df['lang'].apply(lambda l: language_map[l] if l in language_map else 'other')
survived_avg_per_group = df[['lang_group','Survived']].groupby('lang_group').mean()
survived_std_per_group = df[['lang_group','Survived']].groupby('lang_group').std().rename(columns={'Survived':'std'})
pd.concat([survived_avg_per_group, survived_std_per_group], axis=1)
df = pd.concat([df, pd.get_dummies(df['lang_group'], prefix='lang_group')], axis=1)

langs = df['lang_group'].unique()
useful.extend(['lang_group_{}'.format(x) for x in langs])
surnames = df[['surname', 'Name']].groupby('surname').count().rename(columns={'Name':'count'})
df['surname_count'] = df['surname'].apply(lambda x: surnames.loc[x])

useful.append('surname_count')
df['title'] = df['Name'].apply(lambda n: n.split(',')[1].split('.')[0].strip())

df[['title', 'Survived']].groupby('title').agg({'Survived': [
    'mean', 'std', 'count']}).sort_values(('Survived', 'count'), ascending=False)
title_groups = {
    "Capt": "sacrifies",
    "Col": "army",
    "Rev": "sacrifies",
    "Major": "army",
    "Mr" : "Mr",
    "Master": "Master",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Mme": "Mrs",
    "Ms": "Mrs",
    "Mlle": "Miss"
}

df['title_group'] = df['title'].apply(lambda t: title_groups[t] if t in title_groups else 'other')

df = pd.concat([df, pd.get_dummies(df['title_group'], prefix='title_group')], axis=1)

t_g = df['title_group'].unique()
useful.extend(['title_group_{}'.format(x) for x in t_g])
df.corr()['Survived'].sort_values()
train = df[df['Survived'].notnull()]
train_X = train[useful]
train_y = train['Survived']
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=1)
clf = clf.fit(train_X, train_y)
importances = pd.DataFrame(clf.feature_importances_, index=train_X.columns, columns=['importance']).sort_values(by='importance')
importances.plot.barh(figsize=(16,8), legend=None, title='Feature importance')
top_features = importances.tail(8).index.tolist()
top_features.append('Survived')
top_correlations = df[top_features].corr()

sns.heatmap(top_correlations, annot=True)
train_X.shape
useful = importances.tail(33).index.tolist()
useful.remove('title_group_Mr')
useful.remove('Fare')
useful.remove('older_relative_age')
train_X=train[useful]
train_X.shape
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

# Set to True or False to search all combinations or use previous results
search_best_hyperparameters = False

if search_best_hyperparameters:
    parameter_grid = {
        'n_estimators': [10, 20, 50, 100, 200, 500],
        'learning_rate': [0.1, 0.2, 0.5, 1, 1.2],
        'random_state': [1]
    }
    model = AdaBoostClassifier()
    gs = GridSearchCV(
        model,
        scoring='accuracy',
        param_grid=parameter_grid,
        cv=4,
        n_jobs=-1)
    gs.fit(train_X, train_y)
    params = gs.best_params_
    print(params)
else:
    params = {
        'learning_rate': 0.1,
        'n_estimators': 500,
        'random_state': 1
    }
# Use the params to get a score with the training set
clf = AdaBoostClassifier(**params)
clf = clf.fit(train_X, train_y)
clf.score(train_X, train_y)
test = df[df['Survived'].isnull()]
test_X = test[useful]
test_y = clf.predict(test_X)
submit = pd.DataFrame(test_y.astype(int), index=test_X.index, columns=['Survived'])
submit.head()
submit.to_csv('submission.csv')