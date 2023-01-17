# Importing Libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import random



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

from mlxtend.classifier import StackingCVClassifier
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

train_data.head()
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_data.head()
plt.figure(figsize=(10,6))

ax = sns.countplot(x=train_data['Survived'])
plt.figure(figsize=(10,6))

ax = sns.pointplot(x="Pclass",

                   y="Survived",

                   hue="Sex",

                   data=train_data,

                   palette={"male":"g","female":"m"},

                   markers=["^","o"],

                   linestyles=["-","--"])
plt.figure(figsize=(10,6))

ax = sns.countplot(x= 'Sex',data = train_data, hue='Survived')
plt.figure(figsize=(10,6))

ax = sns.boxplot(x='Age', data=train_data, hue='Survived')
plt.figure(figsize=(10,6))

ax = sns.countplot(x='Pclass',data = train_data, hue='Survived')
f,ax = plt.subplots(figsize=(12,10))

sns.heatmap(train_data.corr(), annot = True, cmap='coolwarm')
corr_matrix = train_data.corr()

corr_matrix['Survived'].sort_values(ascending = False)
sns.countplot(train_data[train_data['Survived']==1]['Pclass']).set_title('Count Survived people for each class')
len(train_data[train_data['Pclass'] == 1]), len(train_data[train_data['Pclass'] == 2]), len(train_data[train_data['Pclass'] == 3])
train_data[train_data['Pclass'] == 1]['Survived'].sum(), train_data[train_data['Pclass'] == 2]['Survived'].sum(), train_data[train_data['Pclass'] == 3]['Survived'].sum()
precentages = []

first = 136/216

seconds = 87/184

third = 119/491



precentages.append(first)

precentages.append(seconds)

precentages.append(third)
percents = pd.DataFrame(precentages)

percents.index += 1
percents['Pclass'] = ['1','2','3']

cols = ['Percent','Pclass']

percents.columns = [i for i in cols]

sns.barplot(y = 'Percent',x= 'Pclass', data = percents).set_title('Percentage of survived passenger class')
train_data.isna().sum()
test_data.isna().sum()
df = [train_data,test_data]



for d in df:

    d['Age'].fillna(d['Age'].median(),inplace=True)
train_data['Cabin'].value_counts()
for d in df:

    d['Cabin'].fillna('C',inplace=True)
train_data['Cabin'].isna().sum()
cabins = []

for i in train_data['Cabin']:

    cabins.append(str(i))
words = []

for i in cabins:

    word = i[0]

    words.append(word)
train_data['Cabin'] = words
train_data['Cabin'].head()
train_data['Cabin'].value_counts()
cabins = []

for i in test_data['Cabin']:

    cabins.append(str(i))
words = []

for i in cabins:

    word = i[0]

    words.append(word)
test_data['Cabin'] = words
test_data['Cabin'].value_counts()
train_data['Embarked'].isna().sum()
train_data['Embarked'].value_counts()
for d in df:

    d['Embarked'].fillna('S',inplace=True)
train_data.isna().sum()
for d in df:

    d['Fare'].fillna(d['Fare'].mean(),inplace = True)
test_data.isna().sum()
train_data['Family'] = train_data.apply(lambda x: x['SibSp'] + x['Parch'], axis = 1)

test_data['Family'] = test_data.apply(lambda x: x['SibSp'] + x['Parch'], axis = 1)
train_data.drop(['SibSp','Name','Ticket','Parch'], axis = 1,inplace = True)

test_data.drop(['SibSp','Name','Ticket','Parch'], axis = 1, inplace = True)
train_df = pd.get_dummies(train_data)

test_df = pd.get_dummies(test_data)
train_df.drop('PassengerId', axis = 1, inplace = True)
y = train_df['Survived']

train_df.drop('Survived', axis=1, inplace = True)

train_df.drop('Cabin_T', axis=1, inplace = True)

test_df.drop('PassengerId',axis=1, inplace=True)

X = train_df

X_test = test_df
rfc = RandomForestClassifier()
param_grid = {

    'n_estimators':[200,500,1000],

    'max_features':['auto'],

    'max_depth': [6, 7, 8],

    'criterion': ['entropy']

    }
CV = GridSearchCV(estimator = rfc, param_grid = param_grid, cv=5)

CV.fit(X,y)

CV.best_estimator_
rfc = RandomForestClassifier(criterion='entropy', max_depth=8, n_estimators=200)

ada = AdaBoostClassifier()

gbc = GradientBoostingClassifier()
rfc.fit(X,y)

ada.fit(X,y)

gbc.fit(X,y)
model = StackingCVClassifier(classifiers = (rfc,ada,gbc),

                                 meta_classifier = rfc,

                                 use_features_in_secondary = True)
model.fit(X.values,y)
print(model.score(X, y))
prediction = model.predict(X_test.values)
output = pd.DataFrame({'PassengerId' : test_data.PassengerId, 'Survived' : prediction})

output.to_csv('my_submissions.csv', index = False)