# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()


from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)





output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
train_data.head()
train_data.info()
# Pclass

g = sns.catplot("Survived", col="Pclass", col_wrap=3,



                data=train_data,



                kind="count", height=2.5, aspect=.8)
train_data['Name_clean'] = train_data['Name'].apply(lambda x: x.split(',')[1].lstrip())

train_data['MrMs'] =  train_data['Name_clean'].apply(lambda x: x.split(' ')[0].strip())

train_data.loc[~train_data['MrMs'].isin(['Mr.', 'Miss.', 'Mrs.', 'Master.']), 'MrMs'] = 'Other'

train_data['MrMs'].value_counts()
# # MrMs

g = sns.catplot("Survived", col="MrMs", col_wrap=3,



                data=train_data,



                kind="count", height=2.5, aspect=.8)

mrms = train_data.groupby('MrMs').agg(

    survivors = ('Survived', 'sum'),

    total = ('Survived', 'count')

).reset_index()

mrms['pct_survive'] = mrms['survivors']/mrms['total']

mrms.plot(

    x='MrMs', y='pct_survive'

)
# Sex'

g = sns.catplot("Survived", col="Sex", col_wrap=3,



                data=train_data,



                kind="count", height=2.5, aspect=.8)
# Sex

train_data['Sex_mod'] = train_data['Sex']

train_data.loc[train_data['Age'] <= 12, 'Sex_mod'] = 'children'

g = sns.catplot("Survived", col="Sex_mod", col_wrap=3,



                data=train_data,



                kind="count", height=2.5, aspect=.8)
# Age

sns.distplot(train_data[train_data['Survived']==1]["Age"] , color="green", label="Survived")

sns.distplot(train_data[train_data['Survived']==0]['Age'] , color="red", label="Died")

plt.show()
# How many are inputed and how many are real ages

train_data['Age_imputed'] = (train_data['Age']%1)>0



fig, ax = plt.subplots(figsize=(6,4))

train_data.groupby('Age_imputed').size().plot(

    ax=ax, kind='bar'

)

plt.show()
# SibSp

g = sns.catplot("Survived", col="SibSp", col_wrap=3,



                data=train_data,



                kind="count", height=2.5, aspect=.8)
# Grouping all users who have at least 1 sibiling/spouses in Titanic

train_data['SibSp_1'] = (train_data['SibSp']>0).astype(int)

g = sns.catplot("Survived", col="SibSp_1", col_wrap=3,

                data=train_data,

                kind="count", height=2.5, aspect=.8)
# Parch: # of parents / children aboard the Titanic

g = sns.catplot("Survived", col="Parch", col_wrap=3,

                data=train_data,

                kind="count", height=2.5, aspect=.8)
# Grouping Parch: # of parents / children aboard the Titanic

train_data['Parch_1'] = (train_data['Parch']>0).astype(int)

g = sns.catplot("Survived", col="Parch_1", col_wrap=3,

                data=train_data,

                kind="count", height=2.5, aspect=.8)
# Ticket number

train_data['Ticket']

# Data has to be formatted
ticket = pd.DataFrame(train_data['Ticket'].str.split(' '))

ticket['len'] =  ticket['Ticket'].apply(lambda x: len(x))

ticket['element_0'] = np.where(ticket['len']==1, 'NA', ticket['Ticket'].apply(lambda x: x[0]))

ticket['element_0'].value_counts()
ticket
# Fare

sns.distplot(train_data[train_data['Survived']==1]["Fare"] , color="green", label="Survived")

sns.distplot(train_data[train_data['Survived']==0]['Fare'] , color="red", label="Died")

plt.show()
print('Survived-----')

print(train_data[train_data['Survived']==1]['Fare'].describe())

print('Died-----')

print(train_data[train_data['Survived']==0]['Fare'].describe())
sns.boxplot(x="Survived", y="Fare", data=train_data)

plt.show()
g = sns.catplot(x="Survived", y="Fare", data=train_data,

                height=5, aspect=.8)
# Transforming fare (log)

train_data['Fare_log'] = np.log(train_data['Fare']+1)

sns.distplot(train_data[train_data['Survived']==1]["Fare_log"] , color="green", label="Survived")

sns.distplot(train_data[train_data['Survived']==0]['Fare_log'] , color="red", label="Died")

plt.show()
# Cabin: A lot of nulls so no looking in here for now

# Can I infer cabin from fare?

train_data['Cabin_category'] = train_data['Cabin'].str[0]

train_data['Cabin_category'].value_counts()
g = sns.catplot("Survived", col="Cabin_category", col_wrap=3,

                data=train_data.sort_values(by='Cabin_category'),

                kind="count", height=2.5, aspect=.8)
cabin_categories = train_data.groupby('Cabin_category').agg(

    survivors = ('Survived', 'sum'),

    total = ('Survived', 'count')

).reset_index()

cabin_categories['pct_survive'] = cabin_categories['survivors']/cabin_categories['total']

cabin_categories.plot(

    x='Cabin_category', y='pct_survive'

)
g = sns.FacetGrid(train_data[train_data['Cabin_category']!='T'].sort_values(by='Cabin_category')

                  , col="Cabin_category", palette="Set1", sharey=False, col_wrap=1,

                 aspect = 3)

g.map(sns.distplot, 'Fare')

plt.show()
# Embarked: Port of Embarkation 	C = Cherbourg, Q = Queenstown, S = Southampton

g = sns.catplot("Survived", col="Embarked", col_wrap=3,

                data=train_data,

                kind="count", height=2.5, aspect=.8)
sns.heatmap(data=train_data.isnull(), cbar=False)

plt.show()
sns.heatmap(

    data=train_data[['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'SibSp',

       'Parch', 'Ticket', 'Fare', 'Embarked', 'Age', 'Cabin']].isna().sort_values(by=['Age', 'Cabin']),

    cbar=False)

plt.show()
# Age and SibSp_1

print('Without Sib')

sns.distplot(train_data[(train_data['Survived']==1) & (train_data['Sex']=='male')]["Age"] , color="green", label="Survived")

sns.distplot(train_data[(train_data['Survived']==0) & (train_data['Sex']=='male')]['Age'] , color="red", label="Died")

plt.show()

print('With Sib')

sns.distplot(train_data[(train_data['Survived']==1) & (train_data['Sex']=='female')]["Age"] , color="green", label="Survived")

sns.distplot(train_data[(train_data['Survived']==0) & (train_data['Sex']=='female')]['Age'] , color="red", label="Died")

plt.show()
# Feature engineering

test_data['SibSp_1'] = (test_data['SibSp']>0).astype(int)

test_data['Parch_1'] = (test_data['Parch']>0).astype(int)

test_data['Name_clean'] = test_data['Name'].apply(lambda x: x.split(',')[1].lstrip())

test_data['MrMs'] =  test_data['Name_clean'].apply(lambda x: x.split(' ')[0].strip())

test_data.loc[~test_data['MrMs'].isin(['Mr.', 'Miss.', 'Mrs.', 'Master.']), 'MrMs'] = 'Other'

test_data['Sex_mod'] = test_data['Sex']

test_data.loc[test_data['Age'] <= 12, 'Sex_mod'] = 'children'

test_data.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test_kaggle = pd.get_dummies(test_data[features])



# Model to test locally ----

model_small = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

scores_cv = cross_val_score(model_small, X, y, cv=5, scoring='accuracy')

print('Accuracy: ')

print(np.mean(scores_cv))





# Final model ----

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test_kaggle)





output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission_v0.csv', index=False)

print("Your submission was successfully saved!")

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp_1", "Parch_1", "Embarked"]

X = pd.get_dummies(train_data[features])

X_test_kaggle = pd.get_dummies(test_data[features])



# Model to test locally ----

model_small = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

scores_cv = cross_val_score(model_small, X, y, cv=5, scoring='accuracy')

print('Accuracy: ')

print(np.mean(scores_cv))





# Final model ----

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test_kaggle)





output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission_v0.csv', index=False)

print("Your submission was successfully saved!")
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp_1", "Parch_1", "Embarked"]

X = pd.get_dummies(train_data[features])

X['Age'] = train_data['Age'].fillna(train_data['Age'].mean())

X_test_kaggle = pd.get_dummies(test_data[features])

X_test_kaggle['Age'] = test_data['Age'].fillna(test_data['Age'].mean())



# Model to test locally ----

model_small = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

scores_cv = cross_val_score(model_small, X, y, cv=5, scoring='accuracy')

print('Accuracy: ')

print(np.mean(scores_cv))





# Final model ----

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test_kaggle)





output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission_v2.csv', index=False)

print("Your submission was successfully saved!")
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score



y = train_data["Survived"]



features = ["Pclass", "Sex", "Sex_mod", "SibSp_1", "Parch_1", "Embarked", 'MrMs']

X = pd.get_dummies(train_data[features])

X['Age'] = train_data['Age'].fillna(train_data['Age'].mean())

X['Fare'] = train_data['Fare'].fillna(train_data['Fare'].mean())

X_test_kaggle = pd.get_dummies(test_data[features])

X_test_kaggle['Age'] = test_data['Age'].fillna(test_data['Age'].mean())

X_test_kaggle['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())



# Model to test locally ----

model_small = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

scores_cv = cross_val_score(model_small, X, y, cv=5, scoring='accuracy')

print('Accuracy: ')

print(np.mean(scores_cv))





# Final model ----

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test_kaggle)





output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission_v4.csv', index=False)

print("Your submission was successfully saved!")
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score



y = train_data["Survived"]



features = ["Pclass", "Sex", "Sex_mod", "SibSp_1", "Parch_1", "Embarked", 'MrMs']

X = pd.get_dummies(train_data[features])

X['Age'] = train_data['Age'].fillna(train_data['Age'].mean())

# X['Fare'] = train_data['Fare'].fillna(train_data['Fare'].mean())

X_test_kaggle = pd.get_dummies(test_data[features])

X_test_kaggle['Age'] = test_data['Age'].fillna(test_data['Age'].mean())

# X_test_kaggle['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())



# Model to test locally ----

model_small = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

scores_cv = cross_val_score(model_small, X, y, cv=5, scoring='accuracy')

print('Accuracy: ')

print(np.mean(scores_cv))



# Final model ----

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test_kaggle)





output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission_v4.csv', index=False)

print("Your submission was successfully saved!")