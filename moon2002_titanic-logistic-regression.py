import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head()
train.describe
train.shape
train.isnull().sum()
sns.heatmap(train.isnull(), cbar = False, cmap = 'viridis')
sns.heatmap(test.isnull(), cbar = False, cmap = 'viridis') #test
train.Age.isnull().sum()/train.shape[0]*100  #calculating what percentage of the Ages are null values
ax = train.Age.hist(bins = 30, density = True, stacked = True, color = 'magenta', alpha = 0.7, figsize = (16, 5))

train.Age.plot(kind = 'density')

ax.set_xlabel('Age')

plt.show()
#how age affected survival



Survived = 'Survived'

not_survived = 'did not survive'



fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 4))

women = train[train['Sex'] == 'female']

men = train[train['Sex'] == 'male']



ax = sns.distplot(women[women[Survived]==1].Age.dropna(), bins = 18, label = Survived, ax = axes[0], kde = False)

ax = sns.distplot(women[women[Survived]==0].Age.dropna(), bins = 40, label = not_survived, ax = axes[0], kde = False)

ax.legend()

ax.set_title('Female Survival')



ax = sns.distplot(men[men[Survived]==1].Age.dropna(), bins = 18, label = Survived, ax = axes[1], kde = False)

ax = sns.distplot(men[men[Survived]==0].Age.dropna(), bins = 40, label = not_survived, ax = axes[1], kde = False)

ax.legend()

ax.set_title('Male Survival')
train.Sex.value_counts()
sns.catplot(x = 'Pclass', y = 'Age', data = train, kind = 'box')
sns.catplot(x = 'Pclass', y = 'Fare', data = train, kind = 'box')
train[train['Pclass'] == 1]['Age'].mean() #mean age of 1st class passengers
train[train['Pclass'] == 2]['Age'].mean()
train[train['Pclass'] == 3]['Age'].mean()
print(test[test['Pclass'] == 1]['Age'].mean())

print(test[test['Pclass'] == 2]['Age'].mean())

print(test[test['Pclass'] == 3]['Age'].mean())
#fill missing ages with mean of Pclass Age



def impute_Age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return train[train['Pclass'] == 1]['Age'].mean()

        elif Pclass == 2:

             return train[train['Pclass'] == 2]['Age'].mean()

        elif Pclass == 3:

            return train[train['Pclass'] == 2]['Age'].mean()

        

    else:

        return Age
train['Age'] = train[['Age', 'Pclass']].apply(impute_Age, axis = 1)
sns.heatmap(train.isnull(), cbar = False, cmap = 'viridis')
test['Age'] = test[['Age', 'Pclass']].apply(impute_Age, axis = 1)
sns.heatmap(test.isnull(), cbar = False, cmap = 'viridis')
f = sns.FacetGrid(train, row = 'Embarked', height = 2.5, aspect = 3)

f.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', order = None, hue_order = None)

f.add_legend()
train.Embarked.isnull().sum()
train['Embarked'].value_counts()
common_value = 'S'

train['Embarked'].fillna(common_value, inplace = True)
train.Embarked.isnull().sum()
test.Fare.isnull().sum()
fill_fare = test.Fare.mean()
test['Fare'].fillna(fill_fare, inplace = True)
test.Fare.isnull().sum()
#drop cabin and ticket column



train.drop(labels = ['Cabin', 'Ticket', 'Name', 'PassengerId'], inplace = True, axis = 1)

test.drop(labels = ['Cabin','Name', 'Ticket'], inplace = True, axis = 1)
sns.heatmap(train.isnull(), cbar = False, cmap = 'viridis')
sns.heatmap(test.isnull(), cbar = False, cmap = 'viridis')
#feature transformation, categorical values into integers



train.head()
train.info()
train.Fare = train.Fare.astype('int')

train.Age = train.Age.astype('int')

train.info()
test.Fare = train.Fare.astype('int')

test.Age = train.Age.astype('int')

test.info()
Gender = {'male': 0, 'female': 1}

train.Sex = train.Sex.map(Gender)



Port = {'S': 0, 'C': 1, 'Q': 2}

train.Embarked = train.Embarked.map(Port)

train.head()
Gender = {'male': 0, 'female': 1}

test.Sex = test.Sex.map(Gender)



Port = {'S': 0, 'C': 1, 'Q': 2}

test.Embarked = test.Embarked.map(Port)

test.head()
test.info()
X = train.drop('Survived', axis = 1)

y = train.Survived



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

X_train.shape
model = LogisticRegression(solver = 'lbfgs', max_iter = 400)

model.fit(X_train, y_train)

y_predict = model.predict(X_test)
model.score(X_test, y_test)
test.head()
wanted_test_columns = X_train.columns

wanted_test_columns
predictions = model.predict(test[wanted_test_columns])
submission = pd.DataFrame()

submission['PassengerId'] = test['PassengerId']

submission['Survived'] = predictions

submission
submission.to_csv('../logisticregression_submission.csv', index=False)
test_submission = pd.read_csv('../logisticregression_submission.csv')

test_submission