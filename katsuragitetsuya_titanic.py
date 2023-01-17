import pandas as pd 

import numpy as np

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
sns.countplot('Sex', hue='Survived', data=train)
train.isnull().sum()
test.isnull().sum()
def preprocess(data):

    # 欠損値の補完

    data['Fare'] = data['Fare'].fillna(data['Fare'].mean())

    data['Age'] = data['Age'].fillna(data['Age'].mean())

    data['Embarked'] = data['Embarked'].fillna('S')



    # カテゴリを削除

    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    

    data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)

    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

    data['Alone'] = 0

    data.loc[data['FamilySize'] == 1, 'Alone'] = 1

    

    return data



train = preprocess(train)
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
from sklearn.model_selection import train_test_split

X = train.drop('Survived', axis=1)

y = train['Survived'].values

train_X, test_X ,train_y, test_y = train_test_split(X, y, random_state = 0)
model = RandomForestClassifier(criterion='gini',

            max_depth=10, max_features='auto', min_samples_leaf=10, min_samples_split=15,

            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=5, random_state=0, verbose=0)

model = model.fit(train_X, train_y)

pred = model.predict(test_X)



# parameters = {

#     'n_estimators':[i for i in range(10, 100, 20)],

#     'criterion':["gini", "entropy"],

#     'max_depth':[i for i in range(10, 30, 10)],

#     'min_samples_leaf':[i for i in range(10, 30, 10)],

#     'min_samples_split': [i for i in range(5, 20, 5)],

#     'random_state': [0],

#     'verbose': [0],

#     'max_features': ['auto'],

#     'min_weight_fraction_leaf': [0],

#     'n_jobs': [i for i in range(1, 8, 1)]

# }



# model = GridSearchCV(RandomForestClassifier(), parameters, cv=5, n_jobs=-1)

# model.fit(train_X, train_y)



# result = pd.DataFrame.from_dict(model.cv_results_)

# result.to_csv('gs_result.csv')



# best_model = model.best_estimator_

# pred = best_model.predict(test_X)



print(accuracy_score(test_y, pred))
features = train_X.columns

importances = model.feature_importances_

indexes = np.argsort(importances)



plt.figure(figsize=(6,6))

plt.barh(range(len(indexes)), importances[indexes], color='lightblue', align='center')

plt.yticks(range(len(indexes)), features[indexes])

plt.show()
test_formatted = preprocess(test)

pred = model.predict(test_formatted)



sub = pd.DataFrame(test['PassengerId'])

sub['Survived'] = list(map(int, pred))

sub.to_csv('submission.csv', index=False)