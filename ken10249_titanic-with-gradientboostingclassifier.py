import numpy as np

import pandas as pd



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

import copy

from sklearn.model_selection import train_test_split
train = pd.read_csv('../input/train.csv')
train.head()
for i in train.columns:

    print(i, len(set(train[i])))



# so PassengerId and Name is useless
print(train.isnull().any())

print("")

for i in ['Age', 'Cabin', 'Embarked']:

    print(i, len(train[train[i].isnull()]))
train[train['Embarked'].isnull()]
train[~train['Embarked'].isnull()].groupby(['Embarked','Pclass']).agg({'Embarked' : {'count'}})
train.iloc[[61,829],11] = 'C'
train.iloc[train[train['Age'].isnull()].index,5] = -1
cabin_train = train[~train['Cabin'].isnull()].copy()
cabin_train['Cabin'] = cabin_train['Cabin'].apply(lambda x : x[0])
cabin_train[~cabin_train['Cabin'].isnull()].groupby(['Cabin']).agg({'Fare' : {'count','min','max','mean'}})
useful_train = train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y_train = useful_train['Survived']

x_train = useful_train.drop(labels = ['Survived'], axis = 1)
x_train = pd.get_dummies(x_train, prefix=['Sex', 'Embarked'])
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=1024)
parameters = {

    "loss":["deviance"],

    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],

    "min_samples_split": np.linspace(0.1, 0.5, 12),

    "min_samples_leaf": np.linspace(0.1, 0.5, 12),

    "max_depth":[8,10],

    "max_features":["log2","sqrt"],

    "criterion": ["friedman_mse",  "mae"],

    "subsample":[0.5, 0.8, 0.9,1.0],

    "n_estimators":[50,80]

    }



estimator = GradientBoostingClassifier(random_state = 1024)

model = GridSearchCV(estimator = estimator, param_grid = parameters, cv=5, scoring = 'accuracy')
model.fit(x_train, y_train)

print(model.score(x_train, y_train))

print(model.score(x_test, y_test))

print(model.best_params_)
result = pd.read_csv('../input/test.csv')
result.isnull().any()
useful_result = result[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y_result = useful_result['Survived']

x_result = useful_result.drop(labels = ['Survived'], axis = 1)
x_result = pd.get_dummies(x_result, prefix=['Sex', 'Embarked'])
predict = model.predict(x_result)
submission = pd.read_csv('../input/gender_submission.csv')
submission['Survived'] = predict
submission.to_csv('./gender_submission.csv', index = False)