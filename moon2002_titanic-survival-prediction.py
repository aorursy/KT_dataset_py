import pandas as pd

titanic = pd.read_csv('../input/train.csv')

titanic.head()
#drop name and passenger ID

titanic.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

titanic.head()
#explore continuous features

titanic.describe()
titanic.groupby('Survived').mean()
titanic.groupby(titanic['Age'].isnull()).mean()
import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

for i in ['Age', 'Fare']:

    died = list(titanic[titanic['Survived'] == 0][i].dropna())

    survived = list(titanic[titanic['Survived'] == 1][i].dropna())

    xmin = min(min(died), min(survived))

    xmax = max(max(died), max(survived))

    width = (xmax - xmin) / 40

    sns.distplot(died, color='r', kde=False, bins=np.arange(xmin, xmax, width))

    sns.distplot(survived, color='g', kde=False, bins=np.arange(xmin, xmax, width))

    plt.legend(['Did not survive', 'Survived'])

    plt.title('Overlaid histogram for {}'.format(i))

    plt.show()
titanic['Age'].fillna(titanic['Age'].mean(), inplace=True)

titanic.isnull().sum()
for i, col in enumerate(['Pclass', 'SibSp', 'Parch']):

    plt.figure(i)

    sns.catplot(x=col, y='Survived', data=titanic, kind='point', aspect=2,)
titanic['Family_cnt'] = titanic['SibSp'] + titanic['Parch']

titanic.drop(['SibSp', 'Parch'], axis=1, inplace=True)

titanic.head()
#explore categorical festures

titanic.info()
titanic.groupby(titanic['Cabin'].isnull())['Survived'].mean()
#whether or not the passenger had a cabin seems to be an indicator of survival

titanic['Cabin_ind'] = np.where(titanic['Cabin'].isnull(), 0, 1)

titanic.head(10)
for i, col in enumerate(['Cabin_ind', 'Sex', 'Embarked']):

    plt.figure(i)

    sns.catplot(x=col, y='Survived', data=titanic, kind='point', aspect=2, )
titanic.pivot_table('Survived', index='Sex', columns='Embarked', aggfunc='count')
titanic.pivot_table('Survived', index='Cabin_ind', columns='Embarked', aggfunc='count')
titanic.drop(['Cabin', 'Embarked'], axis=1, inplace=True)

titanic.head()
gender_num = {'male': 0, 'female': 1}



titanic['Sex'] = titanic['Sex'].map(gender_num)

titanic.head()
#train test split

from sklearn.model_selection import train_test_split

features = titanic.drop('Survived', axis=1)

labels = titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, random_state=42)
from sklearn.model_selection import GridSearchCV

import joblib
def print_results(results):

    print('BEST PARAMS: {}\n'.format(results.best_params_))



    means = results.cv_results_['mean_test_score']

    stds = results.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, results.cv_results_['params']):

        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

warnings.filterwarnings('ignore', category=DeprecationWarning)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

parameters = {

    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]

}



cv = GridSearchCV(lr, parameters, cv=5)

cv.fit(features, labels.values.ravel())



print_results(cv)
cv.best_estimator_
joblib.dump(cv.best_estimator_, '../../../LR_model.pkl')
from sklearn.svm import SVC



SVC()
svc = SVC()

parameters = {

    'kernel': ['linear', 'rbf'],

    'C': [0.1, 1, 10]

}



cv = GridSearchCV(svc, parameters, cv=5)

cv.fit(features, labels.values.ravel())



print_results(cv)
cv.best_estimator_
joblib.dump(cv.best_estimator_, '../../../SVM_model.pkl')
from sklearn.neural_network import MLPRegressor, MLPClassifier



print(MLPRegressor())

print(MLPClassifier())
mlp = MLPClassifier()

parameters = {

    'hidden_layer_sizes': [(10,), (50,), (100,)],

    'activation': ['relu', 'tanh', 'logistic'],

    'learning_rate': ['constant', 'invscaling', 'adaptive']

}



cv = GridSearchCV(mlp, parameters, cv=5)

cv.fit(features, labels.values.ravel())



print_results(cv)
cv.best_estimator_
joblib.dump(cv.best_estimator_, '../../../MLP_model.pkl')
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor



print(RandomForestClassifier())

print(RandomForestRegressor())
rf = RandomForestClassifier()

parameters = {

    'n_estimators': [5, 50, 250],

    'max_depth': [2, 4, 8, 16, 32, None]

}



cv = GridSearchCV(rf, parameters, cv=5)

cv.fit(features, labels.values.ravel())



print_results(cv)
cv.best_estimator_
joblib.dump(cv.best_estimator_, '../../../RF_model.pkl')
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor



print(GradientBoostingClassifier())

print(GradientBoostingRegressor())
gb = GradientBoostingClassifier()

parameters = {

    'n_estimators': [5, 50, 250, 500],

    'max_depth': [1, 3, 5, 7, 9],

    'learning_rate': [0.01, 0.1, 1, 10, 100]

}



cv = GridSearchCV(gb, parameters, cv=5)

cv.fit(features, labels.values.ravel())



print_results(cv)
cv.best_estimator_
joblib.dump(cv.best_estimator_, '../../../GB_model.pkl')
models = {}



for mdl in ['LR', 'SVM', 'MLP', 'RF', 'GB']:

    models[mdl] = joblib.load('../../../{}_model.pkl'.format(mdl))
from sklearn.metrics import accuracy_score, precision_score, recall_score

from time import time



def evaluate_model(name, model, features, labels):

    start = time()

    end = time()

    pred = model.predict(features)

    accuracy = round(accuracy_score(labels, pred), 3)

    precision = round(precision_score(labels, pred), 3)

    recall = round(recall_score(labels, pred), 3)

    print('{} -- Accuracy: {} / Precision: {} / Recall: {} / Latency: {}ms'.format(name,

                                                                                   accuracy,

                                                                                   precision,

                                                                                   recall,

                                                                                   round((end - start)*1000, 1)))
for name, mdl in models.items():

    evaluate_model(name, mdl, features, labels)
test = pd.read_csv('../input/test.csv')

test.head()
test.isnull().sum()
test['Age'].fillna(test['Age'].mean(), inplace=True)

test.isnull().sum()
test['Fare'].fillna(test['Fare'].mean(), inplace=True)

test.isnull().sum()
test['Cabin_ind'] = np.where(test['Cabin'].isnull(), 0, 1)

test.drop('Cabin', axis = 1, inplace = True)

test.head(10)
test['Family_cnt'] = test['SibSp'] + test['Parch']

test.drop(['SibSp', 'Parch'], axis=1, inplace=True)

test.head()
gender_num = {'male': 0, 'female': 1}

test['Sex'] = test['Sex'].map(gender_num)

test.head()
test_features = X_train.columns

test_features
model = GradientBoostingClassifier(learning_rate=0.1, max_depth=3, n_estimators=250)

model.fit(features, labels)
predictions = model.predict(test[test_features])

predictions
submission = pd.DataFrame()

submission['PassengerId'] = test['PassengerId']

submission['Survived'] = predictions

submission
submission.to_csv('../titanic_submission.csv', index=False)
test_submission = pd.read_csv('../titanic_submission.csv')

test_submission