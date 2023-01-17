import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



titanic = pd.read_csv('../input/train.csv')

titanic.head()
titanic.isnull().sum()
titanic.describe()
titanic.groupby(titanic['Age'].isnull())['Survived'].mean()
titanic["Age"].fillna(titanic['Age'].mean(), inplace=True)

titanic.head(10)
for i, col in enumerate(['SibSp', 'Parch']):

    plt.figure(i)

    sns.catplot(y='Survived', x=col, data=titanic, kind='point', aspect=2)
titanic['Famity_cnt'] = titanic['SibSp'] + titanic['Parch']
titanic.drop(['SibSp', 'Parch', 'PassengerId'], axis=1, inplace=True)
titanic.head(10)
titanic.isnull().sum()
titanic.groupby(titanic['Cabin'].isnull())['Survived'].mean()
titanic['Cabin_ind'] = np.where(titanic['Cabin'].isnull(), 0, 1)

titanic.head(10)
gender = {'male':0, 'female':1}

titanic['Sex'] = titanic['Sex'].map(gender)
titanic.head()
titanic.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

titanic.head()
titanic.to_csv('titanic_cleaned.csv', index=False)
from sklearn.model_selection import train_test_split
features = titanic.drop('Survived', axis=1)

labels = titanic['Survived']



X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
for dataset in [y_train, y_val, y_test]:

    print(round(len(dataset)/len(labels), 2))
X_train.to_csv('train_features.csv', index=False)

X_val.to_csv('val_features.csv', index=False)

X_test.to_csv('test_features.csv', index=False)



y_train.to_csv('train_labels', index=False)

y_val.to_csv('val_labels', index=False)

y_test.to_csv('test_labels', index=False)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV



import warnings



warnings.filterwarnings('ignore', category=FutureWarning)

warnings.filterwarnings('ignore', category=DeprecationWarning)
def print_result(cv):

    results = cv.cv_results_

    print(f"Best parameterers {cv.best_params_}")

    for mean_train, mean_test, params in zip(results['mean_train_score'], results['mean_test_score'], results['params']):

        print(f"mean train score: {round(mean_train, 2)}  mean test score: {round(mean_test, 2)} for {params}")
lr = LogisticRegression()

parameters = {

    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]

}

cv = GridSearchCV(lr, parameters, cv=5)

cv.fit(X_train, y_train.values.ravel())
print_result(cv)
cv.best_estimator_
import joblib

joblib.dump(cv.best_estimator_, 'LR_model.pkl')
from sklearn.svm import SVC



svc = SVC()

parameters = {

    'C': [0.1, 1, 10],

    'kernel': ['linear', 'rbf']

}



cv = GridSearchCV(svc, parameters, cv=5)

cv.fit(X_train, y_train.values.ravel())
print_result(cv)
joblib.dump(cv.best_estimator_, 'SVM_model.pkl')
from sklearn.neural_network import MLPClassifier



mlp = MLPClassifier()



parameters ={

    'hidden_layer_sizes' : [(10,), (50,), (10,)],

    'activation': ['relu', 'tanh', 'logistic'],

    'learning_rate': ['constant', 'invscaling', 'adaptive']

}



cv = GridSearchCV(mlp, parameters, cv=5)



cv.fit(X_train, y_train.values.ravel())



print_result(cv)



joblib.dump(cv.best_estimator_, 'MLP_model.pkl')
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier()

parameters = {

    'n_estimators':[5, 50, 250],

    'max_depth': [2, 4, 8, 16, 32, None]

}

cv = GridSearchCV(rf, parameters, cv=5)

cv.fit(X_train, y_train.values.ravel())

print_result(cv)
joblib.dump(cv.best_estimator_, 'RF_model.pkl')
from sklearn.ensemble import GradientBoostingClassifier



gb = GradientBoostingClassifier()

parameters ={

    'n_estimators': [5, 50, 250, 500],

    'max_depth': [1, 3, 5, 7, 9],

    'learning_rate':[0.1, 1, 10, 100]

}



cv = GridSearchCV(gb, parameters, cv=5)

cv.fit(X_train, y_train.values.ravel())

print_result(cv)
joblib.dump(cv.best_estimator_, 'GB_model.pkl')
from sklearn.metrics import accuracy_score, precision_score, recall_score

import time

def evaluate_model(name, model, features, labels):

    start = time.time()

    predictions = model.predict(features)

    end = time.time()

    accuracy = accuracy_score(labels, predictions)

    precision = precision_score(labels, predictions)

    recall = recall_score(labels, predictions)

    print(f"{name} accuracy: {round(accuracy, 3)} precision: {round(precision, 3)} recall: {round(recall, 3)}")
models = {}

for name in ['LR','SVM', 'MLP', 'RF', 'GB']:

    models[name] = joblib.load(f"{name}_model.pkl")
for model_name, model in models.items():

    evaluate_model(model_name, model, X_val, y_val.values.ravel())
evaluate_model('Gradient Boosted Trees', models['GB'], X_test, y_test)