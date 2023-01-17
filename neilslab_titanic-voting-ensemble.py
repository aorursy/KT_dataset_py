%matplotlib inline

import numpy as np

import pandas as pd

import re as re



train = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})

test  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})

full_data = [train, test]
for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

print (train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())
for dataset in full_data:

    age_avg 	   = dataset['Age'].mean()

    age_std 	   = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    

train['CategoricalAge'] = pd.cut(train['Age'], 5)



print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())
def get_title(name):

	title_search = re.search(' ([A-Za-z]+)\.', name)

	# If the title exists, extract and return it.

	if title_search:

		return title_search.group(1)

	return ""



for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)



print(pd.crosstab(train['Title'], train['Sex']))
for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4



# Feature Selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',\

                 'Parch', 'FamilySize']

train = train.drop(drop_elements, axis = 1)

train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)



test  = test.drop(drop_elements, axis = 1)



print (train.head(10))



train = train

test  = test
import numpy

import matplotlib.pyplot as plt

from pandas import read_csv

from pandas import set_option

from pandas.tools.plotting import scatter_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
X = pd.DataFrame(train)

Y = X.iloc[:,0]

X = X.drop('Survived', axis=1)

print(X.head())

print("==============================")

print(Y.head())
X = X.values

Y = Y.values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,

test_size=0.2, random_state=0)
folds = 10

scoring = 'accuracy'
models = []

models.append(('KNN', KNeighborsClassifier()))

models.append(('SVC', SVC()))

models.append(('GPC', GaussianProcessClassifier()))

models.append(('DTC', DecisionTreeClassifier()))

models.append(('RFC', RandomForestClassifier()))

models.append(('GBM', GradientBoostingClassifier()))

models.append(('ET', ExtraTreesClassifier()))

models.append(('MLP', MLPClassifier()))

models.append(('ABC', AdaBoostClassifier()))

models.append(('GNB', GaussianNB()))

models.append(('QDA', QuadraticDiscriminantAnalysis()))
scores = []

names = []

for name, model in models:

    kfold = KFold(n_splits=folds, random_state=0)

    cv_scores = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

    scores.append(cv_scores)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_scores.mean(), cv_scores.std())

    print(msg)
xKNN= 0.783725 /(0.058366)

xSVC= 0.820246 /(0.030537)

xGPC= 0.825880 /(0.043756)

xDTC= 0.790767 /(0.046608)

xRFC= 0.797790 /(0.055840)

xGBM= 0.820266 /(0.027638)

xET= 0.794953 /(0.043201)

xMLP= 0.809096 /(0.032059)

xABC= 0.809018 /(0.024148)

xGNB= 0.790806 /(0.039868)

xQDA= 0.810387 /(0.029615)



sharpe_ratios = pd.DataFrame([xKNN,xSVC,xGPC,xDTC,xRFC,xGBM,xET,xMLP,xABC,xGNB,xQDA])

sharpe_ratios.index = ['KNN', 'SVC', 'GPC', 'DTC', 'RFC', 'GBM', 'ET', 'MLP', 'ABC', 'GNB', 'QDA']

sharpe_ratios.columns = ['Ratios']

sharpe_ratios = sharpe_ratios.sort_values(by='Ratios', ascending=False)

sharpe_ratios
from sklearn.ensemble import VotingClassifier
estimators = []



x1 = AdaBoostClassifier()

estimators.append(('ABC', x1))

x2 = GradientBoostingClassifier()

estimators.append(('GBM', x2))

x3 = QuadraticDiscriminantAnalysis()

estimators.append(('QDA', x3))

x4 = SVC()

estimators.append(('SVC', x4))

x5 = MLPClassifier()

estimators.append(('MLP', x5))



ensemble = VotingClassifier(estimators)

results = cross_val_score(ensemble, X_train, Y_train, cv=kfold)

print(results.mean())
inputs = train.drop('Survived', axis=1)

targets = train['Survived']

ensemble.fit(inputs, targets)
print(confusion_matrix(targets, ensemble.predict(inputs)))

print(classification_report(targets, ensemble.predict(inputs)))
pred = ensemble.predict(test)
passenger = pd.DataFrame(list(range(892, 1310)))

survived = pd.DataFrame(pred)

sub = pd.concat([passenger, survived], axis=1)

sub.columns = ['PassengerId', 'Survived']

sub
sub.to_csv('./submission.csv', index=False)