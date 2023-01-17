import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')





data_train.Age = data_train.Age.fillna(data_train.Age.mean())



data_train.isnull().sum()

from sklearn.model_selection import train_test_split



mapping = {'male':0,

          'female':1}

data_train['Sex'] = data_train['Sex'].map(mapping)





X_all = data_train[['Pclass','Sex','Age']]

y_all = data_train['Survived']



num_test = 0.20

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV



clf = RandomForestClassifier()



parameters = {'n_estimators': [4, 6, 9], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }





acc_scorer = make_scorer(accuracy_score)



grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train, y_train)



clf = grid_obj.best_estimator_



clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(accuracy_score(y_test, predictions))
from sklearn.cross_validation import KFold



kf = KFold(891, n_folds=10)    

outcomes = []

    

fold = 0

for train_index, test_index in kf:

    fold += 1

    X_train, X_test = X_all.values[train_index], X_all.values[test_index]

    y_train, y_test = y_all.values[train_index], y_all.values[test_index]

    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    outcomes.append(accuracy)

    print("Fold {0} accuracy: {1}".format(fold, accuracy))     

mean_outcome = np.mean(outcomes)

print("\n\nMean Accuracy: {0}".format(mean_outcome)) 