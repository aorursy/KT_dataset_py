import pandas as pd

import numpy as np; np.random.seed()



#import matplotlib.pyplot as plt

#import seaborn as sns



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier



%matplotlib inline
X_train = pd.read_csv('../input/titanic/train.csv')

X_test = pd.read_csv('../input/titanic/test.csv')

X_train.head(3)
y_train = X_train.Survived
X_train.isna().mean()  # Доля пропусков в данных (NaN значений) по столбцам
X_test.isna().mean()  # Доля пропусков в данных (NaN значений) по столбцам
X = X_train.merge(X_test, how='outer')  # Для вычисления среднего возраста

med = X.Age.median()



X_train.fillna({'Age': med}, inplace=True)

X_test.fillna({'Age': med}, inplace=True)



X_train['Age'] = X_train['Age'].astype(int)

X_test['Age'] = X_train['Age'].astype(int)



X_train.head(3)
# Дропаем ненужные колонки

X_train.index = X_train['PassengerId']

X_test.index = X_test['PassengerId']

X_train.drop(['PassengerId', 'Survived', 'Ticket', 'Cabin'], axis=1, inplace=True)

X_test.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)

X_train.head(3)
X_train['NameLen'] = X_train['Name'].str.len()

X_test['NameLen'] = X_test['Name'].str.len()



X_train.drop('Name', axis=1, inplace=True)

X_test.drop('Name', axis=1, inplace=True)
X_train['Family'] = X_train['SibSp'] + X_train['Parch']

X_train['Family'].loc[X_train['Family'] > 0] = 1

X_train['Family'].loc[X_train['Family'] == 0] = 0



X_test['Family'] = X_test['SibSp'] + X_test['Parch']

X_test['Family'].loc[X_test['Family'] > 0] = 1

X_test['Family'].loc[X_test['Family'] == 0] = 0



X_train.drop(['SibSp', 'Parch'], axis=1, inplace=True)

X_test.drop(['SibSp', 'Parch'], axis=1, inplace=True)
# Разворачиваем колонки Sex и Embarked

X_train = pd.get_dummies(X_train)  

X_test = pd.get_dummies(X_test)



# Дропаем избыточные

X_train.drop(['Sex_male', 'Embarked_S'], axis=1, inplace=True)

X_test.drop(['Sex_male', 'Embarked_S'], axis=1, inplace=True)



X_train.head()
# У Fare в тестовых данных тоже были пропуски, заполним их медианой

med = (X.Fare.median()) // 2



X_train.fillna({'Fare': med}, inplace=True)

X_test.fillna({'Fare': med}, inplace=True)
# Наивный Байес



clf = GaussianNB()

clf.fit(X_train, y_train)



s = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1)

s.mean(), s.std()
# Логистическая регрессия



clf = LogisticRegression()

clf.fit(X_train, y_train)



s = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1)

s.mean(), s.std()
# kNN - k ближайших соседей



clf = KNeighborsClassifier(n_neighbors = 3)

clf.fit(X_train, y_train)



s = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1)

s.mean(), s.std()
# SVM



clf = SVC()

clf.fit(X_train, y_train)



s = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1)

s.mean(), s.std()
# Perceptron



clf = Perceptron()

clf.fit(X_train, y_train)



s = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1)

s.mean(), s.std()
# Решающие деревья



clf = DecisionTreeClassifier()

parameters = {'criterion': ['gini', 'entropy'], 

              'max_depth': range(1, 10, 1), 

              'min_samples_split': range(2, 10, 1), 

              'min_samples_leaf': range(1, 10, 1)}

search_cv = GridSearchCV(clf, parameters, cv=5, n_jobs=-1) #, n_iter=500)



search_cv.fit(X_train, y_train);

best_clf = search_cv.best_estimator_



s = cross_val_score(best_clf, X_train, y_train, cv=5, n_jobs=-1)

s.mean(), s.std()
y_pred_prob = best_clf.predict_proba(X_train)

pd.Series(y_pred_prob[:, 1]).hist();
# Случайный лес



clf = RandomForestClassifier()

parameters = {'n_estimators': range(1, 500),

              'criterion': ['gini', 'entropy'],

              'max_depth': range(1, 50),

              'min_samples_split': range(2, 100),

              'min_samples_leaf': range(1, 100)}

#search = GridSearchCV(clf, parameters, cv=5, n_jobs=-1) #, n_iter=5000)

search_cv = RandomizedSearchCV(clf, parameters, cv=5, n_jobs=-1, n_iter=1000)



search_cv.fit(X_train, y_train);

best_clf = search_cv.best_estimator_



s = cross_val_score(best_clf, X_train, y_train, cv=5, n_jobs=-1)

s.mean(), s.std()
best_clf.score(X_train, y_train)
search_cv.best_params_
y_pred_prob = best_clf.predict_proba(X_train)

pd.Series(y_pred_prob[:, 1]).hist();
y_pred = X_test.iloc[:, :0]

y_pred['Survived'] = best_clf.predict(X_test)
y_pred.to_csv('../output/gender_submission.csv')