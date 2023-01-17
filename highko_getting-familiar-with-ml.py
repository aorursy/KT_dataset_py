import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-darkgrid')

import os
print(os.listdir("../input"))

%matplotlib inline

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_train.head()
df_train.info()
df_test.info()
df_train.describe()
df_train.columns.values
# dropping the columns 'PassengerId', 'Ticket' and 'Cabin'
df_train = df_train.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1)
df_test = df_test.drop(['Ticket', 'Cabin'], axis=1)
df_train['Age'].isnull().sum()
g = sns.FacetGrid(df_train, col="Survived")
g.map(plt.hist, 'Age', bins = 15);
df_train['Pclass'].isnull().sum()
pd.crosstab(df_train.Pclass, df_train.Survived, margins=True).style.background_gradient(cmap='summer_r')
f, ax = plt.subplots(1, 2, figsize=(11,8))
df_train['Pclass'].value_counts().plot.bar(color=['#045FB4'], ax=ax[0])
ax[0].set_title('Number of passenger by Pclass')
ax[0].set_ylabel('Count')
ax[0]
sns.countplot('Pclass', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('Survived vs. Dead by Pclass')

plt.show();
df_train['Embarked'].isnull().sum()
pd.crosstab(df_train.Embarked, df_train.Survived ,margins=True).style.background_gradient(cmap='summer_r')
f, ax = plt.subplots(1, 2, figsize=(11,8))
df_train['Embarked'].value_counts().plot.bar(color=['#045FB4'], ax=ax[0])
ax[0].set_title('Number of passenger by Embarked')
ax[0].set_ylabel('Count')
sns.countplot('Embarked', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('Survived vs. Dead by Embarked')
plt.show();
sns.countplot('Embarked', hue="Pclass", data=df_train);
df_train['Sex'].isnull().sum()
pd.crosstab(df_train.Sex, df_train.Survived).style.background_gradient(cmap='summer_r')
sns.countplot('Sex', hue="Survived", data=df_train);
df_train['Fare'].isnull().sum()
f, ax = plt.subplots(1, 3, figsize=(18,8))
sns.distplot(df_train[df_train['Pclass'] == 1]['Fare'], kde=False, ax=ax[0])
ax[0].set_title('Fares in Pclass 1')
sns.distplot(df_train[df_train['Pclass'] == 2]['Fare'], kde=False, ax=ax[1])
ax[1].set_title('Fares in Pclass 2')
sns.distplot(df_train[df_train['Pclass'] == 3]['Fare'], kde=False, ax=ax[2])
ax[2].set_title('Fares in Pclass 3')
plt.show();
df_train['SibSp'].isnull().sum()
pd.crosstab(df_train.SibSp, df_train.Survived).style.background_gradient(cmap='summer_r')
sns.barplot('SibSp', 'Survived', data = df_train, errwidth=0);
df_train['Parch'].isnull().sum()
pd.crosstab(df_train.Parch, df_train.Survived).style.background_gradient(cmap='summer_r')
sns.barplot('Parch', 'Survived', data = df_train, errwidth=0);
all_data = [df_train, df_test]
for dataset in all_data:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
    
df_train.head()
most_freq_port = df_train.Embarked.dropna().mode()[0]
most_freq_port
for dataset in all_data:
    dataset['Embarked'] = dataset['Embarked'].fillna(most_freq_port)
    
df_train['Embarked'].unique()
for dataset in all_data:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    
df_train.head()
for dataset in all_data:
    dataset['Initial'] = dataset.Name.str.extract('([A-Za-z]+)\.')
    
pd.crosstab(df_train['Initial'], df_train['Sex']).T
for dataset in all_data:
    dataset['Initial'] = dataset['Initial'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Dr. ', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Initial'] = dataset['Initial'].replace('Mlle', 'Miss')
    dataset['Initial'] = dataset['Initial'].replace('Ms', 'Miss')
    dataset['Initial'] = dataset['Initial'].replace('Mme', 'Mrs')
df_train[['Initial', 'Survived']].groupby(['Initial'], as_index=False).mean()
df_train.groupby('Initial')['Age'].mean()
for dataset in all_data:
    dataset.loc[(dataset.Age.isnull())&(dataset.Initial=='Mr'),'Age']=33
    dataset.loc[(dataset.Age.isnull())&(dataset.Initial=='Mrs'),'Age']=36
    dataset.loc[(dataset.Age.isnull())&(dataset.Initial=='Master'),'Age']=5
    dataset.loc[(dataset.Age.isnull())&(dataset.Initial=='Miss'),'Age']=22
    dataset.loc[(dataset.Age.isnull())&(dataset.Initial=='Other'),'Age']=46
df_train[df_train['Age'].isnull()]
df_train['AgeRange'] = pd.cut(df_train['Age'], 5)
df_train[['AgeRange', 'Survived']].groupby(['AgeRange'], as_index=False).mean().sort_values(by='AgeRange', ascending=True)
df_train[['AgeRange', 'Survived']].groupby(['AgeRange'], as_index=False).count().sort_values(by='AgeRange', ascending=True)
for dataset in all_data:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4
    dataset['Age'] = dataset['Age'].astype(int)

df_train = all_data[0]
df_test = all_data[1]
df_train = df_train.drop(['AgeRange'], axis=1)
all_data = [df_train, df_test]
df_train.head()
df_train['FareRange'] = pd.cut(df_train['Fare'], 4)
df_train[['FareRange', 'Survived']].groupby(['FareRange'], as_index=False).mean().sort_values(by='FareRange', ascending=True)
df_train[['FareRange', 'Survived']].groupby(['FareRange'], as_index=False).count().sort_values(by='FareRange', ascending=True)
df_train.head()
for dataset in all_data:
    dataset.loc[dataset['Fare'] <= 128.082, 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 128.082) & (dataset['Fare'] <= 256.165), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 256.165) & (dataset['Fare'] <= 384.247), 'Fare'] = 3
    dataset.loc[dataset['Fare'] > 384.247, 'Fare'] = 4
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)

df_train = all_data[0]
df_test = all_data[1]
df_train = df_train.drop(['FareRange'], axis=1)
all_data = [df_train, df_test]
df_train.head()
df_train[['Initial', 'Survived']].groupby(['Initial'], as_index=False).mean()
for dataset in all_data:
    dataset['Initial'] = dataset['Initial'].fillna(0)
df_train['Initial'] = df_train['Initial'].replace(['Mr', 'Miss', 'Mrs', 'Master', 'Other'],[1, 2, 3, 4, 5]).astype(int)
df_test['Initial'] = df_test['Initial'].replace(['Mr', 'Miss', 'Mrs', 'Master', 'Other'],[1, 2, 3, 4, 5]).astype(int)
df_train.head()
df_train = df_train.drop(['Name'], axis=1)
df_test = df_test.drop(['Name'], axis=1)
all_data = [df_train, df_test]
for dataset in all_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
df_train = df_train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
df_test = df_test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
all_data = [df_train, df_test]

df_train.head()
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
train, test = train_test_split(df_train, test_size=0.3, stratify=df_train['Survived'])
x_train = train.drop('Survived', axis=1)
y_train = train['Survived']
x_test = test.drop('Survived', axis=1)
y_test = test['Survived']
# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred_nb = gaussian.predict(x_test)
print(gaussian.score(x_train, y_train))
print('Accuracy score for Naive Bayes: ', accuracy_score(y_test, y_pred_nb))
print('Recall score for Naive Bayes: ', recall_score(y_test, y_pred_nb))
print('Precision score for Naive Bayes: ', precision_score(y_test, y_pred_nb))
# Support Vector Machines

svc = SVC()

# grid search for optimal parameters
svc_param_grid = {
    'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'kernel': ['poly', 'linear', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],
    'gamma': [0.01, 0.1, 1.0],
    'decision_function_shape': ['ovo', 'ovr'],
    'coef0': [0.1, 0.2, 0.3, 0.4, 0.5]
}

svc_clf = GridSearchCV(svc, param_grid = svc_param_grid, n_jobs=8)

svc_clf.fit(x_train, y_train)
svc_best = svc_clf.best_estimator_
print(svc_best)

y_pred_svc = svc_clf.predict(x_test)
print(svc_clf.score(x_train, y_train))
print('Accuracy score for Support Vector Machines: ', accuracy_score(y_test, y_pred_svc))
print('Recall score for Support Vector Machines: ', recall_score(y_test, y_pred_svc))
print('Precision score for Support Vector Machines: ', precision_score(y_test, y_pred_svc))
# Random Forest

forest = RandomForestClassifier()

# grid search for optimal parameters
forest_param_grid = {
    'n_estimators': [5, 6, 7, 8, 9, 10], 'criterion': ['gini', 'entropy'], 'max_depth': [2, 3, 4, 5, 6, 7, 8], 'max_features': [2, 3, 4, 5, 6, 7], 'min_samples_split': [2, 3, 4, 5, 6, 7, 8]
}

forest_clf = GridSearchCV(forest, param_grid = forest_param_grid)

forest_clf.fit(x_train, y_train)
forest_best = forest_clf.best_estimator_
print(forest_best)

y_pred_forest = forest_clf.predict(x_test)
print(forest_clf.score(x_train, y_train))
print('Accuracy score for Random Forest: ', accuracy_score(y_test, y_pred_forest))
print('Recall score for Random Forest: ', recall_score(y_test, y_pred_forest))
print('Precision score for Random Forest: ', precision_score(y_test, y_pred_forest))
# Logistic Regression

logreg = LogisticRegression()

# grid search for optimal parameters
logreg_param_grid = {
    'tol': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
    'C': [1, 2, 3, 4, 5],
}

logreg_clf = GridSearchCV(logreg, param_grid = logreg_param_grid)

logreg_clf.fit(x_train, y_train)
logreg_best = logreg_clf.best_estimator_
print(logreg_best)

y_pred_logreg = logreg_clf.predict(x_test)
print(logreg_clf.score(x_train, y_train))
print('Accuracy score for Logistic Regression: ', accuracy_score(y_test, y_pred_logreg))
print('Recall score for Logistic Regression: ', recall_score(y_test, y_pred_logreg))
print('Precision score for Logistic Regression: ', precision_score(y_test, y_pred_logreg))
# Perceptron

perceptron = Perceptron(max_iter=20)

# grid search for optimal parameters
perceptron_param_grid = {
    'tol': [0.0001, 0.0001, 0.001, 0.01, 0.1, 1.0],
    'eta0': [1, 2, 3, 4, 5]
}

perceptron_clf = GridSearchCV(perceptron, param_grid = perceptron_param_grid)

perceptron_clf.fit(x_train, y_train)
perceptron_best = perceptron_clf.best_estimator_
print(perceptron_best)

y_pred_perceptron = perceptron_clf.predict(x_test)
print(perceptron_clf.score(x_train, y_train))
print('Accuracy score for Perceptron: ', accuracy_score(y_test, y_pred_perceptron))
print('Recall score for Perceptron: ', recall_score(y_test, y_pred_perceptron))
print('Precision score for Perceptron: ', precision_score(y_test, y_pred_perceptron))
# GradientBoostingClassifier

gb = GradientBoostingClassifier()

# grid search for optimal parameters
gb_param_grid = {
    'n_estimators': [77, 78, 79, 80, 81],
    'max_depth': [2, 3, 4],
    'min_samples_split': [2, 3, 4],
    'criterion': ['friedman_mse', 'mse', 'mae'],
    'max_features': [5, 6, 7]
}

gb_clf = GridSearchCV(gb, param_grid = gb_param_grid)

gb_clf.fit(x_train, y_train)
gb_best = gb_clf.best_estimator_
print(gb_best)

y_pred_gb = gb_clf.predict(x_test)
print(gb_clf.score(x_train, y_train))
print('Accuracy score for Gradient Boosting: ', accuracy_score(y_test, y_pred_gb))
print('Recall score for Gradient Boosting: ', recall_score(y_test, y_pred_gb))
print('Precision score for Gradient Boosting: ', precision_score(y_test, y_pred_gb))
# Stochastic Gradient Descent

#sgd = SGDClassifier(eta0=1, loss='hinge', max_iter=6, learning_rate='invscaling')
sgd = SGDClassifier()

# grid search for optimal parameters
sgd_param_grid = {
    'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'max_iter': list(range(1, 10, 1)),
    'learning_rate': ['constant', 'optimal', 'invscaling'],
    'eta0': list(range(1, 10, 1))
}

sgd_clf = GridSearchCV(sgd, param_grid = sgd_param_grid)

sgd_clf.fit(x_train, y_train)
sgd_best = sgd_clf.best_estimator_
print(sgd_best)

y_pred_sgd = sgd_clf.predict(x_test)
print(sgd_clf.score(x_train, y_train))
print('Accuracy score for Stochastik Gradient Descent: ', accuracy_score(y_test, y_pred_sgd))
print('Recall score for Stochastik Gradient Descent: ', recall_score(y_test, y_pred_sgd))
print('Precision score for Stochastik Gradient Descent: ', precision_score(y_test, y_pred_sgd))
# LinearSVC

#linsvc = LinearSVC(loss='hinge')
linsvc = LinearSVC()

# grid search for optimal parameters
linsvc_param_grid = {
    'loss': ['hinge', 'squared_hinge'],
    'tol': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
}

linsvc_clf = GridSearchCV(linsvc, param_grid = linsvc_param_grid)

linsvc_clf.fit(x_train, y_train)
linsvc_best = linsvc_clf.best_estimator_
print(linsvc_best)

y_pred_linsvc = linsvc_clf.predict(x_test)
print(linsvc_clf.score(x_train, y_train))
print('Accuracy score for LinearSVC: ', accuracy_score(y_test, y_pred_linsvc))
print('Recall score for LinearSVC: ', recall_score(y_test, y_pred_linsvc))
print('Precision score for LinearSVC: ', precision_score(y_test, y_pred_linsvc))
# Adaboost

ab = AdaBoostClassifier()

# grid search for optimal parameters
ab_param_grid = {
    'learning_rate': [0.02, 0.03, 0.04, 0.05, 0.06],
    'n_estimators': list(range(100,1100,100))
}

ab_clf = GridSearchCV(ab, param_grid = ab_param_grid)

ab_clf.fit(x_train, y_train)
ab_best = ab_clf.best_estimator_
print(ab_best)

y_pred_ab = ab_clf.predict(x_test)
print(ab_clf.score(x_train, y_train))
print('Accuracy score for Adaboost: ', accuracy_score(y_test, y_pred_ab))
print('Recall score for Adaboost: ', recall_score(y_test, y_pred_ab))
print('Precision score for Adaboost: ', precision_score(y_test, y_pred_ab))
x_train = df_train.drop('Survived', axis=1)
y_train = df_train['Survived']
x_test = df_test.drop('PassengerId', axis=1)
logreg = LogisticRegression(tol=0.01)

logreg_param_grid = {
    'tol': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
    'C': [1, 2, 3, 4, 5],
}

logreg_clf = GridSearchCV(logreg, param_grid = logreg_param_grid)

logreg_clf.fit(x_train, y_train)
logreg_best = logreg_clf.best_estimator_
print(logreg_best)

y_pred_logreg = logreg_clf.predict(x_test)
print(logreg_clf.score(x_train, y_train))
submission = pd.DataFrame({
    'PassengerId': df_test['PassengerId'],
    "Survived": y_pred_logreg
})
#submission.to_csv('firstsubmission.csv', index=False)