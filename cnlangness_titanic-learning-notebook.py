# Main data analysis

import pandas as pd

import numpy as np

import random as rndm
# Visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Machine Learning - Basics

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
# Machine Learning - XGBoost and KFold for final model

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

from sklearn.base import TransformerMixin

from sklearn.cross_validation import KFold

from sklearn.metrics import accuracy_score
train_raw = pd.read_csv('../input/train.csv')

test_raw = pd.read_csv('../input/test.csv')

input_data = [train_raw, test_raw]
# Each additional parameter in the train_raw prints different values.

print(train_raw.columns.values)
train_raw.head()
train_raw.tail()
train_raw.info()

print('-'*40)

test_raw.info()
train_raw.describe()
train_raw.describe(include=['O'])
train_raw[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',ascending=False)
train_raw[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_raw[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_raw[['Parch','Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train_raw, col='Survived')

g.map(plt.hist, 'Age', bins = 20);
grid = sns.FacetGrid(train_raw, col = 'Survived', row = 'Pclass', size = 2.2, aspect = 1.6)

grid.map(plt.hist, 'Age', alpha = 0.5, bins = 20)

grid.add_legend();
grid = sns.FacetGrid(train_raw, row = 'Embarked', size = 2.2, aspect = 1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette = 'deep')

grid.add_legend();
grid = sns.FacetGrid(train_raw, row = 'Embarked', col = 'Survived', size = 2.2, aspect = 1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha = 0.5, ci = None)

grid.add_legend();
print("Before", train_raw.shape, test_raw.shape, input_data[0].shape, input_data[1].shape)



train_raw = train_raw.drop(['Ticket', 'Cabin'], axis = 1)

test_raw = test_raw.drop(['Ticket', 'Cabin'], axis = 1)

combine = [train_raw, test_raw]



print("After", train_raw.shape, test_raw.shape, combine[0].shape, combine[1].shape)
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand = False)



print(pd.crosstab(train_raw['Title'],train_raw['Sex']))
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr' ,'Jonkheer', 'Lady', 'Major', \

                                                 'Sir', 'Rev'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

print(pd.crosstab(train_raw['Title'],train_raw['Sex']))



print(train_raw[['Title', 'Survived']].groupby(['Title'], as_index = False).mean())
title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

train_raw.head()
train_raw = train_raw.drop(['Name', 'PassengerId'], axis = 1)

test_raw = test_raw.drop(['Name'], axis = 1)

combine = [train_raw, test_raw]

train_raw.shape, test_raw.shape
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train_raw.head()
grid = sns.FacetGrid(train_raw, row = 'Pclass', col = 'Sex', size = 2.2, aspect = 1.6)

grid.map(plt.hist, 'Age', alpha = 0.5, bins = 20)

grid.add_legend();
guess_ages = np.zeros((2,3))

guess_ages
for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna(axis = 0, how=None)

            guess_df = guess_df.dropna()

            age_guess = guess_df.median()

            guess_ages[i,j] = int( age_guess/0.5 + 0.5) * 0.5

        

    for i in range(0,2):

        for j in range(0,3):

            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1), 'Age'] = guess_ages[i,j]

        

    dataset['Age'] = dataset['Age'].astype(int)



train_raw.head()
train_raw['AgeBand'] = pd.cut(train_raw['Age'],5)

train_raw[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index = False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 4

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

train_raw.head()
train_raw = train_raw.drop(['AgeBand'], axis = 1)

combine = [train_raw, test_raw]

train_raw.head()
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_raw[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



train_raw[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index = False).mean().sort_values(by='Survived', ascending=False)
train_raw = train_raw.drop(['Parch','SibSp','FamilySize'], axis = 1)

test_raw = test_raw.drop(['Parch','SibSp','FamilySize'], axis = 1)

combine = [train_raw, test_raw]



train_raw.head()
for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



train_raw.loc[:,['Age*Class','Age','Pclass']].head(10)
freq_port = train_raw.Embarked.dropna().mode()[0]

print(freq_port)
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)



train_raw[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by = 'Survived', ascending = False)
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    

train_raw.head()
test_raw['Fare'].fillna(test_raw['Fare'].dropna().median(), inplace = True)

test_raw.head()
train_raw['FareBand'] = pd.qcut(train_raw['Fare'],4)

train_raw[['FareBand','Survived']].groupby(['FareBand'], as_index = False).mean().sort_values(by = 'FareBand', ascending = True)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] < 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2

    dataset.loc[(dataset['Fare'] > 31), 'Fare'] = 4

    dataset['Fare'] =dataset['Fare'].astype(int)



train_raw = train_raw.drop(['FareBand'], axis = 1)

combine = [train_raw, test_raw]



train_raw.head()
test_raw.head()
X_train = train_raw.drop('Survived', axis = 1)

Y_train = train_raw['Survived']

X_test = test_raw.drop('PassengerId', axis = 1).copy()

X_train.shape, Y_train.shape, X_test.shape
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

print(acc_log)
coeff_df = pd.DataFrame(train_raw.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df['Correlation'] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by = 'Correlation', ascending = False)
svc = SVC()

svc.fit(X_train, Y_train)

Y_pred_svc = svc.predict(X_test)

Y_train_svc = svc.predict(X_train)

acc_svc = round(svc.score(X_train, Y_train)*100, 2)

print(acc_svc)
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, Y_train)

Y_pred_knn = knn.predict(X_test)

Y_train_knn = knn.predict(X_train)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

print(acc_knn)
gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred_gaussian = gaussian.predict(X_test)

Y_train_gaussian = gaussian.predict(X_train)

acc_gausian = round(gaussian.score(X_train, Y_train) * 100 , 2)

print(acc_gausian)
perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred_perceptron = perceptron.predict(X_test)

Y_train_preceptron = perceptron.predict(X_train)

acc_perceptron = round(perceptron.score(X_train, Y_train)*100,2)

print(acc_perceptron)
linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred_svc = linear_svc.predict(X_test)

Y_train_svc = linear_svc.predict(X_train)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

print(acc_linear_svc)
sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred_sgd = sgd.predict(X_test)

Y_train_sgd = sgd.predict(X_train)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

print(acc_sgd)
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred_decision_tree = decision_tree.predict(X_test)

Y_train_decision_tree = decision_tree.predict(X_train)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

print(acc_decision_tree)
random_forest = RandomForestClassifier(n_estimators=200)

random_forest.fit(X_train, Y_train)

Y_pred_random_forest = random_forest.predict(X_test)

Y_train_random_forest = random_forest.predict(X_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print(acc_random_forest)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 'Random Forest', 'Naive Bayes',

              'Perceptron', 'Stochastic Gradient Descent', 'Linear SVC', 'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, acc_random_forest, acc_gausian,

              acc_perceptron, acc_sgd, acc_linear_svc, acc_decision_tree]

})



models.sort_values(by='Score', ascending = False)
X_train['RFC'] = Y_train_random_forest

X_train['DecisionTree'] = Y_train_decision_tree

X_train['KNN'] = Y_train_knn

X_train['SVM'] = Y_train_svc

X_train['Logisticreg'] = Y_train_gaussian

X_test['RFC'] = Y_pred_random_forest

X_test['DecisionTree'] = Y_pred_decision_tree

X_test['KNN'] = Y_pred_knn

X_test['SVM'] = Y_pred_svc

X_test['Logisticreg'] = Y_pred_gaussian
X_train['Combo'] = X_train['RFC'] + X_train['DecisionTree'] + X_train['KNN'] + X_train['SVM'] + X_train['Logisticreg']

X_train.head()

X_test['Combo'] = X_test['RFC'] + X_test['DecisionTree'] + X_test['KNN'] + X_test['SVM'] + X_test['Logisticreg']
knn_2 = KNeighborsClassifier(n_neighbors=3)

knn_2.fit(X_train, Y_train)

Y_pred_knn_2 = knn_2.predict(X_test)

acc_knn_2 = round(knn_2.score(X_train, Y_train) * 100, 2)

print(acc_knn_2)

random_forest_2 = RandomForestClassifier(n_estimators=200)

random_forest_2.fit(X_train, Y_train)

Y_pred_random_forest_2 = random_forest_2.predict(X_train)

acc_random_forest_2 = round(random_forest_2.score(X_train, Y_train) * 100, 2)

svc_2 = SVC()

svc_2.fit(X_train, Y_train)

Y_pred_svc_2 = svc_2.predict(X_test)

Y_train_svc_2 = svc_2.predict(X_train)

acc_svc_2 = round(svc_2.score(X_train, Y_train)*100, 2)

print(acc_svc_2)

print(acc_svc)

print(acc_random_forest_2)
class DataFrameImputer(TransformerMixin):

    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]

                              if X[c].dtype == np.dtype('O') else X[c].median() for c in X], index = X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.fill)
feature_columns_to_use = ['Pclass', 'Sex', 'Title', 'Embarked', 'IsAlone', 'Age', 'Fare', 'Age*Class']

nonnumeric_columns = ['Sex']
big_X = train_raw[feature_columns_to_use].append(test_raw[feature_columns_to_use])

big_X_imputed = DataFrameImputer().fit_transform(big_X)
le = LabelEncoder()

#for feature in nonnumeric_columns:

#    big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])
train_X = big_X_imputed[0:train_raw.shape[0]].as_matrix()

test_X = big_X_imputed[train_raw.shape[0]::].as_matrix()

train_y = train_raw['Survived']
gbm = xgb.XGBClassifier(max_depth=6, n_estimators=700, learning_rate=0.07).fit(train_X, train_y)

predictions = gbm.predict(test_X)

acc_xgb = round(gbm.score(train_X, train_y) * 100, 2)

print(acc_xgb)
train_raw.head()


def run_kfold():

    kf = KFold(len(train_raw.index), n_folds=10)

    outcomes_rndmfrst = []

    outcomes_gmb = []

    fold = 0

    x_raw = train_raw.drop(['Survived'], axis = 1)

    y_raw = train_raw['Survived']

    for train_index, test_index in kf:

        fold += 1

        X_train, X_test = x_raw.values[train_index], x_raw.values[test_index]

        y_train, y_test = y_raw.values[train_index], y_raw.values[test_index]

        gbm = xgb.XGBClassifier(max_depth=6, n_estimators=700, learning_rate=0.07).fit(X_train, y_train)

        random_forest.fit(X_train, y_train)

        Y_pred_rndmfrst = random_forest.predict(X_test)

        Y_pred_gbm = gbm.predict(X_test)

        acc_random_forest = accuracy_score(y_test, Y_pred_rndmfrst)

        acc_gbm = accuracy_score(y_test, Y_pred_gbm)

        outcomes_rndmfrst.append(acc_random_forest)

        outcomes_gmb.append(acc_gbm)

        print("Fold {0} Random Forest accuracy: {1}".format(fold, acc_random_forest))

        print("Fold {0} XGBoost accuracy: {1}".format(fold, acc_gbm))

    mean_outcome_rndmfrst = np.mean(outcomes_rndmfrst)

    print("Mean accuracy of Random Forest: {0}".format(mean_outcome_rndmfrst))

    mean_outcome_gbm = np.mean(outcomes_gmb)

    print("Mean accuracy of XGBoost: {0}".format(mean_outcome_gbm))



run_kfold()
X_train_new = X_train

def run_kfold():

    kf = KFold(len(train_raw.index), n_folds=10)

    outcomes_rndmfrst = []

    outcomes_gmb = []

    fold = 0

    x_raw = X_train_new

    y_raw = Y_train

    for train_index, test_index in kf:

        fold += 1

        X_train, X_test = x_raw.values[train_index], x_raw.values[test_index]

        y_train, y_test = y_raw.values[train_index], y_raw.values[test_index]

        gbm = xgb.XGBClassifier(max_depth=6, n_estimators=700, learning_rate=0.07).fit(X_train, y_train)

        random_forest.fit(X_train, y_train)

        Y_pred_rndmfrst = random_forest.predict(X_test)

        Y_pred_gbm = gbm.predict(X_test)

        acc_random_forest = accuracy_score(y_test, Y_pred_rndmfrst)

        acc_gbm = accuracy_score(y_test, Y_pred_gbm)

        outcomes_rndmfrst.append(acc_random_forest)

        outcomes_gmb.append(acc_gbm)

        print("Fold {0} Random Forest accuracy: {1}".format(fold, acc_random_forest))

        print("Fold {0} XGBoost accuracy: {1}".format(fold, acc_gbm))

    mean_outcome_rndmfrst = np.mean(outcomes_rndmfrst)

    print("Mean accuracy of Random Forest: {0}".format(mean_outcome_rndmfrst))

    mean_outcome_gbm = np.mean(outcomes_gmb)

    print("Mean accuracy of XGBoost: {0}".format(mean_outcome_gbm))

    sd_outcome_rndmfrst = np.std(outcomes_rndmfrst)

    print("Standard Deviation of accuracy of Random Forest: {0}".format(sd_outcome_rndmfrst))

    sd_outcome_gbm = np.std(outcomes_gmb)

    print("Standard Deviation of accuracy of XGBoost: {0}".format(sd_outcome_gbm))

    return(gbm)



gbm=run_kfold()
Y_pred_gbm = gbm.predict(X_test.values)
submission = pd.DataFrame({

        "PassengerId": test_raw["PassengerId"],

        "Survived": Y_pred_gbm

    })

submission.head()

submission.to_csv("../output/submission.csv", index=False)