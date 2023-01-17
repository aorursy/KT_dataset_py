# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.head()
train.shape
sns.countplot(train['Survived'])
test.head()
train.info()
test.info()
# count

train.isna().sum()
test.isna().sum()
sns.heatmap(train.isna(), yticklabels = False, cmap = 'Oranges')
sns.heatmap(test.isna(), yticklabels = False, cmap = 'Oranges')
# train data

percent = (train['Age'].isna().sum() / train.shape[0]) * 100

print("{:.2f} % of records in 'Age' are missing.".format(percent))
sns.distplot(train['Age'], kde=True)
train['Age'].mean()
train['Age'].median()
new_train = train.copy()

new_train['Age'].fillna(train['Age'].median(), inplace = True)
new_train.head()
new_train['Age'].isna().sum()
# test data

percent = (test['Age'].isna().sum() / test.shape[0]) * 100

print("{:.2f} % of records in 'Test' are missing.".format(percent))
sns.distplot(train['Age'], kde=True)
new_test = test.copy()

new_test['Age'].fillna(test['Age'].median(), inplace = True)
new_test['Age'].isna().sum()
# train data

percent = (train['Cabin'].isna().sum() / train.shape[0]) * 100

print("{:.2f} % of records in 'Cabin' are missing.".format(percent))
new_train['Cabin'].fillna('Unknown', inplace = True)
new_train.head()
new_train['Cabin'].isna().sum()
# test data

percent = (test['Cabin'].isna().sum() / test.shape[0]) * 100

print("{:.2f} % of records in 'Cabin' are missing.".format(percent))
new_test['Cabin'].fillna('Unknown', inplace = True)
new_test['Cabin'].isna().sum()
# train data

percent = (train['Embarked'].isna().sum() / train.shape[0]) * 100

print("{:.2f} % of records in 'Embarked' are missing.".format(percent))
train['Embarked'].value_counts()
new_train['Embarked'].fillna('S', inplace = True)
new_train.head()
new_train['Embarked'].isna().sum()
sns.heatmap(new_test.corr(), annot = True)
new_test.loc[new_test['Fare'].isna()]
mean_class_fare = new_test.groupby('Pclass')['Fare'].mean()

mean_class_fare
new_test['Fare'].fillna(mean_class_fare[3], inplace = True)
new_test['Fare'].isna().sum()
new_train.describe()
new_train['Pclass'].unique()
new_train['Pclass'].value_counts()
sns.countplot(new_train['Pclass'])
sns.countplot(x = 'Survived', hue = 'Pclass', data = new_train)
sns.barplot("Pclass", "Survived", data = new_train)
sns.swarmplot(x = 'Pclass', y = 'Fare', data = new_train)
new_train.groupby('Pclass')['Fare'].mean()
sns.distplot(new_train['Fare'], kde=False)
# How many people paid less than 50?

total_less = new_train.loc[(new_train['Fare'] < 50)].shape[0]
# How many of these survived?

survived_less = new_train.loc[(new_train['Fare'] < 50) & (new_train['Survived'] == 1)].shape[0]

print("{:.2f} % ({} of {}) of people with fare less than 50 survived.".format(((survived_less * 100)/total_less), survived_less, total_less))
# How many people paid more than 50?

total_more = new_train.loc[(new_train['Fare'] > 50)].shape[0]

total_more
survived_more = new_train.loc[(new_train['Fare'] > 50) & (new_train['Survived'] == 1)].shape[0]

print("{:.2f} % ({} of {}) of people with fare more than 50 survived.".format(((survived_more * 100)/total_more), survived_more, total_more))
new_train['FareGroup'] = new_train['Fare'].apply(lambda x: 0 if x < 50 else 1)

new_train.head()
sns.barplot("FareGroup", "Survived", data = new_train)
sns.countplot(new_train['Sex'])
new_train['Sex'].value_counts()
male = 577

female = 314
male_survivers = new_train.loc[(new_train['Survived'] == 1) & (new_train['Sex'] == 'male')].shape[0]

female_survivers = new_train.loc[(new_train['Survived'] == 1) & (new_train['Sex'] == 'female')].shape[0]
fig, ax = plt.subplots(1, 2)

ax[0].pie([female - female_survivers, female_survivers], 

          labels=['Female: non-survivers', 'Female: Survivers'], autopct="%1.1f%%", startangle = 90)



ax[1].pie([male - male_survivers, male_survivers], 

          labels=['Male: non-survivers', 'Male: Survivers'], autopct="%1.1f%%", startangle = 90)



plt.subplots_adjust(right=2)
sns.distplot(new_train['Age'], kde=False)
def age(x):

    if x > 0 and x <= 15:

        return 0

    elif x > 15 and x <= 40:

        return 1

    elif x > 40 and x <= 60:

        return 2

    else:

        return 3
new_train['AgeGroup'] = new_train['Age'].apply(age)
new_train.head()
# Children an young (0-15)

group0_survivers = new_train.loc[(new_train['Survived'] == 1) & (new_train['AgeGroup'] == 0)].shape[0]

group0_notsurvivers = new_train.loc[(new_train['Survived'] == 0) & (new_train['AgeGroup'] == 0)].shape[0]

fig, ax = plt.subplots()

ax.pie([group0_survivers, group0_notsurvivers], labels=['Children/Young who survived', 'Children/Young who did not survived'],

      autopct="%1.1f%%", startangle = 90)
# 15-40 years old

group1_survivers = new_train.loc[(new_train['Survived'] == 1) & (new_train['AgeGroup'] == 1)].shape[0]

group1_notsurvivers = new_train.loc[(new_train['Survived'] == 0) & (new_train['AgeGroup'] == 1)].shape[0]

# 41-60 years old

group2_survivers = new_train.loc[(new_train['Survived'] == 1) & (new_train['AgeGroup'] == 2)].shape[0]

group2_notsurvivers = new_train.loc[(new_train['Survived'] == 0) & (new_train['AgeGroup'] == 2)].shape[0]

# 61+ years old

group3_survivers = new_train.loc[(new_train['Survived'] == 1) & (new_train['AgeGroup'] == 3)].shape[0]

group3_notsurvivers = new_train.loc[(new_train['Survived'] == 0) & (new_train['AgeGroup'] == 3)].shape[0]
fig, ax = plt.subplots(1, 3)

ax[0].pie([group1_survivers, group1_notsurvivers], labels=['15-40 years old who survived', '15-40 years old who not survived'],

      autopct="%1.1f%%", startangle = 90, radius=1.2, textprops={'fontsize': 14})

ax[1].pie([group2_survivers, group2_notsurvivers], labels=['41-60 years old who survived', '41-60 years old who not survived'],

      autopct="%1.1f%%", startangle = 90, radius=1.2, textprops={'fontsize': 14})

ax[2].pie([group3_survivers, group3_notsurvivers], labels=['61+ years old who survived', '61+ years old who not survived'],

      autopct="%1.1f%%", startangle = 90, radius=1.2, textprops={'fontsize': 14})



plt.subplots_adjust(right=3)
sns.violinplot(x = 'Pclass', y = 'Age', data = new_train)
fig, ax = plt.subplots()

ax.pie([group0_survivers, group1_survivers, group2_survivers, group3_survivers], 

       labels=['Age 0-15', 'Age 15-40', 'Age 41-60', 'Age 61+'], autopct="%1.1f%%", startangle = 90)
cabins = new_train['Cabin'].unique()
cabins.sort()

cabins
def cabin(x):

    if x[0] == 'G' or x[0] == 'T':

        return 1

    elif x == 'Unknown':

        return 2

    else: 

        return 0
new_train['CabinGroup'] = new_train['Cabin'].apply(cabin)
new_train.head()
sns.countplot(x = 'Survived', hue = 'CabinGroup', data = new_train)
number_of_filled_cabins = new_train.loc[new_train['Cabin'] != 'Unknown'].shape[0]

number_of_non_filled_cabins = new_train.loc[new_train['Cabin'] == 'Unknown'].shape[0]
fig, ax = plt.subplots()

ax.pie([number_of_filled_cabins, number_of_non_filled_cabins], labels=["Known Cabin", "Unknown Cabin"], autopct="%1.1f%%", startangle = 90)
sns.barplot("CabinGroup", "Survived", data = new_train)
new_train['FamilySize'] = new_train['SibSp'] + new_train['Parch']
new_train.head()
new_train['FamilySize'].value_counts()
sns.countplot(x = 'Pclass', hue = 'FamilySize', data=new_train)
sns.countplot(x = "Survived", hue = "FamilySize", data = new_train)
new_train['Embarked'].unique()
new_train['Embarked'].value_counts()
sns.countplot(x = 'Survived', hue = 'Embarked', data = new_train)
sns.barplot('Embarked', 'Survived', data = new_train)
sns.countplot(x = 'Pclass', hue = 'Embarked', data = new_train)
new_train.head()
new_train.drop(['PassengerId', 'Name', 'Ticket'], axis = 1, inplace = True)
new_train.head()
new_train.drop(['FareGroup', 'AgeGroup'], axis = 1, inplace = True)
new_train.drop(['SibSp', 'Parch', 'Cabin'], axis = 1, inplace = True)
new_train.head()
new_test.head()
new_test.drop(['Name', 'Ticket'], axis = 1, inplace = True)
new_test['FamilySize'] = new_test['SibSp'] + new_test['Parch']
new_test['CabinGroup'] = new_test['Cabin'].apply(cabin)
new_test.head()
new_test.drop(['SibSp', 'Parch', 'Cabin'], axis = 1, inplace = True)
new_test.head()
# Independent columns

X = new_train.iloc[:, 1:8]

# Dependent column

y = new_train.iloc[:, 0].values
XTest = new_test.iloc[:, 1:8]
from sklearn.preprocessing import LabelEncoder
X.head()
labelencoder_X = LabelEncoder()

Xlabel = X.values

Xlabel[:, 1] = labelencoder_X.fit_transform(Xlabel[:, 1])

Xlabel[:, 4] = labelencoder_X.fit_transform(Xlabel[:, 4])
XTest.head()
labelencoder_Xtest = LabelEncoder()

XTest_predict = XTest.values

XTest_predict[:, 1] = labelencoder_X.fit_transform(XTest_predict[:, 1])

XTest_predict[:, 4] = labelencoder_X.fit_transform(XTest_predict[:, 4])
Xlabel
XTest_predict
from sklearn.ensemble import ExtraTreesClassifier



def feature_importances(X, y):

    model = ExtraTreesClassifier()

    model.fit(X, y)

    

    return model.feature_importances_
# Feature importances with label encoder

importances = feature_importances(Xlabel, y)
importances = pd.Series(importances, index = X.columns)

importances
importances.sort_values(ascending = False, inplace = True)
importances
sns.barplot(x = importances, y = importances.index, orient = 'h')
### StandardScaler (optional)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

Xlabel_scaled = scaler.fit_transform(Xlabel)

XTest_scaled = scaler.fit_transform(XTest_predict)
from sklearn.model_selection import train_test_split

def split_dataset(features, target, test_size = 0.2):

    X_train, X_test, y_train, y_test = train_test_split(features, target,

                                                       test_size = test_size,

                                                       stratify = target,

                                                       random_state = 0)

    

    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = split_dataset(Xlabel, y)
models = {}

models_results = {}

test_accuracy = {}
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

classifier_rf = RandomForestClassifier(random_state=0)

results_rf = cross_val_score(classifier_rf, X_train, y_train, cv = 10)
results_rf.mean()
results_rf.std()
from sklearn.model_selection import GridSearchCV
param_grid = {

    'max_depth': [None, 3, 6, 9, 12],

    'max_features': range(1, 7, 2),

    'min_samples_leaf': range(2, 10, 2),

    'min_samples_split': range(2, 10, 2),

    'n_estimators': range(50, 300, 50)

}



rf = RandomForestClassifier(random_state=0)

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,

                          cv = 5, n_jobs = -1, verbose = True,

                          scoring = 'accuracy')



grid_search.fit(X_train, y_train)

best_estimator = {'Random Forest': grid_search.best_estimator_}

best_score = {'Random Forest': grid_search.best_score_}



models = {**models, **best_estimator}

models_results = {**models_results, **best_score}



print("Best Estimator: ", grid_search.best_estimator_)

print("Best Score: ", grid_search.best_score_)
best_random_forest = RandomForestClassifier(max_depth=12, max_features=1, min_samples_leaf=2,

                       min_samples_split=6, n_estimators=50, random_state=0)

best_random_forest.fit(X_train, y_train)
rf_predict = best_random_forest.predict(X_test)
from sklearn.metrics import accuracy_score
test_accuracy = {**test_accuracy, **{'Random Forest': accuracy_score(y_test, rf_predict)}}

test_accuracy
from sklearn.ensemble import GradientBoostingClassifier

classifier_gbc = GradientBoostingClassifier(random_state=0)

results_gbc = cross_val_score(classifier_gbc, X_train, y_train, cv = 10)
results_gbc.mean()
results_gbc.std()
param_grid = {

    'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],

    'n_estimators': [100, 250, 500, 750, 1000, 1250, 1500]

}



gbc = GradientBoostingClassifier(random_state=0)

grid_search = GridSearchCV(estimator = gbc, param_grid = param_grid,

                          cv = 5, n_jobs = -1, verbose = True,

                          scoring = 'accuracy')



grid_search.fit(X_train, y_train)

best_estimator = {'Gradient Boosting': grid_search.best_estimator_}

best_score = {'Gradient Boosting': grid_search.best_score_}



models = {**models, **best_estimator}

models_results = {**models_results, **best_score}



print("Best Estimator: ", grid_search.best_estimator_)

print("Best Score: ", grid_search.best_score_)
best_gbc = GradientBoostingClassifier(learning_rate=0.05, n_estimators=500, random_state=0)

best_gbc.fit(X_train, y_train)
gbc_predict = best_gbc.predict(X_test)
test_accuracy = {**test_accuracy, **{'Gradient Boosting': accuracy_score(y_test, gbc_predict)}}

test_accuracy
XL_train, XL_test, yL_train, yL_test = split_dataset(Xlabel_scaled, y)
from sklearn.linear_model import LogisticRegression

classifier_lr = LogisticRegression(random_state=0)

results_lr = cross_val_score(classifier_lr, XL_train, yL_train, cv = 10)
results_lr.mean()
results_lr.std()
param_grid = {

    'penalty': ['l1', 'l2'],

    'C': np.logspace(-4, 4, 20)

}

lr = LogisticRegression(random_state=0)

grid_search = GridSearchCV(estimator = lr, param_grid = param_grid,

                          cv = 5, n_jobs = -1, verbose = True,

                          scoring = 'accuracy')



grid_search.fit(XL_train, yL_train)

best_estimator = {'Logistic Regression': grid_search.best_estimator_}

best_score = {'Logistic Regression': grid_search.best_score_}



models = {**models, **best_estimator}

models_results = {**models_results, **best_score}



print("Best Estimator: ", grid_search.best_estimator_)

print("Best Score: ", grid_search.best_score_)
best_logistic_regression = LogisticRegression(C=1.623776739188721, random_state=0)

best_logistic_regression.fit(XL_train, yL_train)
lr_predict = best_logistic_regression.predict(XL_test)
test_accuracy = {**test_accuracy, **{'Logistic Regression': accuracy_score(y_test, lr_predict)}}

test_accuracy
sns.scatterplot(x = list(models_results.keys()), y = list(models_results.values()), s = 100)
sns.scatterplot(x = list(test_accuracy.keys()), y = list(test_accuracy.values()), s = 100)
final_model = RandomForestClassifier(max_depth=12, max_features=1, min_samples_leaf=2,

                       min_samples_split=6, n_estimators=50)
final_model.fit(Xlabel, y)
predictions = final_model.predict(XTest_predict)
output = pd.DataFrame({'PassengerId': new_test['PassengerId'], 'Survived': predictions})
output.to_csv('titanic_submission.csv', index = False)

print("Your submission was successfully saved!")