import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

%matplotlib inline

import seaborn as sns # data visualization

sns.set()
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.shape
train.describe()
train.describe(include = ['O'])
train.info
train.isnull().sum()
test.shape
test.describe()
test.describe(include = ['O'])
test.info
test.isnull().sum()
survived = train[train.Survived == 1]

not_survived = train[train.Survived == 0]

print('Survived: %i (%.1f%%)' %(len(survived), float(100*len(survived)/(len(survived)+len(not_survived)))))

print('Did not Survive: %i (%.1f%%)' %(len(not_survived), float(100*len(not_survived)/(len(survived)+len(not_survived)))))
print(train.Pclass.value_counts())

# Gives the number of people in each class



print(train.groupby('Pclass').Survived.value_counts())

# Counts the number of people in each class who survived and did not survive



print(train[['Pclass', 'Survived']].groupby('Pclass', as_index = False).mean())

# Counts the proportion of people in each class that survived (1 = Survived and 0 = Did Not Survive, so mean = proportion who survived)
#train.groupby('Pclass').Survived.mean().plot(kind = 'bar')

sns.barplot(x = 'Pclass', y = 'Survived', data = train)
print(train.Sex.value_counts())

# Displays the number of each sex



print(train.groupby('Sex').Survived.value_counts())

# Displays the number of each sex that survived and did not survive



print(train.groupby('Sex').Survived.mean())

# Displays the proportion of each sex that survived or did not survive
#train.groupby('Sex').Survived.mean().plot(kind = 'bar')

sns.barplot(x = 'Sex', y = 'Survived', data = train)
train.Age.describe()
age = train.Age.dropna()

sns.distplot(age, bins = 25, kde = False)
train['AgeBand'] = np.where(train.Age <= 16, 1, 

                            np.where((train.Age > 16) & (train.Age <= 32), 2, 

                                     np.where((train.Age > 32) & (train.Age <= 48), 3, 

                                              np.where((train.Age > 48) & (train.Age <= 64), 4, 

                                                      np.where((train.Age > 64) & (train.Age <= 80), 5, False)))))
print(train.AgeBand.value_counts())

# Displays the number of each sex



print(train.groupby('AgeBand').Survived.value_counts())

# Displays the number of each sex that survived and did not survive



print(train.groupby('AgeBand').Survived.mean())

# Displays the proportion of each sex that survived or did not survive
train.groupby('AgeBand').Age.describe()
#train.groupby('AgeBand').Survived.mean().plot(kind = 'bar')

sns.barplot(x = 'AgeBand', y = 'Survived', data = train)
train.SibSp.describe()
print(train.SibSp.value_counts())

# Displays the number of each sex



print(train.groupby('SibSp').Survived.value_counts())

# Displays the number of each sex that survived and did not survive



print(train.groupby('SibSp').Survived.mean())

# Displays the proportion of each sex that survived or did not survive
#train.groupby('SibSp').Survived.mean().plot(kind = 'bar')

sns.barplot(x = 'SibSp', y = 'Survived', data = train)
train.Parch.describe()
print(train.Parch.value_counts())

# Displays the number of each sex



print(train.groupby('Parch').Survived.value_counts())

# Displays the number of each sex that survived and did not survive



print(train.groupby('Parch').Survived.mean())

# Displays the proportion of each sex that survived or did not survive
#train.groupby('Parch').Survived.mean().plot(kind = 'bar')

sns.barplot(x = 'Parch', y = 'Survived', data = train)
train.Embarked.describe(include = ['O'])
print(train.Embarked.value_counts())

# Displays the number of each sex



print(train.groupby('Embarked').Survived.value_counts())

# Displays the number of each sex that survived and did not survive



print(train.groupby('Embarked').Survived.mean())

# Displays the proportion of each sex that survived or did not survive
#train.groupby('Embarked').Survived.mean().plot(kind = 'bar')

sns.barplot(x = 'Embarked', y = 'Survived', data = train)
train['FamilySize'] = train.SibSp + train.Parch
train.FamilySize.describe()
print(train.FamilySize.value_counts())

# Displays the number of each sex



print(train.groupby('FamilySize').Survived.value_counts())

# Displays the number of each sex that survived and did not survive



print(train.groupby('FamilySize').Survived.mean())

# Displays the proportion of each sex that survived or did not survive
#train.groupby('FamilySize').Survived.mean().plot(kind = 'bar')

sns.barplot(x = 'FamilySize', y = 'Survived', data = train)
train.Fare.describe()
sns.distplot(train.Fare, bins = 15, kde = False)
train['logFare'] = np.where(train.Fare != 0, np.log(train.Fare), train.Fare)
sns.distplot(train.logFare, bins = 15)
from sklearn.impute import SimpleImputer



#Loads the data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



#Creates the FamilySize and logFate variables

train['FamilySize'] = train.SibSp + train.Parch

train['logFare'] = np.where(train.Fare != 0, np.log(train.Fare), train.Fare)

test['FamilySize'] = test.SibSp + test.Parch

test['logFare'] = np.where(test.Fare != 0, np.log(test.Fare), test.Fare)



#Puts the features that should have no effect on survival in a list

cols_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']



#Drops the aforementioned features

train = train.drop(cols_to_drop, axis=1)

X_test = test.drop(cols_to_drop, axis=1)



#Creates boolean variables for categorical features

train_data = pd.get_dummies(train)

X_test = pd.get_dummies(X_test)



#Creates the training feature matrix and the training target vector

X_train = train_data.drop('Survived', axis=1)

y_train = train_data.Survived



#Replaces missing values with averages

my_imputer = SimpleImputer()

X_train = my_imputer.fit_transform(X_train)

X_test = my_imputer.fit_transform(X_test)
from xgboost import XGBClassifier

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import StratifiedKFold



#Splits the training data up for use to score model accuracy and model selection.

train_X, test_X, train_y, test_y = train_test_split(X_train, y_train, train_size = 0.7, test_size = 0.3, random_state = 0)
#Defines a function to return the best parameters for the SVC model

def svc_param_selection(X, y, nfolds):

    Cs = [0.001, 0.01, 0.1, 1, 10]

    gammas = [0.001, 0.01, 0.1, 1]

    param_grid = {'C': Cs, 'gamma' : gammas}

    grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=nfolds)

    grid_search.fit(X, y)

    grid_search.best_params_

    return grid_search.best_params_



svc_param_selection(X_train, y_train, 5)
my_svc_model = svm.SVC(C = 0.01, kernel ='linear', gamma = 0.001)

my_svc_model.fit(X_train, y_train)



kfold = KFold(n_splits=10, random_state=7)

results = cross_val_score(my_svc_model, X_train, y_train, cv=kfold)

print("SVM Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# This is the number of trees starting at 200 and going to 2000 in increments of 10

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Sets a maximum number of levels in each tree to avoid overfitting

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Sets a minimum number of observations to allow a node to make a split

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}





my_rf_model = RandomForestClassifier()

my_rf_model = RandomizedSearchCV(estimator = my_rf_model, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=7, n_jobs = -1)

my_rf_model.fit(X_train, y_train)

my_rf_model.best_params_
my_forest_model = RandomForestClassifier(n_estimators = 200,

                                         min_samples_split = 5,

                                         min_samples_leaf = 4,

                                         max_features = 'auto',

                                         max_depth = 80,

                                         bootstrap = True)

my_forest_model.fit(X_train, y_train)



kfold = KFold(n_splits=10, random_state=7)

results = cross_val_score(my_forest_model, X_train, y_train, cv=kfold)

print("Random Forest Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#Creates a list of possible ks from 1 to 30

k_range = list(range(1, 31))

#Creates a list of 2 possible weighting options

weight_options = ['uniform', 'distance']

#Creates a dictionary containing the k_range and weight_options that is the parameter grid

param_grid = {'n_neighbors': k_range, 'weights': weight_options}



my_knn_model = KNeighborsClassifier(algorithm = 'brute')

clf = GridSearchCV(my_knn_model, param_grid, cv=5)

clf.fit(X_train, y_train)

print(clf.best_params_)
my_knn_model = KNeighborsClassifier(n_neighbors = 13, weights = 'distance')

my_knn_model.fit(X_train, y_train)



kfold = KFold(n_splits=10, random_state=7)

results = cross_val_score(my_knn_model, X_train, y_train, cv=kfold)

print("Knn Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
my_gnb_model = GaussianNB()

my_gnb_model.fit(X_train, y_train)



kfold = KFold(n_splits=10, random_state=7)

results = cross_val_score(my_gnb_model, X_train, y_train, cv=kfold)

print("GNB Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
my_logit_model = LogisticRegression(solver = 'liblinear')

my_logit_model.fit(X_train, y_train)



kfold = KFold(n_splits=10, random_state=7)

results = cross_val_score(my_logit_model, X_train, y_train, cv=kfold)

print("Logistic Regression Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
my_xgb_model = XGBClassifier()



parameters = {'nthread':[4],

              'objective':['binary:logistic'],

              'learning_rate': [0.03, 0.04, 0.05, 0.06, 0.07, 0.08], 

              'max_depth': [5, 6, 7, 8],

              'min_child_weight': [9, 10, 11, 12, 13],

              'silent': [1],

              'subsample': [0.8],

              'colsample_bytree': [0.7],

              'n_estimators': [10], #kept small so the grid search doesn't take too long

              'missing':[-999],

              'seed': [7]}



clf = GridSearchCV(my_xgb_model, parameters, n_jobs=5, 

                   cv=StratifiedKFold(n_splits=5), 

                   scoring='roc_auc',

                   verbose=2, refit=True)



clf.fit(X_train, y_train)



print(clf.best_params_)
my_xgb_model = XGBClassifier(colsample_bytree = 0.7, 

                             learning_rate = 0.07, 

                             max_depth = 5, 

                             min_child_weight = 9, 

                             missing = -999, 

                             n_estimators = 125, 

                             nthread = 4, 

                             objective = 'binary:logistic', 

                             seed = 7, 

                             silent = 1, 

                             subsample = 0.8)

my_xgb_model.fit(X_train, y_train, early_stopping_rounds = 5, eval_set = [(test_X, test_y)], verbose = False)



kfold = KFold(n_splits=10, random_state=7)

results = cross_val_score(my_xgb_model, X_train, y_train, cv=kfold)

print("XGBTree Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
import numpy as np

import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train['FamilySize'] = train.SibSp + train.Parch

train['logFare'] = np.where(train.Fare != 0, np.log(train.Fare), train.Fare)



cols_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']



train.Pclass = train.Pclass.astype(str)

train = train.drop(cols_to_drop, axis=1)

test.Pclass = test.Pclass.astype(str)

X_test = test.drop(cols_to_drop, axis=1).copy()



X_test['FamilySize'] = X_test.SibSp + X_test.Parch

X_test['logFare'] = np.where(X_test.Fare != 0, np.log(X_test.Fare), X_test.Fare)



train_data = pd.get_dummies(train)

X_test = pd.get_dummies(X_test)



X_train = train_data.drop('Survived', axis=1)

y_train = train_data.Survived



my_imputer = SimpleImputer()

X_train = my_imputer.fit_transform(X_train)

X_test = my_imputer.fit_transform(X_test)



train_X, test_X, train_y, test_y = train_test_split(X_train, y_train, train_size = 0.7, test_size = 0.25, random_state = 0)



my_xgb_model = XGBClassifier(colsample_bytree = 0.7, 

                             learning_rate = 0.07, 

                             max_depth = 5, 

                             min_child_weight = 9, 

                             missing = -999, 

                             n_estimators = 125, 

                             nthread = 4, 

                             objective = 'binary:logistic', 

                             seed = 1337, 

                             silent = 1, 

                             subsample = 0.8)

my_xgb_model.fit(X_train, y_train, early_stopping_rounds = 5, eval_set = [(test_X, test_y)], verbose = False)



my_predictions = my_xgb_model.predict(X_test)



jcleme_submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": my_predictions})



jcleme_submission.to_csv('jcleme_xgb_submission.csv', index = False)