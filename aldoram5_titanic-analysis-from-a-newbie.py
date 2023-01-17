# Needed imports



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Load the models

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

# Get info about our test set

train.info()
train = train.drop('Cabin',axis=1)

test = test.drop('Cabin',axis=1)
train["Ticket"].value_counts()
train = train.drop('Ticket',axis=1)

test = test.drop('Ticket',axis=1)

# To reduce space we'll check Name here as well

train["Name"].value_counts()
train = train.drop('Name',axis=1)

test = test.drop('Name',axis=1)

# To reduce space we'll check Sex here as well

train["Sex"].value_counts()
# Simple Label encoder would work

from sklearn.preprocessing import LabelEncoder



encoder = LabelEncoder()

train["Sex"] = encoder.fit_transform(train["Sex"])

test["Sex"] = encoder.fit_transform(test["Sex"])

# We confirm that the Sex feature got encoded

train.head()
train["Embarked"].value_counts()
# We have to fill the 2 null values or else it'll throw an exception

# We are going to fill them with the most common values

encoder2 = LabelEncoder()

train["Embarked"].fillna(train["Embarked"].mode()[0], inplace=True)

test["Embarked"].fillna(test["Embarked"].mode()[0], inplace=True)

train["Embarked"] = encoder2.fit_transform(train["Embarked"])

test["Embarked"] = encoder2.fit_transform(test["Embarked"])

# We confirm that the Embarked feature got encoded

train.head()
train.info()
# We remove the passenger id since that won't help our model

train = train.drop('PassengerId',axis=1)

corr_matrix = train.corr()

corr_matrix["Survived"].sort_values(ascending=False)
from pandas.plotting import scatter_matrix



attributes = ["Survived", "Age", "SibSp", "Parch"]

scatter_matrix(train[attributes], figsize=(16, 16))

# We'll get more insights about age first to define appropiate ranges



train["Age"].value_counts()
train.loc[train['Age'] <= 17, 'Age'] = 0

train.loc[(train['Age'] > 17) & (train['Age'] <= 36), 'Age'] = 1

train.loc[train['Age'] > 36, 'Age'] = 2



test.loc[test['Age'] <= 17, 'Age'] = 0

test.loc[(test['Age'] > 17) & (test['Age'] <= 36), 'Age'] = 1

test.loc[test['Age'] > 36, 'Age'] = 2
# Please note that the eval function is not optimized for tiny data sets

# I used it because I think is easier to understand what exactly we are

# doing to the data

train["HasRelative"] = np.where(train.eval("SibSp > 0 or Parch > 0"), 1, 0)

test["HasRelative"] = np.where(test.eval("SibSp > 0 or Parch > 0"), 1, 0)
corr_matrix = train.corr()

corr_matrix["Survived"].sort_values(ascending=False)
train["Age"].value_counts()
train = train.drop('Parch',axis=1)

train = train.drop('SibSp',axis=1)

train.info()
train['Age'].fillna(1, inplace=True)

train.info()
X_train = train.drop('Survived',axis=1)

Y_train = train['Survived']



from sklearn.ensemble import RandomForestClassifier



random_forest = RandomForestClassifier(n_estimators=200)

random_forest.fit(X_train, Y_train)

test.info()
test = test.drop('Parch',axis=1)

test = test.drop('SibSp',axis=1)

test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)

test['Age'].fillna(test['Age'].mode()[0], inplace=True)

test.info()
predictions = random_forest.predict(test.drop('PassengerId',axis=1))

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions

    })

submission.to_csv('./predictions.csv', index=False)
from sklearn.model_selection import GridSearchCV

param_grid = {"n_estimators": [ 10, 20, 30, 50, 100, 200],

              "max_features": [3, 4, 5, 6],

              "max_depth": [ 6, 9, 12, 15],

              "bootstrap": [True, False]}    



optimal_forest = RandomForestClassifier()

# run grid search

grid_search = GridSearchCV(optimal_forest, param_grid=param_grid,scoring="average_precision")

grid_search.fit(X_train, Y_train)

print(grid_search.best_params_)

print(grid_search)
param_grid = {"n_estimators": [ 25, 28, 30, 33, 36, 40],

              "max_features": [2, 3, 4, 5],

              "max_depth": [ 4, 5, 6, 7],

              "bootstrap": [False]}

optimal_forest = RandomForestClassifier()

# run grid search

grid_search_rfo = GridSearchCV(optimal_forest, param_grid=param_grid,scoring="average_precision")

grid_search_rfo.fit(X_train, Y_train)

print(grid_search_rfo.best_params_)

print(grid_search_rfo)
predictions_rfo = grid_search_rfo.predict(test.drop('PassengerId',axis=1))

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions_rfo

    })

submission.to_csv('./predictions_rf_optimized.csv', index=False)
from sklearn.svm import LinearSVC

param_grid = {"C": [ 1, 10, 30, 50, 100],

              "loss": ["hinge", "squared_hinge"]}

linear_svc = LinearSVC()

grid_search_lsvc = GridSearchCV(linear_svc, param_grid=param_grid,scoring="average_precision")

grid_search_lsvc.fit(X_train, Y_train)

print(grid_search_lsvc.best_params_)

print(grid_search_lsvc)
predictions_lsvc = grid_search_lsvc.predict(test.drop('PassengerId',axis=1))

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions_lsvc

    })

submission.to_csv('./predictions_lsvc_optimized.csv', index=False)
from sklearn.svm import SVC

param_grid = {"kernel":["rbf"],

              "gamma": [ 0.1, 1, 3, 5],

              "C": [ 0.001 , 0.1, 1, 10, 100]}

rbf_svc = SVC()

grid_search_rbf_svc = GridSearchCV(rbf_svc, param_grid=param_grid,scoring="average_precision")

grid_search_rbf_svc.fit(X_train, Y_train)

print(grid_search_rbf_svc.best_params_)

print(grid_search_rbf_svc)
#Another grid search to find even better parameters

param_grid = {"kernel":["rbf"],

              "gamma": [ 0.01, 0.05, 0.1, 0.15, 0.5],

              "C": [ 8 , 10, 15, 20, 40]}

rbf_svc = SVC()

grid_search_rbf_svc = GridSearchCV(rbf_svc, param_grid=param_grid,scoring="average_precision")

grid_search_rbf_svc.fit(X_train, Y_train)

print(grid_search_rbf_svc.best_params_)

print(grid_search_rbf_svc)
#Yet another grid search to find even better parameters

param_grid = {"kernel":["rbf"],

              "gamma": [ 0.001, 0.005, 0.01, 0.05],

              "C": [ 20, 40, 50, 60, 70]}

rbf_svc = SVC()

grid_search_rbf_svc = GridSearchCV(rbf_svc, param_grid=param_grid,scoring="average_precision")

grid_search_rbf_svc.fit(X_train, Y_train)

print(grid_search_rbf_svc.best_params_)

print(grid_search_rbf_svc)
predictions_rbf_svc = grid_search_rbf_svc.predict(test.drop('PassengerId',axis=1))

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions_rbf_svc

    })

submission.to_csv('./predictions_rbf_svc.csv', index=False)
from sklearn.linear_model import SGDClassifier



param_grid = {"loss":["hinge","modified_huber"],

              "penalty":["l2","l1","elasticnet"]}

sgdc = SGDClassifier()

grid_search_sgdc = GridSearchCV(sgdc, param_grid=param_grid,scoring="average_precision")

grid_search_sgdc.fit(X_train, Y_train)

print(grid_search_sgdc.best_params_)

print(grid_search_sgdc)
predictions_sgdc = grid_search_sgdc.predict(test.drop('PassengerId',axis=1))

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions_sgdc

    })

submission.to_csv('./predictions_sgdc.csv', index=False)
from sklearn.neighbors import KNeighborsClassifier

param_grid = {"weights":["uniform","distance"],

              "n_neighbors":[5,7,9,12,15,18]}

nbrs = KNeighborsClassifier()

grid_search_nbrs = GridSearchCV(nbrs, param_grid=param_grid,scoring="average_precision")

grid_search_nbrs.fit(X_train, Y_train)

print(grid_search_nbrs.best_params_)

print(grid_search_nbrs)
from sklearn.ensemble import VotingClassifier



nbrs = KNeighborsClassifier(18, weights='distance')

sgdc = SGDClassifier(loss='hinge', penalty= 'l1')

rbf_svc = SVC(kernel="rbf",C=10, gamma= 0.1)

linear_svc = LinearSVC(loss="hinge",C=1)

optimal_forest = RandomForestClassifier(bootstrap= False, max_depth=6, max_features= 4, n_estimators= 28)

voting_clf = VotingClassifier(

            estimators=[('rf', optimal_forest), ('lsvc', linear_svc), ('rbfsvc', rbf_svc), ('sgdc', sgdc), ('nbrs',nbrs)],

            voting='hard'

        )

voting_clf.fit(X_train, Y_train)



predictions = voting_clf.predict(test.drop('PassengerId',axis=1))

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions

    })

submission.to_csv('./predictions_voting.csv', index=False)
sgdc = SGDClassifier(loss='hinge', penalty= 'l1')

rbf_svc = SVC(kernel="rbf",C=10, gamma= 0.1)

optimal_forest = RandomForestClassifier(bootstrap= False, max_depth=6, max_features= 4, n_estimators= 28)

voting_clf = VotingClassifier(

            estimators=[('rf', optimal_forest), ('rbfsvc', rbf_svc), ('sgdc', sgdc)],

            voting='hard'

        )

voting_clf.fit(X_train, Y_train)

#Save predictions

predictions = voting_clf.predict(test.drop('PassengerId',axis=1))

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions

    })

submission.to_csv('./predictions_voting_2.csv', index=False)
sgdc = SGDClassifier(loss='modified_huber', penalty= 'l1')

rbf_svc = SVC(kernel="rbf",C=10, gamma= 0.1, probability=True)

optimal_forest = RandomForestClassifier(bootstrap= False, max_depth=6, max_features= 4, n_estimators= 28)

voting_clf = VotingClassifier(

            estimators=[('rf', optimal_forest), ('rbfsvc', rbf_svc), ('sgdc', sgdc)],

            voting='soft'

        )

voting_clf.fit(X_train, Y_train)

#Save predictions

predictions = voting_clf.predict(test.drop('PassengerId',axis=1))

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions

    })

submission.to_csv('./predictions_voting_2_soft.csv', index=False)
# LinearSVC

param_grid = {"C": [ 30, 50, 100],

              "loss": ["hinge", "squared_hinge"]}

linear_svc = LinearSVC()

grid_search_lsvc = GridSearchCV(linear_svc, param_grid=param_grid,scoring="accuracy")

grid_search_lsvc.fit(X_train, Y_train)

print(grid_search_lsvc.best_params_)

predictions_lsvc = grid_search_lsvc.predict(test.drop('PassengerId',axis=1))

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions_lsvc

    })

linear_svc = LinearSVC()

submission.to_csv('./predictions_lsvc_accuracy.csv', index=False)

grid_search_lsvc = GridSearchCV(linear_svc, param_grid=param_grid,scoring="f1")

grid_search_lsvc.fit(X_train, Y_train)

print(grid_search_lsvc.best_params_)

predictions_lsvc = grid_search_lsvc.predict(test.drop('PassengerId',axis=1))

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions_lsvc

    })

submission.to_csv('./predictions_lsvc_f1.csv', index=False)
#Random Forest

param_grid = {"n_estimators": [ 10, 20, 35, 50, 100],

              "max_features": [3, 4, 5, 6],

              "max_depth": [ 6, 9, 12, 15],

              "bootstrap": [True, False]}    



optimal_forest = RandomForestClassifier()

grid_search = GridSearchCV(optimal_forest, param_grid=param_grid,scoring="accuracy")

grid_search.fit(X_train, Y_train)

print(grid_search.best_params_)

predictions_rfo = grid_search_rfo.predict(test.drop('PassengerId',axis=1))

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions_rfo

    })

submission.to_csv('./predictions_rf_accuracy.csv', index=False)



optimal_forest = RandomForestClassifier()

grid_search = GridSearchCV(optimal_forest, param_grid=param_grid,scoring="f1")

grid_search.fit(X_train, Y_train)

print(grid_search.best_params_)

predictions_rfo = grid_search_rfo.predict(test.drop('PassengerId',axis=1))

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions_rfo

    })

submission.to_csv('./predictions_rf_f1.csv', index=False)
#K-Nearest

param_grid = {"weights":["uniform","distance"],

              "n_neighbors":[4,5,6,7,8,10]}

nbrs = KNeighborsClassifier()

grid_search_nbrs = GridSearchCV(nbrs, param_grid=param_grid,scoring="accuracy")

grid_search_nbrs.fit(X_train, Y_train)

print(grid_search_nbrs.best_params_)

predictions_nbrs = grid_search_nbrs.predict(test.drop('PassengerId',axis=1))

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions_nbrs

    })

submission.to_csv('./predictions_knbrs_accuracy.csv', index=False)

nbrs = KNeighborsClassifier()

grid_search_nbrs = GridSearchCV(nbrs, param_grid=param_grid,scoring="f1")

grid_search_nbrs.fit(X_train, Y_train)

print(grid_search_nbrs.best_params_)

grid_search_nbrs = grid_search_nbrs.predict(test.drop('PassengerId',axis=1))

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": grid_search_nbrs

    })

submission.to_csv('./predictions_knbrs_f1.csv', index=False)
param_grid = {"loss":["hinge","modified_huber"],

              "penalty":["l2","l1","elasticnet"]}

sgdc = SGDClassifier()

grid_search_sgdc = GridSearchCV(sgdc, param_grid=param_grid,scoring="accuracy")

grid_search_sgdc.fit(X_train, Y_train)

print(grid_search_sgdc.best_params_)

predictions_sgdc = grid_search_sgdc.predict(test.drop('PassengerId',axis=1))

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions_sgdc

    })

submission.to_csv('./predictions_sgdc_accuracy.csv', index=False)

sgdc = SGDClassifier()

grid_search_sgdc = GridSearchCV(sgdc, param_grid=param_grid,scoring="f1")

grid_search_sgdc.fit(X_train, Y_train)

print(grid_search_sgdc.best_params_)

predictions_sgdc = grid_search_sgdc.predict(test.drop('PassengerId',axis=1))

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions_sgdc

    })

submission.to_csv('./predictions_sgdc_f1.csv', index=False)
param_grid = {"kernel":["rbf"],

              "gamma": [ 0.001, 0.005,0.01, 0.10],

              "C": [  55,60, 62, 65, 67]}

rbf_svc = SVC()

grid_search_rbf_svc = GridSearchCV(rbf_svc, param_grid=param_grid,scoring="accuracy")

grid_search_rbf_svc.fit(X_train, Y_train)

print(grid_search_rbf_svc.best_params_)

predictions_rbf_svc = grid_search_rbf_svc.predict(test.drop('PassengerId',axis=1))

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions_rbf_svc

    })

submission.to_csv('./predictions_rbf_svc_accuracy.csv', index=False)

rbf_svc = SVC()

grid_search_rbf_svc = GridSearchCV(rbf_svc, param_grid=param_grid,scoring="f1")

grid_search_rbf_svc.fit(X_train, Y_train)

print(grid_search_rbf_svc.best_params_)

predictions_rbf_svc = grid_search_rbf_svc.predict(test.drop('PassengerId',axis=1))

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions_rbf_svc

    })

submission.to_csv('./predictions_rbf_svc_f1.csv', index=False)
nbrs_1 = KNeighborsClassifier(18, weights='distance')

nbrs_2 = KNeighborsClassifier(8, weights='distance')

nbrs_3 = KNeighborsClassifier(4, weights='distance')

sgdc_1 = SGDClassifier(loss='hinge', penalty= 'l1')

sgdc_2 = SGDClassifier(loss='modified_huber', penalty= 'l1')

sgdc_3 = SGDClassifier(loss='modified_huber', penalty= 'elasticnet')

rbf_svc_1 = SVC(kernel="rbf",C=10, gamma= 0.1)

rbf_svc_2 = SVC(kernel="rbf",C=65, gamma= 0.01)

rbf_svc_3 = SVC(kernel="rbf",C=67, gamma= 0.01)

linear_svc_1 = LinearSVC(loss="hinge",C=1)

linear_svc_2 = LinearSVC(loss="squared_hinge",C=30)

linear_svc_3 = LinearSVC(loss="hinge",C=50)

optimal_forest_1 = RandomForestClassifier(bootstrap= False, max_depth=6, max_features= 4, n_estimators= 28)

optimal_forest_2 = RandomForestClassifier(bootstrap= True, max_depth=9, max_features= 5, n_estimators= 35)

optimal_forest_3 = RandomForestClassifier(bootstrap= True, max_depth=6, max_features= 5, n_estimators= 20)

voting_clf = VotingClassifier(

            estimators=[('rf_1', optimal_forest_1), 

                        ('rf_2', optimal_forest_2), 

                        ('rf_3', optimal_forest_3), 

                        ('lsvc_1', linear_svc_1),

                        ('lsvc_2', linear_svc_2),

                        ('lsvc_3', linear_svc_3),

                        ('rbfsvc_1', rbf_svc_1),

                        ('rbfsvc_2', rbf_svc_2),

                        ('rbfsvc_3', rbf_svc_3),

                        ('sgdc_1', sgdc_1),

                        ('sgdc_2', sgdc_2),

                        ('sgdc_3', sgdc_3),

                        ('nbrs_1',nbrs_1),

                        ('nbrs_2',nbrs_2),

                        ('nbrs_3',nbrs_3)],

            voting='hard'

        )

voting_clf.fit(X_train, Y_train)



predictions = voting_clf.predict(test.drop('PassengerId',axis=1))

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions

    })

submission.to_csv('./predictions_voting_many.csv', index=False)
#removed some SDG Classifiers

nbrs_1 = KNeighborsClassifier(18, weights='distance')

nbrs_2 = KNeighborsClassifier(8, weights='distance')

nbrs_3 = KNeighborsClassifier(4, weights='distance')

sgdc_1 = SGDClassifier(loss='hinge', penalty= 'l1')

rbf_svc_1 = SVC(kernel="rbf",C=10, gamma= 0.1)

rbf_svc_2 = SVC(kernel="rbf",C=65, gamma= 0.01)

rbf_svc_3 = SVC(kernel="rbf",C=67, gamma= 0.01)

linear_svc_1 = LinearSVC(loss="hinge",C=1)

linear_svc_2 = LinearSVC(loss="squared_hinge",C=30)

linear_svc_3 = LinearSVC(loss="hinge",C=50)

optimal_forest_1 = RandomForestClassifier(bootstrap= False, max_depth=6, max_features= 4, n_estimators= 28)

optimal_forest_2 = RandomForestClassifier(bootstrap= True, max_depth=9, max_features= 5, n_estimators= 35)

optimal_forest_3 = RandomForestClassifier(bootstrap= True, max_depth=6, max_features= 5, n_estimators= 20)

voting_clf = VotingClassifier(

            estimators=[('rf_1', optimal_forest_1), 

                        ('rf_2', optimal_forest_2), 

                        ('rf_3', optimal_forest_3), 

                        ('lsvc_1', linear_svc_1),

                        ('lsvc_2', linear_svc_2),

                        ('lsvc_3', linear_svc_3),

                        ('rbfsvc_1', rbf_svc_1),

                        ('rbfsvc_2', rbf_svc_2),

                        ('rbfsvc_3', rbf_svc_3),

                        ('sgdc_1', sgdc_1),

                        ('nbrs_1',nbrs_1),

                        ('nbrs_2',nbrs_2),

                        ('nbrs_3',nbrs_3)],

            voting='hard'

        )

voting_clf.fit(X_train, Y_train)



predictions = voting_clf.predict(test.drop('PassengerId',axis=1))

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions

    })

submission.to_csv('./predictions_voting_many_one_sdg.csv', index=False)