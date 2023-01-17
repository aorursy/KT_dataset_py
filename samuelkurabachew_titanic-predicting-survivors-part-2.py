# Import some important libraries



import numpy as np 

import pandas as pd  
train_data = pd.read_csv("../input/titanic/train.csv")

test_data = pd.read_csv("../input/titanic/test.csv")
print("Train data size: ", train_data.shape)

print("Test data size: ", test_data.shape)
train_data.head()
test_data.head()
train_data.describe()
print('\n****************** Train info ****************\n')

train_data.info()



print('\n****************** Test info ****************\n')

test_data.info()
train_data_na = (train_data.isnull().sum() / len(train_data)) * 100

train_data_na = train_data_na.drop(train_data_na[train_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Train data Missing Ratio' :train_data_na})

missing_data
train_data.drop("Cabin", axis=1, inplace=True)
train_data["Age"].fillna(train_data["Age"].median(), inplace = True)
train_data["Embarked"].value_counts(ascending=False)
train_data["Embarked"].fillna("S", inplace=True)
train_data.info()
test_data_na = (test_data.isnull().sum() / len(test_data)) * 100

test_data_na = test_data_na.drop(test_data_na[test_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Test data Missing Ratio' :test_data_na})

missing_data
test_data.drop("Cabin", axis=1, inplace=True)

test_data["Age"].fillna(test_data["Age"].median(), inplace = True)

test_data["Fare"].fillna(test_data["Fare"].median(), inplace = True)
test_data.info()
n_train = train_data.shape[0]

n_test = test_data.shape[0]



X_train = train_data.drop("Survived", axis=1)

y_train = pd.DataFrame(train_data["Survived"], columns=["Survived"])



all_data = pd.concat((X_train, test_data))
all_data.tail()
# Drop Name column since it won't have any effect on our model (This is only for now. In the future there is important

# info we can extract).



all_data.drop('Name', axis=1, inplace=True)

all_data = pd.get_dummies(all_data)
all_data.shape
X_train = all_data[:n_train]

test_data = all_data[n_train:]
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
# Import models.

from sklearn.ensemble import RandomForestClassifier
# Import accuracy metrics since it's how our model is evaluated

from sklearn.metrics import accuracy_score
rf_clf = RandomForestClassifier(random_state=42) # Set the random state to some value to get similar result on every iteration.

rf_clf.fit(X_train, y_train)

pred_rf = rf_clf.predict(X_test)

acc_rf = accuracy_score(y_test, pred_rf)



print(acc_rf)
from sklearn.model_selection import GridSearchCV



parameters = {'n_estimators': range(10, 150, 10)}

random_forest = RandomForestClassifier(random_state=42)

gs_clf_1 = GridSearchCV(random_forest, parameters, cv=5)

gs_clf_1.fit(X_train, y_train.values.ravel())
print(gs_clf_1.best_params_)
pred_gs_clf_1 = gs_clf_1.predict(X_test)

acc_gs_clf_1 = accuracy_score(y_test, pred_gs_clf_1)



print(acc_gs_clf_1)
parameters = {'n_estimators': range(10, 150, 10), 'max_depth': range(50, 100, 5)}

random_forest = RandomForestClassifier(random_state=42)

gs_clf_2 = GridSearchCV(random_forest, parameters, cv=5)

gs_clf_2.fit(X_train, y_train.values.ravel())
print(gs_clf_2.best_params_)
pred_gs_clf_2 = gs_clf_2.predict(X_test)

acc_gs_clf_2 = accuracy_score(y_test, pred_gs_clf_2)



print(acc_gs_clf_2)
model_performance = pd.DataFrame({

    "Model": ["Random_Forest with n_estimator",

              "Random_Forest with n_estimator and max_depth",

              "Random Forest"

              ],

    "Accuracy": [acc_gs_clf_1,

                 acc_gs_clf_2,

                 acc_rf

                ]

})



model_performance.sort_values(by="Accuracy", ascending=False)
submission_predictions = gs_clf_2.predict(test_data)



submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": submission_predictions

    })



submission.to_csv("survivors.csv", index=False)

print(submission.shape)