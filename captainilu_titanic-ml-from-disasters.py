import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train.head()
train = train.drop(['PassengerId','Name','Ticket'], axis=1)
test = test.drop(['Name','Ticket'], axis=1)
train.head()
train["Embarked"] = train["Embarked"].fillna("S")
embark_dummies_train  = pd.get_dummies(train['Embarked'])
#embark_dummies_train.drop(['S'], axis=1, inplace=True)

embark_dummies_test  = pd.get_dummies(test['Embarked'])
#embark_dummies_test.drop(['S'], axis=1, inplace=True)

embark_dummies_train.head()
train = train.join(embark_dummies_train)
test = test.join(embark_dummies_test)

train.head()
train.drop(['Embarked'], axis=1, inplace=True)
test.drop(['Embarked'], axis=1, inplace=True)

train.head()
test["Fare"].fillna(test["Fare"].median(), inplace=True)

train['Fare'] = train['Fare'].astype(int)
test['Fare'] = test['Fare'].astype(int)

train.head()

fare_not_survived = train["Fare"][train["Survived"] == 0]
fare_survived = train["Fare"][train["Survived"] == 1]


average_age_train   = train["Age"].mean()
std_age_train = train["Age"].std()
count_nan_age_train = train["Age"].isnull().sum()

average_age_test   = test["Age"].mean()
std_age_test       = test["Age"].std()
count_nan_age_test = test["Age"].isnull().sum()

rand_1 = np.random.randint(average_age_train - std_age_train, average_age_train + std_age_train, size = count_nan_age_train)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

train["Age"][np.isnan(train["Age"])] = rand_1
test["Age"][np.isnan(test["Age"])] = rand_2

train['Age'] = train['Age'].astype(int)
test['Age']    = test['Age'].astype(int)
train.drop("Cabin",axis=1,inplace=True)
test.drop("Cabin",axis=1,inplace=True)
train['Family'] =  train["Parch"] + train["SibSp"]
train['Family'].loc[train['Family'] > 0] = 1
train['Family'].loc[train['Family'] == 0] = 0

test['Family'] =  test["Parch"] + test["SibSp"]
test['Family'].loc[test['Family'] > 0] = 1
test['Family'].loc[test['Family'] == 0] = 0

train = train.drop(['SibSp','Parch'], axis=1)
test = test.drop(['SibSp','Parch'], axis=1)
def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex
    
train['Person'] = train[['Age','Sex']].apply(get_person,axis=1)
test['Person'] = test[['Age','Sex']].apply(get_person,axis=1)

train.drop(['Sex'],axis=1,inplace=True)
test.drop(['Sex'],axis=1,inplace=True)

person_dummies_train  = pd.get_dummies(train['Person'])
person_dummies_train.columns = ['Child','Female','Male']
#person_dummies_train.drop(['Male'], axis=1, inplace=True)

person_dummies_test  = pd.get_dummies(test['Person'])
person_dummies_test.columns = ['Child','Female','Male']
#person_dummies_test.drop(['Male'], axis=1, inplace=True)

train = train.join(person_dummies_train)
test = test.join(person_dummies_test)

train.drop(['Person'],axis=1,inplace=True)
test.drop(['Person'],axis=1,inplace=True)
train.head()
pclass_dummies_train  = pd.get_dummies(train['Pclass'])
pclass_dummies_train.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_train.drop(['Class_3'], axis=1, inplace=True)

pclass_dummies_test  = pd.get_dummies(test['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)

train.drop(['Pclass'],axis=1,inplace=True)
test.drop(['Pclass'],axis=1,inplace=True)

train = train.join(pclass_dummies_train)
test = test.join(pclass_dummies_test)
X_train = train.drop("Survived",axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId",axis=1).copy()
X_train.head()
import sys
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
scaler.fit(X_test)
X_test = scaler.transform(X_test)
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)
# Support Vector Machines

# svc = SVC()

# svc.fit(X_train, Y_train)

# Y_pred = svc.predict(X_test)

# svc.score(X_train, Y_train)
# # Random Forests

# random_forest = RandomForestClassifier(n_estimators=100)

# random_forest.fit(X_train, Y_train)

# Y_pred = random_forest.predict(X_test)

# random_forest.score(X_train, Y_train)
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

knn.score(X_train, Y_train)
# Gaussian Naive Bayes

# gaussian = GaussianNB()

# gaussian.fit(X_train, Y_train)

# Y_pred = gaussian.predict(X_test)

# gaussian.score(X_train, Y_train)
#from sklearn.neural_network import MLPClassifier
# mlp = MLPClassifier(hidden_layer_sizes=(1000, 1000, 1000))
# mlp.fit(X_train,Y_train)
# Y_pred = mlp.predict(X_test)
# mlp.score(X_train, Y_train)
from sklearn.model_selection import GridSearchCV
n_neighbors = [6,7,8,9,10,11,12,14,16,18,20,22]
algorithm = ['auto']
weights = ['uniform', 'distance']
leaf_size = list(range(1,50,5))
hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 
               'n_neighbors': n_neighbors}
gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True, 
                cv=10, scoring = "roc_auc")
gd.fit(X_train, Y_train)
print(gd.best_score_)
print(gd.best_estimator_)
print(gd.best_score_)
gd.best_estimator_.fit(X_train, Y_train)
Y_pred = gd.best_estimator_.predict(X_test)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)
