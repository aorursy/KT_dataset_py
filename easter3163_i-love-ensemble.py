import pandas as pd

train = pd.read_csv("../input/train.csv", index_col = "PassengerId")

test = pd.read_csv("../input/test.csv", index_col="PassengerId")

train.head()
train.loc[train["Sex"] == "male", "Sex_encode"] = 0

train.loc[train["Sex"] == "female", "Sex_encode"] = 1



train[["Sex", "Sex_encode"]].head()
test.loc[test["Sex"] == "male", "Sex_encode"] = 0

test.loc[test["Sex"] == "female", "Sex_encode"] = 1



test[["Sex", "Sex_encode"]].head()
train["Fare_fillin"] = train["Fare"]

test["Fare_fillin"] = test["Fare"]
test.loc[test["Fare"].isnull(), "Fare_fillin"] = 0
train["Fare_fillin"] = train["Fare_fillin"]

test["Fare_fillin"] = test["Fare_fillin"]

train["Fare_fillin"].head()
train["Embarked"].fillna("S")
train["Embarked_C"] = False

train.loc[train["Embarked"]=='C', "Embarked_C"] = True

train["Embarked_S"] = False

train.loc[train["Embarked"]=='S', "Embarked_S"] = True

train["Embarked_Q"] = False

train.loc[train["Embarked"]=='Q', "Embarked_Q"] = True

train[["Embarked", "Embarked_C", "Embarked_S", "Embarked_Q"]].head()
test["Embarked_C"] = False

test.loc[test["Embarked"]=='C', "Embarked_C"] = True

test["Embarked_S"] = False

test.loc[test["Embarked"]=='S', "Embarked_S"] = True

test["Embarked_Q"] = False

test.loc[test["Embarked"]=='Q', "Embarked_Q"] = True

test[["Embarked", "Embarked_C", "Embarked_S", "Embarked_Q"]].head()
train["Age"].fillna(train["Age"].mean(), inplace=True)

test["Age"].fillna(test["Age"].mean(), inplace=True)
train["Child"] = False

train.loc[train["Age"] < 16, "Child"] = True

train[["Age", "Child"]].head(10)
test["Child"] = False

test.loc[test["Age"] < 16, "Child"] = True

test[["Age", "Child"]].head(10)
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1

test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
test[["FamilySize"]].head()
train["Single"] = False

train.loc[train["FamilySize"]==1, "Single"] = True

train["Nuclear"] = False

train.loc[(train["FamilySize"]>1)&(train["FamilySize"]<5), "Nuclear"] = True

train["Big"] = False

train.loc[train["FamilySize"] >=5, "Big"] = True

train[["FamilySize", "Single", "Nuclear", "Big"]].head(10)
test["Single"] = False

test.loc[test["FamilySize"]==1, "Single"] = True

test["Nuclear"] = False

test.loc[(test["FamilySize"]>1)&(test["FamilySize"]<5), "Nuclear"] = True

test["Big"] = False

test.loc[test["FamilySize"] >=5, "Big"] = True

test[["FamilySize", "Single", "Nuclear", "Big"]].head(10)
train["Master"] = False

train.loc[train["Name"].str.contains("Master"), "Master"] = True

train[["Name", "Master"]].head(10)
test["Master"] = False

test.loc[test["Name"].str.contains("Master"), "Master"] = True

test[["Name", "Master"]].head(10)
# feature_names = ["Pclass", "Sex_encode", "Fare_fillin", "Embarked_C", "Embarked_S", "Embarked_Q", "Child", "Single", "Nuclear", "Big", "Master"]

feature_names = ["Pclass", "Sex_encode","Embarked_C", "Embarked_S", "Child", "Single", "Nuclear", "Master"]

feature_names
label_name = "Survived"

label_name
from sklearn.model_selection import train_test_split

random_seed=0

Y = train[label_name]

X = train[feature_names]

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=random_seed)

X.head()
X_test = test[feature_names]
# Modelling Algorithms

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier

# Modelling Helpers

from sklearn.preprocessing import Imputer , Normalizer , scale

from sklearn.feature_selection import RFECV

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

import catboost
rforest_model = RandomForestClassifier(n_estimators=100)

svc = SVC()

knn = KNeighborsClassifier(n_neighbors = 3)

gbmodel = GradientBoostingClassifier(max_depth=12)

gnb = GaussianNB()

LR = LogisticRegression()

mlp = MLPClassifier(solver='lbfgs', random_state=0)

xgb = xgb.XGBClassifier()

cat = catboost.CatBoostClassifier(iterations=1000,learning_rate=0.03, depth=10, l2_leaf_reg=5, loss_function='Logloss', border_count=32,task_type="GPU")
rforest_model.fit(X, Y)

svc.fit(X, Y)

knn.fit(X, Y)

gbmodel.fit(X, Y)

gnb.fit(X, Y)

LR.fit(X, Y)

mlp.fit(X, Y)

xgb.fit(X, Y)

cat.fit(X, Y)
# predictions = 0.1 * rforest_model.predict(X_test) + 0.1 * svc.predict(X_test) + 0.1 * knn.predict(X_test) + 0.1 * gbmodel.predict(X_test) + 0.1 * gnb.predict(X_test) + 0.1 * LR.predict(X_test) + 0.1 * mlp.predict(X_test) + 0.1 * xgb.predict(X_test) + 0.2 * cat.predict(X_test)

predictions = cat.predict(X_test)
predictions = [0 if pred<0.5 else 1 for pred in predictions ]
submission = pd.read_csv('../input/sample_submission.csv')

submission['Survived'] = predictions

submission.head()
submission.to_csv('./simpletitanic.csv', index=False)