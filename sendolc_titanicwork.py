
%matplotlib inline
import numpy as np 
import pandas as pd


titanic_train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, ) 
titanic_test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

print("\n\nSummary statistics of training data") 
print(titanic_train.describe())   
#learn variables' info 
print("\ninfo of variables in titanic_train\n")
titanic_train.info()
print("\ninfo of variables in titanic_test\n")
titanic_test.info()

#titanic_train["Embarked"] = titanic_train["Embarked"].fillna("S")
#titanic_train.loc[titanic_train["Embarked"]=="S", "Embarked"] = 0
#titanic_train.loc[titanic_train["Embarked"]=="C", "Embarked"] = 1
#titanic_train.loc[titanic_train["Embarked"]=="Q", "Embarked"] = 2
titanic_train.drop(["Embarked"], axis=1,inplace=True)
titanic_test.drop(["Embarked"], axis=1,inplace=True)

titanic_train.loc[titanic_train["Sex"]=="male", "Sex"] = 0
titanic_train.loc[titanic_train["Sex"]=="female", "Sex"] = 1
titanic_test.loc[titanic_test["Sex"]=="male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"]=="female", "Sex"] = 1

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
average_age_train  = titanic_train["Age"].mean()
std_age_train       = titanic_train["Age"].std()
average_age_test   = titanic_test["Age"].mean()
std_age_test       = titanic_test["Age"].std()

rand_1 = np.random.randint(average_age_train - std_age_train, average_age_train + std_age_train)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test)
titanic_train["Age"] = titanic_train["Age"].fillna(rand_1)
titanic_test["Age"] = titanic_test["Age"].fillna(rand_2 )

titanic_train["Family"] = titanic_train["SibSp"] + titanic_train["Parch"]
titanic_train["NameLength"] = titanic_train["Name"].apply(lambda x: len(x))

import re
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""
titles = titanic_train["Name"].apply(get_title)

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v


titanic_train["Title"] = titles

titles = titanic_test["Name"].apply(get_title)

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
for k,v in title_mapping.items():
    titles[titles == k] = v
titanic_test["Title"] = titles



#Linear Regression

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Title"]
alg = LinearRegression()
kf = KFold(titanic_train.shape[0], n_folds=3, random_state=1)
predictions = []
print(kf)
for train ,test in kf:
    train_predictors= (titanic_train[predictors].iloc[train,:])
    train_target = titanic_train["Survived"].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(titanic_train[predictors].iloc[test,:])
    predictions.append(test_predictions)
    
predictions = np.concatenate(predictions, axis=0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0
summ=sum(predictions == titanic_train["Survived"])
accuracy=float(summ)/len(predictions)
print(accuracy)
#Logistic Regression
from sklearn.linear_model import LogisticRegression 
from sklearn import cross_validation
alg = LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(alg, titanic_train[predictors], titanic_train["Survived"], cv=3)
print(scores.mean())
#Random forests
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Title"]

alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2 )
scores = cross_validation.cross_val_score(alg, titanic_train[predictors], titanic_train["Survived"], cv=3)
print(scores.mean())
from sklearn.ensemble import GradientBoostingClassifier

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Title" ]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare",  "Title", "Age"]]
]

kf = KFold(titanic_train.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_target = titanic_train["Survived"].iloc[train]
    full_test_predictions = []
    for alg, predictors in algorithms:
        alg.fit(titanic_train[predictors].iloc[train,:], train_target)
        test_predictions = alg.predict_proba(titanic_train[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)
summ=sum(predictions == titanic_train["Survived"])
accuracy=float(summ)/len(predictions)
print(accuracy)
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "Title", "Age"]]
]

full_predictions = []
for alg, predictors in algorithms:
    alg.fit(titanic_train[predictors], titanic_train["Survived"])
    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]
    full_predictions.append(predictions)

predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4
predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1
predictions = predictions.astype(int)
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv("kaggle.csv", index=False)
#alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2 )
#alg.fit(titanic_train[predictors], titanic_train["Survived"])
#predictions = alg.predict(titanic_test[predictors])
#submission = pd.DataFrame({
#        "PassengerId": titanic_test["PassengerId"],
#       "Survived": predictions
#    })
#submission.to_csv("kaggle.csv", index=False)