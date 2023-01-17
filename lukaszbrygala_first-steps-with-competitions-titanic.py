import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

import os

print(os.listdir("../input"))
train_orig = pd.read_csv("../input/train.csv")

train_orig.sample(5)
train_orig.shape
train_orig.isnull().sum()/train_orig.shape[1]
train_orig.corr()["Survived"].sort_values()
def preprocess(data):

    result = data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Age", "Fare"])

    

    # result["Age"].fillna(value=data["Age"].median(), inplace=True)

    # result["Fare"].fillna(value=data["Fare"].median(), inplace=True)

    result["Embarked"].fillna(value="S", inplace=True)

    

    title = data["Name"].str.extract(' (\w+)\. ', expand=False)

    title = title.replace("Mlle", "Miss")

    title = title.replace("Ms", "Miss")

    title = title.replace("Mme", "Mrs")

    title[~title.isin(["Mr", "Miss", "Mrs", "Master"])] = "Rare"

    result["Title"] = title

    

    result = pd.get_dummies(result)



    result["Fare_Binned"] = pd.qcut(data["Fare"].fillna(value=data["Fare"].median()), 6, labels=False)

    result["Age_Binned"] = pd.qcut(data["Age"].fillna(value=data["Age"].median()), 6, labels=False)

    return result
train_prep = preprocess(train_orig) 

y_train = train_prep["Survived"]

X_train = train_prep.drop(columns=["Survived"])
X_train.sample(5)
train_prep.corr()["Survived"].sort_values()
models = {

    "LR": LogisticRegression(),

    "SVC C=0.1": SVC(probability=True, C=0.1),

    "SVC C=0.3": SVC(probability=True, C=0.3),

    "SVC C=0.9": SVC(probability=True, C=0.9),

    "SVC C=3": SVC(probability=True, C=3),

    "SVC C=9": SVC(probability=True, C=9),

    "RFC": RandomForestClassifier()

}
score_by_model = {}

for model_name in models:

    # model.fit(X_train_sp, y_train_sp)

    # predictions = model.predict(X_test_sp)

    # scores[model.__class__.__name__] = accuracy_score(y_test_sp, predictions)

    model = models[model_name]

    scores = cross_val_score(model, X_train, y_train, cv=3)

    score_by_model[model_name] = scores.mean(), scores.std()*2
score_by_model
model = SVC(probability=True, C=0.3)

model.fit(X_train, y_train)
test_orig = pd.read_csv("../input/test.csv")

test_orig.isnull().sum()/test_orig.shape[1]
X_test = preprocess(test_orig) 
X_test.sample(5)
ids = test_orig["PassengerId"]

predictions = model.predict(X_test)
result = pd.DataFrame({ "PassengerId": ids, "Survived": predictions })
result.head()
result.to_csv("submission.csv", index=False)
! head submission.csv