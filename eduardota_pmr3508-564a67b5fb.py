import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
import category_encoders as ce
train = pd.read_csv("../input/train.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
train.shape
train.head()
train.pop("Id")
encoder = ce.TargetEncoder(handle_unknown="ignore")
encoder.fit(train, train.Target)
train = encoder.transform(train, train["Target"])
train.head()
imputer = sklearn.preprocessing.Imputer(strategy = "mean")
imputer.fit(train)
trainImputed = train.copy()
trainImputed = pd.DataFrame(imputer.transform(train))
trainImputed.columns = train.columns
trainImputed.index = train.index
trainImputed.head()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
Xtrain = trainImputed.loc[:, trainImputed.columns != "Target"]
Ytrain = trainImputed["Target"]
list = []

i=10
while (i <= 600):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(Xtrain,Ytrain)
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=7)
    accuracy = sum(scores)/len(scores)
    list.append([i,accuracy])
    i += 10
pd.DataFrame(list, columns=["k","accuracy"]).plot(x="k",y="accuracy",style="")
pd.DataFrame(list, columns=["k","accuracy"])
pd.DataFrame(list, columns=["k","accuracy"])["accuracy"].max()
#Best fit: K = 300
test = pd.read_csv("../input/test.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
test.pop("Id")
test.head()
test["Target"] = "0"

test = encoder.transform(test)

testImputed = test.copy()
testImputed = pd.DataFrame(imputer.transform(testImputed))
testImputed.columns = test.columns
testImputed.index = test.index

testImputed["Target"] = ""
testImputed.head()
Xtest = testImputed.loc[:, testImputed.columns != "Target"]
Ytest = testImputed["Target"]

knn = KNeighborsClassifier(n_neighbors=300)
knn.fit(Xtrain,Ytrain)
Ytest = knn.predict(Xtest)
Ytest
test = pd.read_csv("../input/test.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

Evaluation = pd.DataFrame(test["Id"])
Evaluation["Target"] = Ytest
Evaluation
Evaluation.to_csv("Evaluation.csv", index=False)