import pandas as pd
import sklearn
from sklearn import preprocessing
train = pd.read_csv("../input/adultpmr3508/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
train
train.shape
train
train.tail(10)
#workclass marital.status relationship race native.country hours.per.week
train["hours.per.week"].value_counts()
import matplotlib.pyplot as plt
#drop: native.country; education; fnlwgt; capital.gain; capital.loss
train["native.country"].value_counts().plot(kind="bar")
train["education"].value_counts().plot(kind="bar")
train["fnlwgt"].value_counts()
na_train = train.dropna()
test = pd.read_csv("../input/adultpmr3508/test_data.csv")
test
na_test = test.dropna()
na_test
x = na_train.apply(preprocessing.LabelEncoder().fit_transform)
Xtrain = x.iloc[:,0:15]
Xtrain = Xtrain.loc[:, na_train.columns != "income"]
Xtrain = Xtrain.loc[:, Xtrain.columns != "Id"]
Xtrain = Xtrain.loc[:, Xtrain.columns != "native.country"]
Xtrain = Xtrain.loc[:, Xtrain.columns != "fnlwgt"]
Xtrain = Xtrain.loc[:, Xtrain.columns != "relationship"]
#Xtrain = Xtrain.loc[:, Xtrain.columns != "race"]
#Xtrain = Xtrain.loc[:, Xtrain.columns != "hours.per.week"]
#Xtrain = Xtrain.loc[:, Xtrain.columns != "workclass"]
Xtrain = Xtrain.loc[:, Xtrain.columns != "marital.status"]
Xtrain
na_test = na_test.apply(preprocessing.LabelEncoder().fit_transform)
#workclass marital.status relationship race native.country hours.per.week fnlwgt
Xtest = na_test.loc[:, na_test.columns != "Id"]
Xtest = Xtest.loc[:, Xtest.columns != "native.country"]
Xtest = Xtest.loc[:, Xtest.columns != "fnlwgt"]
#Xtest = Xtest.loc[:, Xtest.columns != "workclass"]
Xtest = Xtest.loc[:, Xtest.columns != "marital.status"]
Xtest = Xtest.loc[:, Xtest.columns != "relationship"]
#Xtest = Xtest.loc[:, Xtest.columns != "race"]
#Xtest = Xtest.loc[:, Xtest.columns != "hours.per.week"]
Xtest
Ytrain = na_train.income
Ytrain
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=25)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
scores
knn.fit(Xtrain,Ytrain)
YtestPred = knn.predict(Xtest)
YtestPred
pred = pd.DataFrame(test.Id)
pred["income"] = YtestPred
pred
pred.to_csv("prediction.csv", index=False)