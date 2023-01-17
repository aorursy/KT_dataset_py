import pandas as pd
import sklearn
import numpy as nm
filetest = "../input/test-data/test_data.csv"
adult_test = pd.read_csv(filetest, 
        names=[
        "ID", "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult_test.shape
adult_test.head()
adult_test.drop(adult_test.index[0], inplace=True)
adult_test.drop("ID",axis=1, inplace=True)
adult_test.drop("Target",axis=1, inplace=True)
adult_test["Relationship"].value_counts()
adult_test.head()
for i in range(1,len(adult_test["Sex"])+1):
    x = adult_test["Race"][i]
    adult_test["Race"][i] = '1' if x=="White" else '-1'
    y = adult_test["Sex"][i]
    adult_test["Sex"][i] = '1' if y=="Male" else '-1'
    z = adult_test["Country"][i]
    adult_test["Country"][i] = '1' if z=="United-States" or z=="Canada" or z=="India" or z=="Germany" else '-1'if z=="Mexico" or z=="Jamaica" or z=="El-Salvador" or z=="Guatemala" or z=="Dominican-Republic" or z=="Haiti" else '0'
adult_test.head()
adult_test["Race"].value_counts()
adult_test["Sex"].value_counts()
adult = pd.read_csv("../input/train-data/train_data.csv",
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult.head()
adult.drop(adult.index[0], inplace=True)
for i in range(0,len(adult["Sex"])):
    x = adult["Race"][i]
    adult["Race"][i] = '1' if x=="White" else '-1'
    y = adult["Sex"][i]
    adult["Sex"][i] = '1' if y=="Male" else '-1'
    z = adult["Country"][i]
    adult["Country"][i] = '1' if z=="United-States" or z=="Canada" or z=="India" or z=="Germany" else '-1'if z=="Mexico" or z=="Jamaica" or z=="El-Salvador" or z=="Guatemala" or z=="Dominican-Republic" or z=="Haiti" else '0'
adult.head()
graph=adult.groupby(["Target","Country"]).size().unstack().plot(kind='bar')
Xadult = adult[["Age", "Education-Num", "Capital Gain", "Capital Loss", "Hours per week", "Sex", "Race", "Country"]]
Yadult = adult.Target
Xadult_test = adult_test[["Age", "Education-Num", "Capital Gain", "Capital Loss", "Hours per week", "Sex", "Race", "Country"]]
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
x = 5
for i in range(0,8):
    knn = KNeighborsClassifier(n_neighbors=(x))
    scores = cross_val_score(knn, Xadult, Yadult, cv=10)
    print(x)
    print(nm.mean(scores))
    x = x + 5
knn = KNeighborsClassifier(n_neighbors=(20))
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
nm.mean(scores)
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(Xadult_test)
YtestPred
fim = pd.DataFrame(YtestPred, columns = ["income"])
fim.to_csv("prediction.csv", index_label="Id")
fim
