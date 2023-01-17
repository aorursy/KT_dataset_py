import pandas as pd
import sklearn

adult = pd.read_csv("../input/adult-dataset/train_data.csv",
        names=[
        "ID","Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
Paises=[["United-States", "Canada"],
["China", "Japan", "Taiwan", "Hong", "India", "South"],
["Vietnam", "Cambodia", "Laos", "Thailand", "Philippines", "Iran"],
["Germany", "England", "France", "Italy", "Holand-Netherlands", "Poland"], 
["Honduras", "Scotland", "Yugoslavia", "Ireland", "Greece", "Hungary", "Portugal"],
["Mexico", "Puerto-Rico", "El-Salvador", "Cuba", "Jamaica", "Dominican-Republic", 
"Guatemala", "Columbia", "Haiti", "Nicaragua", "Peru", "Ecuador", "Trinadad&Tobago", "Outlying-US(Guam-USVI-etc)"]]
'Mexico' in Paises[5]
for i in range(len(adult["Country"])):
    for j in range(len(Paises)):
        if adult.loc[i,("Country")] in Paises[j]:
            adult.loc[i,"Country"] = str(j)
nadult = adult.dropna()
testAdult = pd.read_csv("../input/adult-dataset/test_data.csv",
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
testAdult
for i in range(len(testAdult["Country"])):
    for j in range(len(Paises)):
        if testAdult.loc[i,("Country")] in Paises[j]:
            testAdult.loc[i,"Country"] = str(j)
nTestAdult = testAdult.dropna()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)
numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)
Xadult = numAdult[["Age", "Workclass", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"]]
XtestAdult = numTestAdult[["Age", "Workclass", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"]]
Yadult = nadult.Target

knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(Xadult,Yadult)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
YtestPred = knn.predict(XtestAdult)
YtestPred