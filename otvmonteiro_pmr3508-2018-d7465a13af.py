import pandas as pd
import sklearn
adult = pd.read_csv("../input/pmr-base-adult/Adults.txt",
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

adult.shape
adult
adult["Target"].value_counts()
import matplotlib.pyplot as plt
adult["Education-Num"].plot(kind="hist")
nadult = adult.dropna()
nadult.shape
from sklearn.model_selection import train_test_split
Xadult, XtestAdult, Yadult, YtestAdult = train_test_split(

    nadult[["Age","Education-Num","Capital Gain", "Hours per week"]],
    nadult[["Target"]],
    test_size=0.3,
    random_state=101
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=40)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
accuracy_score(YtestAdult,YtestPred)
from sklearn import preprocessing
numXadult, numXtestAdult, Yadult, YtestAdult = train_test_split(

    nadult.apply(preprocessing.LabelEncoder().fit_transform),
    nadult[["Target"]],
    test_size=0.3,
    random_state=101
)
knn = KNeighborsClassifier(n_neighbors=40)
Xadult = numXadult[["Age", "Education-Num",
        "Occupation", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"]]

XtestAdult = numXtestAdult[["Age", "Education-Num",
        "Occupation", "Race", "Sex", "Capital Gain", "Capital Loss" ,
        "Hours per week", "Country"]]

knn.fit(Xadult, Yadult)
YtestPred = knn.predict(XtestAdult)
accuracy_score(YtestAdult,YtestPred)