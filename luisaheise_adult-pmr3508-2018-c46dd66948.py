import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import os
print(os.listdir("../input"))
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
adult = pd.read_csv("../input/adultb/train_data.csv", 
                    sep=r'\s*,\s*',
                    engine='python',
                    na_values="?")
complete_train_adult = adult.dropna()
complete_train_adult
complete_train_adult.age.value_counts().plot(kind='barh')
complete_train_adult.sex.value_counts().plot(kind='barh')
complete_train_adult.workclass.value_counts().plot(kind='barh')
numAdult = complete_train_adult.apply(preprocessing.LabelEncoder().fit_transform)
test_adult = pd.read_csv("../input/adultb/test_data.csv", 
                    sep=r'\s*,\s*',
                    engine='python',
                    na_values="?")
complete_test_adult = test_adult.dropna()
numTestAdult = complete_test_adult.apply(preprocessing.LabelEncoder().fit_transform)
Xadult = numAdult[["age", "workclass", "education.num", 
        "occupation", "race", "sex", "capital.gain", "capital.loss",
        "hours.per.week"]]
Yadult = complete_train_adult.income
XtestAdult = numTestAdult[["age", "workclass", "education.num", 
        "occupation", "race", "sex", "capital.gain", "capital.loss",
        "hours.per.week"]]
#tentativa de descobrir o melhor k de 0 até 100
better=0
melhork=0
for c in range(1,101):
    knn = KNeighborsClassifier(n_neighbors=c)
    scores = cross_val_score(knn, Xadult, Yadult, cv=10)
    if(scores.mean() > better):
        melhork = c
        better = scores.mean() 

melhork
better
knn = KNeighborsClassifier(n_neighbors=melhork)
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
YtestPred 
prediction = pd.DataFrame(XtestAdult)
prediction['income'] = YtestPred
prediction.to_csv()
prediction.head()
prediction.to_csv("prediction.csv")