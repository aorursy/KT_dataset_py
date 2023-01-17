import pandas as pd

import sklearn

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

import matplotlib.pyplot

import numpy 
adultTrainData = pd.read_csv("../input/adultb/train_data.csv",

            names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        skiprows=[0],

        na_values="?")
adultTrainData.shape
adultTrainData.head(5)
cleanAdultTrainData = adultTrainData.dropna()

cleanAdultTrainData
from sklearn import preprocessing 

numCleanAdultTrainData = cleanAdultTrainData.apply(preprocessing.LabelEncoder().fit_transform)
adultTestData = pd.read_csv("../input/adultb/test_data.csv",

            names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"],

        sep=r'\s*,\s*',

        engine='python',

        skiprows=[0],

        na_values="?")
adultTestData.shape
adultTestData.head(5)
cleanAdultTestData = adultTestData.dropna()

cleanAdultTestData
from sklearn import preprocessing 

numCleanAdultTestData = cleanAdultTestData.apply(preprocessing.LabelEncoder().fit_transform)

numCleanAdultTestData
adultTrainData["Workclass"].value_counts().plot(kind = "bar")
adultTrainData["Age"].value_counts().plot(kind = "bar")
adultTrainData["Education"].value_counts().plot(kind = "bar")
adultTrainData["Capital Gain"].value_counts().plot(kind = "pie")
adultTrainData["Capital Loss"].value_counts().plot(kind = "pie")
adultTrainData["Hours per week"].value_counts().plot(kind = "pie")
adultTrainData["Race"].value_counts().plot(kind = "pie")
adultTrainData["Sex"].value_counts().plot(kind = "pie")
adultTrainData.describe()
Xadult = numCleanAdultTrainData[["Age", "Relationship","Race", "Martial Status", "Sex", "Country","Education-Num","Capital Gain", "Capital Loss"]]

XTestAdult = numCleanAdultTestData[["Age", "Relationship","Race", "Martial Status", "Sex", "Country","Education-Num","Capital Gain", "Capital Loss"]]

Yadult = cleanAdultTrainData.Target
n = 1;

knn = KNeighborsClassifier(n_neighbors=n)
scores = cross_val_score(knn, Xadult, Yadult, cv = 10) 

scoresAux = numpy.copy(scores)

while scores.mean() >= scoresAux.mean():

    scoresAux = numpy.copy(scores)

    n = n + 5

    knn = KNeighborsClassifier(n_neighbors=n)

    scores = cross_val_score(knn, Xadult, Yadult, cv = 10)

scores = numpy.copy(scoresAux)

n = n - 5

while scores.mean() >= scoresAux.mean():

    scoresAux = numpy.copy(scores)

    n = n + 1

    knn = KNeighborsClassifier(n_neighbors=n)

    scores = cross_val_score(knn, Xadult, Yadult, cv = 10)

scores = numpy.copy(scoresAux)

n = n - 1
print('n: ' + str(n))

print('Scores mean: ' + str(scores.mean()))

knn = KNeighborsClassifier(n_neighbors=n)
knn.fit(Xadult, Yadult)
YtestPred2 = knn.predict(XTestAdult)

YtestPred2
YtestPredFinal = []



count = 0

for i in range(YtestPred2.size):

    row = i - count

    dif =numCleanAdultTestData.index.values[row] - i 

    if ( dif == 0):

        YtestPredFinal.append(YtestPred2[row])

    else:

        for j in range(dif):

            YtestPredFinal.append(YtestPred2[row])

            count = count + 1

YtestPredFinal
Id = []

for i in range(0, YtestPred2.size):

    Id.append(i)
d = {'Id' : Id, 'Income' : YtestPredFinal}

my_df = pd.DataFrame(d) 

my_df.to_csv('prediction.csv',

             index=False, sep=',', line_terminator = '\n', header = ["Id", "income"])
my_df