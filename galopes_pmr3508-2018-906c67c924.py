%matplotlib inline
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import os
print(os.listdir('../input/mydata'))
adult = pd.read_csv("../input/mydata/train_data.csv",
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        skiprows=1,
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult.describe()
adult.shape
adult.head()
adult["Country"].value_counts()
adult["Age"].value_counts().plot(kind="bar")
adult["Sex"].value_counts().plot(kind="bar")
adult["Occupation"].value_counts().plot(kind="bar")
nadult = adult.dropna()
nadult
testAdult = pd.read_csv("../input/mydata/test_data.csv",
        names=[
        "ID","Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"],
        skiprows=1,
        index_col=0,
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
testAdult.head()
testAdult['Capital Gain'].plot()
nTestAdult = testAdult.dropna()
nTestAdult.shape
Xadult = nadult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
Xadult.head()
Yadult = nadult.Target
XtestAdult = nTestAdult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
XtestAdult.head()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
YtestPred
from sklearn.metrics import accuracy_score
UCITest = pd.read_csv("../input/mydata/adult.test",
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
UCITest.head()
nUCITest = UCITest.dropna()
nUCITest.head()
XUCITest = nUCITest[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
XUCITest.head()
YUCITest = nUCITest.Target
YUCIPred = knn.predict(XUCITest)
YUCIPred
accuracy_score(YUCIPred, YUCITest)
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(Xadult, Yadult)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores.mean()
YUCIPred = knn.predict(XUCITest)
accuracy_score(YUCITest, YUCIPred)
accuracies = {}

for i in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(Xadult, Yadult)
    scores = cross_val_score(knn, Xadult, Yadult, cv=10)
    Ypred = knn.predict(XUCITest)
    accuracy = accuracy_score(YUCITest,Ypred)
    accuracies[i] = accuracy
    print('k={}, accuracy={}, CVmean={}'.format(i, accuracy, scores.mean()))
    
ks = list(accuracies.keys())
acc = list(accuracies.values())
plt.plot(ks, acc)
plt.show()
k_optimal = max(accuracies, key=lambda key: accuracies[key])
print('O melhor k é {1}, com acurácia de {0}'.format(accuracies[k_optimal], k_optimal))
knn = KNeighborsClassifier(n_neighbors=28)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
knn.fit(Xadult, Yadult)
adult['Sex'] = adult['Sex'].transform(lambda x: 1 if x=='Male' else 0 if x==x else x)
testAdult.head()
predictions = knn.predict(testAdult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]])
print(len(testAdult.index.values), len(predictions))
result = np.vstack((testAdult.index.values, predictions)).T
x = ['Id','income']
resultado = pd.DataFrame(columns=x, data=result)
resultado.set_index('Id', inplace=True)
resultado.to_csv('mypredictions.csv')
