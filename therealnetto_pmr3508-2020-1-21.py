import pandas as pd

import sklearn

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn import preprocessing



adult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", na_values = "?").dropna()

adultTest = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", na_values = "?").dropna()
adult.shape
adult["age"].value_counts().plot(kind="bar")
adult["education"].value_counts().plot(kind="bar")
Xadult = adult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]

XadultTest = adult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]

Yadult = adult.income
knn = KNeighborsClassifier(n_neighbors=3)

scores = cross_val_score(knn, Xadult, Yadult) #default 5-fold cross validation

knn.fit(Xadult,Yadult)

YtestPred = knn.predict(XadultTest)

YtestPred
accuracy3 = accuracy_score(Yadult,YtestPred)

accuracy3
knn = KNeighborsClassifier(n_neighbors=30)

scores = cross_val_score(knn, Xadult, Yadult) #default 5-fold cross validation

knn.fit(Xadult,Yadult)

YtestPred = knn.predict(XadultTest)

YtestPred
accuracy30 = accuracy_score(Yadult,YtestPred)

accuracy30
knn = KNeighborsClassifier(n_neighbors=15)

scores = cross_val_score(knn, Xadult, Yadult) #default 5-fold cross validation

knn.fit(Xadult,Yadult)

YtestPred = knn.predict(XadultTest)

YtestPred
accuracy15 = accuracy_score(Yadult,YtestPred)

accuracy15
results = [accuracy3,accuracy15,accuracy30]
results
sub = pd.DataFrame()

sub[0] = adult.index

sub[1] = YtestPred

sub.columns = ['Id', 'income']

sub.to_csv('submission.csv', index=False)