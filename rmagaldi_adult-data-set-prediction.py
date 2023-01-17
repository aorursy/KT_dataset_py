import pandas as pd
import numpy as np
import sklearn
import matplotlib as plt
adult_data = pd.read_csv("../input/adult-data-set/train_data.csv", sep=r'\s*,\s*', engine='python', na_values="?")

adult_test = pd.read_csv('../input/adult-data-set/test_data.csv', sep=r'\s*,\s*', engine='python', na_values="?")
adult_data.head()
adult_data.shape
adult_data.income.value_counts().plot(kind="bar")
adult_data["sex"].value_counts().plot(kind="pie", radius=1.2, autopct='%1.1f%%')
adult_data.race.value_counts().plot(kind="bar")
adult_data["workclass"].value_counts()
adult_data_aux = adult_data.dropna()
adult_data_ = adult_data_aux[["age", "workclass", "fnlwgt", "education", "education.num", "marital.status", 
                                      "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss", 
                                      "hours.per.week", "native.country", "income"]]
adult_data_.shape
adult_test.head()
adult_test.shape
adult_test_ = adult_test[["age", "workclass", "fnlwgt", "education", "education.num", "marital.status", 
                                      "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss", 
                                      "hours.per.week", "native.country"]]
adult_test_.shape
Xadult = adult_data_[["age","education.num","capital.gain","capital.loss","hours.per.week"]]
Yadult = adult_data_.income
XtestAdult = adult_test_[["age","education.num","capital.gain","capital.loss","hours.per.week"]]
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=30)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
scores.mean()
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
YtestPred.shape
predict = pd.DataFrame(adult_test.Id)
predict["income"] = YtestPred
predict
predict.to_csv("prediction.csv", index=False)