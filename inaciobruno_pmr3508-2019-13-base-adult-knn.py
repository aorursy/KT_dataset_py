import pandas as pd

import numpy as np

import sklearn

import matplotlib.pyplot as plt



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn import preprocessing



%matplotlib inline
adult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", na_values = "?")

adult.shape
adult.head()
adult.describe().loc[:, 'age':]
adult.groupby(['sex', 'income']).size().unstack().plot(kind = 'bar', stacked = True)
relationship = adult.groupby(['relationship', 'income']).size().unstack()

relationship['sum'] = adult.groupby('relationship').size()

relationship = relationship.sort_values('sum', ascending = False)[['<=50K', '>50K']]

relationship.plot(kind = 'bar', stacked = True)
adult.groupby(['age', 'income']).size().unstack().plot(kind = 'bar', stacked = True)
education = adult.groupby(['education', 'income']).size().unstack()

education['sum'] = adult.groupby('education').size()

education = education.sort_values('sum', ascending = False)[['<=50K', '>50K']]

education.plot(kind = 'bar', stacked = True)
occupation = adult.groupby(['occupation', 'income']).size().unstack()

occupation['sum'] = adult.groupby('occupation').size()

occupation = occupation.sort_values('sum', ascending = False)[['<=50K', '>50K']]

occupation.plot(kind = 'bar', stacked = True)
adult.pivot(columns='income')['hours.per.week'].plot(kind = 'hist', stacked = True)
adult.groupby('occupation')['hours.per.week'].mean().sort_values(ascending = False).plot(kind = 'bar')
adult['native.country'].value_counts()
nadult = adult.dropna()

nadult.shape
nadult.describe().loc[:, 'age':]
testAdult = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", na_values = "?")

testAdult.shape
nTestAdult = testAdult.dropna()

nTestAdult.shape
numeric = ['age', 'education.num', 'capital.gain', 'capital.loss']

labels = ['sex', 'race', 'occupation', 'relationship', 'marital.status']

parameters = numeric + labels
numAdult = adult.fillna('?')

numAdult = pd.concat((numAdult[numeric], numAdult[labels].apply(preprocessing.LabelEncoder().fit_transform)), axis = 1)
Xadult = numAdult[parameters]

Yadult = adult.income
knn = KNeighborsClassifier(33, p = 1)

scores = cross_val_score(knn, Xadult, Yadult, cv = 5)

print(scores)

np.mean(scores)
%%time



acc = []



for k in range(15, 40):

    knn = KNeighborsClassifier(k, p = 1)

    scores = cross_val_score(knn, Xadult, Yadult, cv = 10)

    acc.append(np.mean(scores))



bestK = np.argmax(acc) + 15

print("Best acc: {}, K = {}".format(max(acc), bestK))
knn = KNeighborsClassifier(bestK, p = 1)

knn.fit(Xadult, Yadult)
numTestAdult = testAdult.fillna('?')

numTestAdult = pd.concat((numTestAdult[numeric], numTestAdult[labels].apply(preprocessing.LabelEncoder().fit_transform)), axis = 1)
XtestAdult = numTestAdult[parameters]

YtestAdult = knn.predict(XtestAdult)
id_index = pd.DataFrame({'Id' : list(range(len(YtestAdult)))})

income = pd.DataFrame({'income' : YtestAdult})

result = income

result
result.to_csv("submission.csv", index = True, index_label = 'Id')