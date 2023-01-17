import pandas as pd
import numpy as np

import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

import matplotlib as plt


# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
adult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",
        na_values="?")
adult.shape
adult.head()
adult.describe(include = [np.number])
adult.describe(exclude = [np.number])
adultRich = adult.loc[adult['income'] == '>50K']
adultNotRich = adult.loc[adult['income'] == '<=50K']
categoricColumns = adult.select_dtypes(exclude = [np.number]).columns

categoricAdult = adult[categoricColumns].apply(pd.Categorical)

for col in categoricColumns:
    adult[col + "_cat"] = categoricAdult[col].cat.codes
fig, axes = plt.pyplot.subplots(nrows = 1, ncols = 2)
adultNotRich['age'].plot(kind = 'density', title = 'NotRich', ax = axes[0],figsize = (15, 4))
adultRich['age'].plot(kind = 'density', title = 'Rich', ax = axes[1],figsize = (15, 4))
adult[['age','income_cat']].corr()
adult["workclass"].value_counts()
workclass = adult.groupby(['workclass', 'income']).size().unstack()
workclass['sum'] = adult.groupby('workclass').size()
workclass = workclass.sort_values('sum', ascending = False)[['<=50K', '>50K']]
workclass.plot(kind = 'bar', stacked = True, figsize = (15, 4))
adult[['workclass_cat','income_cat']].corr()
fig, axes = plt.pyplot.subplots(nrows = 1, ncols = 2)
adultNotRich['fnlwgt'].plot(kind = 'density', title = 'NotRich', ax = axes[0],figsize = (15, 4))
adultRich['fnlwgt'].plot(kind = 'density', title = 'Rich', ax = axes[1],figsize = (15, 4))
adult[['fnlwgt','income_cat']].corr()
adult["education"].value_counts()
fig, axes = plt.pyplot.subplots(nrows = 1, ncols = 2)
adultNotRich['education.num'].plot(kind = 'density', title = 'NotRich', ax = axes[0],figsize = (15, 4))
adultRich['education.num'].plot(kind = 'density', title = 'Rich', ax = axes[1],figsize = (15, 4))
adult[['education.num','income_cat']].corr()
adult["occupation"].value_counts()
occupation = adult.groupby(['occupation', 'income']).size().unstack()
occupation['sum'] = adult.groupby('occupation').size()
occupation = occupation.sort_values('sum', ascending = False)[['<=50K', '>50K']]
occupation.plot(kind = 'bar', stacked = True, figsize = (15, 4))
adult[['occupation_cat','income_cat']].corr()
adult["relationship"].value_counts()
relationship = adult.groupby(['relationship', 'income']).size().unstack()
relationship['sum'] = adult.groupby('relationship').size()
relationship = relationship.sort_values('sum', ascending = False)[['<=50K', '>50K']]
relationship.plot(kind = 'bar', stacked = True, figsize = (15, 4))
adult[['relationship_cat','income_cat']].corr()
adult["marital.status"].value_counts()
maritalStatus = adult.groupby(['marital.status', 'income']).size().unstack()
maritalStatus['sum'] = adult.groupby('marital.status').size()
maritalStatus = maritalStatus.sort_values('sum', ascending = False)[['<=50K', '>50K']]
maritalStatus.plot(kind = 'bar', stacked = True, figsize = (15, 4))
adult[['marital.status_cat','income_cat']].corr()
adult["race"].value_counts()
race = adult.groupby(['race', 'income']).size().unstack()
race['sum'] = adult.groupby('race').size()
race = race.sort_values('sum', ascending = False)[['<=50K', '>50K']]
race.plot(kind = 'bar', stacked = True, figsize = (15, 4))
adult[['race_cat','income_cat']].corr()
adult["sex"].value_counts()
sex = adult.groupby(['sex', 'income']).size().unstack()
sex['sum'] = adult.groupby('sex').size()
sex = sex.sort_values('sum', ascending = False)[['<=50K', '>50K']]
sex.plot(kind = 'bar', stacked = True, figsize = (15, 4))
adult[['sex_cat','income_cat']].corr()
fig, axes = plt.pyplot.subplots(nrows = 1, ncols = 2)
adultNotRich['capital.gain'].plot(kind = 'density', title = 'NotRich', ax = axes[0],figsize = (15, 4))
adultRich['capital.gain'].plot(kind = 'density', title = 'Rich', ax = axes[1],figsize = (15, 4))
adult[['capital.gain','income_cat']].corr()
fig, axes = plt.pyplot.subplots(nrows = 1, ncols = 2)
adultNotRich['capital.loss'].plot(kind = 'density', title = 'NotRich', ax = axes[0],figsize = (15, 4))
adultRich['capital.loss'].plot(kind = 'density', title = 'Rich', ax = axes[1],figsize = (15, 4))
adult[['capital.loss','income_cat']].corr()
fig, axes = plt.pyplot.subplots(nrows = 1, ncols = 2)
adultNotRich['hours.per.week'].plot(kind = 'density', title = 'NotRich', ax = axes[0],figsize = (15, 4))
adultRich['hours.per.week'].plot(kind = 'density', title = 'Rich', ax = axes[1],figsize = (15, 4))
adult[['hours.per.week','income_cat']].corr()
adult["native.country"].value_counts()
adult[['native.country_cat','income_cat']].corr()
adult["income"].value_counts()
def findBestK(Xadult, Yadult, metric, weights):
    bestK = 0
    bestMeanScore = 0
    for i in range(15,35):
        knn = KNeighborsClassifier(n_neighbors=i, metric=metric, weights=weights)
        scores = cross_val_score(knn, Xadult, Yadult, cv=10)
        newMeanScore = scores.mean()
        if newMeanScore > bestMeanScore:
            bestMeanScore = newMeanScore
            bestK = i
    return bestK, bestMeanScore
def testAux(Xadult,Yadult):
    for metric in ('manhattan', 'minkowski'):
        for weights in ('distance', 'uniform'):
            print ("########## metric = %s , weights = %s ##########"%(metric, weights))
            K, meanScore = findBestK(Xadult, Yadult, metric, weights)
            print('K = %d'%(K))
            print('meanScore = %f'%(meanScore))
            print(' ')
Xadult = adult[["age","education.num","capital.gain", "capital.loss", "hours.per.week", 
                "sex_cat", "marital.status_cat", "relationship_cat"]]
Yadult = adult["income_cat"]

testAux(Xadult, Yadult)
# Retirando os dados faltantes

Xadult = adult.dropna()[["age","education.num","capital.gain", "capital.loss", "hours.per.week", 
                "sex_cat", "marital.status_cat", "relationship_cat"]]
Yadult = adult.dropna()["income_cat"]

testAux(Xadult, Yadult)
# Colocando outras características que potencialmente podem melhorar o desempenho

Xadult = adult[["age","education.num","capital.gain", "capital.loss", "hours.per.week", 
                "sex_cat", "marital.status_cat", "relationship_cat", "race_cat", "occupation_cat"]]
Yadult = adult["income_cat"]

testAux(Xadult, Yadult)
# recarregando o DataSet pra garantir

adult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",
        na_values="?")

categoricColumns = adult.select_dtypes(exclude = [np.number]).columns

categoricAdult = adult[categoricColumns].apply(pd.Categorical)

for col in categoricColumns:
    adult[col + "_cat"] = categoricAdult[col].cat.codes
Xadult = adult[["age","education.num","capital.gain", "capital.loss", "hours.per.week", 
                "sex_cat", "marital.status_cat", "relationship_cat", "race_cat", "occupation_cat"]]
Yadult = adult["income"]
knn = KNeighborsClassifier(n_neighbors=20, metric='manhattan', weights='uniform')
knn.fit(Xadult,Yadult)
XtestAdult = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", na_values = "?")

categoricColumns = XtestAdult.select_dtypes(exclude = [np.number]).columns
categoricAdult = XtestAdult[categoricColumns].apply(pd.Categorical)

for col in categoricColumns:
    XtestAdult[col + "_cat"] = categoricAdult[col].cat.codes

XtestAdult.head()
numXtestAdult = XtestAdult[["age","education.num","capital.gain", "capital.loss", "hours.per.week", 
                "sex_cat", "marital.status_cat", "relationship_cat", "race_cat", "occupation_cat"]]

YtestPred = knn.predict(numXtestAdult)
id_index = pd.DataFrame({'Id' : list(range(len(YtestPred)))})
income = pd.DataFrame({'income' : YtestPred})
result = income
result
result.to_csv("submission.csv", index = True, index_label = 'Id')