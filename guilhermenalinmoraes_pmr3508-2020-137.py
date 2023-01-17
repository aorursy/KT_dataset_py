#importando as bibliotecas necessária

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import seaborn as sns
#Dados de Treino

train_data = pd.read_csv("../input/adult-pmr3508/train_data.csv",na_values="?")

train_data.set_index('Id', inplace=True)
#Dados de Teste

test_data = pd.read_csv("../input/adult-pmr3508/test_data.csv",na_values="?")

test_data.set_index('Id',inplace = True)
#drop da coluna fnlwgt

train_data.drop(columns=["fnlwgt"], inplace=True)

test_data.drop(columns=["fnlwgt"], inplace=True)



#Leitura geral dos dados

train_data.head()
#Leitura dos tipos de dados por coluna

train_data.dtypes
#Diferenciação de colunas numéricas e categóricas

print("Colunas Categóricas")

categoric_Col=train_data.select_dtypes(['object']).columns

print(categoric_Col)



print("\nColunas Numéricas")

numeric_Col=train_data.select_dtypes(['int64']).columns

print(numeric_Col)
#Descrição dos dados

train_data.describe()
#Verificação da existência de nulos

print(train_data.isnull().sum())
#Dado que existem três colunas com missing data, entenderemos mais afundo essas colunas



print("Workclass")

print(train_data['workclass'].describe())

print("\nOccupation")

print(train_data['occupation'].describe())

print("\nnative.country")

print(train_data['native.country'].describe())
train_data.dropna(subset=["occupation"],inplace=True)

train_data.isnull().sum()



moda1 = train_data['workclass'].mode()[0]

train_data['workclass'].fillna(moda1, inplace = True)



moda2 = train_data['native.country'].mode()[0]

train_data['native.country'].fillna(moda2,inplace = True)
#missing data preenchida com sucesso

train_data.isnull().sum()
categoricTrain = train_data[categoric_Col].apply(pd.Categorical)



for col in categoric_Col:

    train_data[col + "_cat"] = categoricTrain[col].cat.codes



categoricTest = test_data[categoric_Col[:-1]].apply(pd.Categorical)



for col in categoric_Col[:-1]:

    test_data[col + "_cat"] = categoricTest[col].cat.codes



sns.heatmap(train_data.loc[:, [*numeric_Col, 'income_cat']].corr().round(2), vmin = -1., vmax = 1., 

            cmap = plt.cm.RdYlGn_r, annot = True)
fig, axes = plt.subplots(nrows = 3, ncols = 2)

plt.tight_layout(pad = .4, w_pad = .5, h_pad = 1.)





train_data.groupby(['sex', 'income']).size().unstack().plot(kind = 'bar', stacked = True, ax = axes[0, 0], figsize = (20, 15))



relationship = train_data.groupby(['relationship', 'income']).size().unstack()

relationship['sum'] = train_data.groupby('relationship').size()

relationship = relationship.sort_values('sum', ascending = False)[['<=50K', '>50K']]

relationship.plot(kind = 'bar', stacked = True, ax = axes[0, 1])



education = train_data.groupby(['education', 'income']).size().unstack()

education['sum'] = train_data.groupby('education').size()

education = education.sort_values('sum', ascending = False)[['<=50K', '>50K']]

education.plot(kind = 'bar', stacked = True, ax = axes[1, 0])



occupation = train_data.groupby(['occupation', 'income']).size().unstack()

occupation['sum'] = train_data.groupby('occupation').size()

occupation = occupation.sort_values('sum', ascending = False)[['<=50K', '>50K']]

occupation.plot(kind = 'bar', stacked = True, ax = axes[1, 1])



workclass = train_data.groupby(['workclass', 'income']).size().unstack()

workclass['sum'] = train_data.groupby('workclass').size()

workclass = workclass.sort_values('sum', ascending = False)[['<=50K', '>50K']]

workclass.plot(kind = 'bar', stacked = True, ax = axes[2, 0])



race = train_data.groupby(['race', 'income']).size().unstack()

race['sum'] = train_data.groupby('race').size()

race = race.sort_values('sum', ascending = False)[['<=50K', '>50K']]

race.plot(kind = 'bar', stacked = True, ax = axes[2, 1])
import sklearn

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
xColumns=train_data.select_dtypes(include=[np.number]).columns

xColumns = xColumns.drop("income_cat")
xColumns
x_train = train_data[xColumns]

y_train = train_data.income
%%time



score_rec=0.0





for k in range(30, 35):

    knn = KNeighborsClassifier(k, metric = 'manhattan')

    score = np.mean(cross_val_score(knn, x_train, y_train, cv = 10))

    if score > score_rec:

        bestK = k

        score_rec = score



print("Best acc: {}, K = {}".format(score, bestK))
knn = KNeighborsClassifier(bestK, metric = "manhattan")

knn.fit(x_train,y_train)
%%time



x_test = test_data[xColumns]

y_test = knn.predict(x_test)

y_test
prediction = pd.DataFrame(y_test)

prediction.columns=['income']

prediction['Id'] = prediction.index

prediction = prediction[['Id','income']]

prediction.head()
prediction.to_csv('submission.csv',index = False)