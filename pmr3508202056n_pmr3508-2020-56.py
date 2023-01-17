import pandas as pd

import sklearn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
adultTrain = pd.read_csv("../input/adult-pmr3508/train_data.csv", index_col=['Id'], na_values="?")
adultTrain.shape
adultTrain.head()
adultTrain.info()
adultTrain.describe()
substituicao = {"<=50K": 0, ">50K": 1}

adultTrain.income = [substituicao[i] for i in adultTrain.income]
plt.figure(figsize=(10,10))

sns.heatmap(adultTrain.corr(method="pearson"), square=True, annot=True, vmin=-1, vmax=1, cmap="YlGnBu")

plt.show()
sns.catplot(y="race", x="income", kind="bar", data=adultTrain)
sns.catplot(y="sex", x="income", kind="bar", data=adultTrain)
sns.catplot(y="occupation", x="income", kind="bar", data=adultTrain)
sns.catplot(y="workclass", x="income", kind="bar", data=adultTrain)
sns.catplot(y="education", x="income", kind="bar", data=adultTrain)
sns.catplot(y="marital.status", x="income", kind="bar", data=adultTrain)
sns.catplot(y="native.country", x="income", kind="bar", data=adultTrain)
adultTrain["native.country"].value_counts()
adultTrainParcial = adultTrain[["education", "education.num"]]

adultTrainParcial.drop_duplicates()
adultTrain = adultTrain.drop(["fnlwgt", "native.country", "education"], axis=1)
yTrain = adultTrain.pop("income")

xTrain = adultTrain
from sklearn.pipeline import  Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



pipelineCategorico = Pipeline(steps = [

    ('imputador', SimpleImputer(strategy="most_frequent")),

    ('onehot', OneHotEncoder(drop="if_binary"))

])
from sklearn.preprocessing import StandardScaler



pipelineNumerico = Pipeline(steps = [

    ('scaler', StandardScaler())

])
from sklearn.compose import ColumnTransformer



colunasNumericas = list(xTrain.select_dtypes(include = [np.number]).columns.values)

colunasCategoricas = list(xTrain.select_dtypes(exclude = [np.number]).columns.values)



preprocessador = ColumnTransformer(transformers = [

    ('numerico', pipelineNumerico, colunasNumericas),

    ('categorico', pipelineCategorico, colunasCategoricas)

])



xTrain = preprocessador.fit_transform(xTrain)
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier



melhorK = 10

melhorAcuracia = 0.0



for k in range(10, 31):

    acuracia = cross_val_score(KNeighborsClassifier(n_neighbors=k), xTrain, yTrain, cv=10, scoring="accuracy").mean()

    print('k = {} -> Acuracia = {:3.3f}%'.format(k, 100 * acuracia))

    if acuracia > melhorAcuracia:

        melhorAcuracia = acuracia

        melhorK = k

        

print('\nk que maximiza a acurácia: {}'.format(melhorK))

print('Máxima acurácia obtida: {:3.3f}% \n'.format(100 * melhorAcuracia))
melhorK = 10

melhorAcuracia = 0.0



for k in range(10, 31):

    acuracia = cross_val_score(KNeighborsClassifier(n_neighbors=k, weights='distance'), xTrain, yTrain, cv=10, scoring="accuracy").mean()

    print('k = {} -> Acuracia = {:3.3f}%'.format(k, 100 * acuracia))

    if acuracia > melhorAcuracia:

        melhorAcuracia = acuracia

        melhorK = k

        

print('\nk que maximiza a acurácia (com ponderação por distância): {}'.format(melhorK))

print('Máxima acurácia obtida (com ponderação por distância): {:3.3f}%'.format(100 * melhorAcuracia))
kNN = KNeighborsClassifier(n_neighbors=27)

kNN.fit(xTrain, yTrain)



adultTest = pd.read_csv("../input/adult-pmr3508/test_data.csv", index_col=['Id'], na_values="?")

xTest = adultTest.drop(['fnlwgt', 'native.country', 'education'], axis=1)

xTest = preprocessador.transform(xTest)

predicoes = kNN.predict(xTest)



predicoes
retorno = {0: "<=50K", 1: ">50K"}

predicoesCodificacaoOriginal = np.array([retorno[i] for i in predicoes], dtype=object)



predicoesCodificacaoOriginal
submissao = pd.DataFrame()

submissao[0] = adultTest.index

submissao[1] = predicoesCodificacaoOriginal

submissao.columns = ['Id', 'income']



submissao.to_csv('submission.csv', index = False)