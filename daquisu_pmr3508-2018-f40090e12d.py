# Adicionando diversas bibliotecas que iremos utilizar
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
adult = pd.read_csv("../input/testes-com-knn/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult
# note como o gráfico em pizza da education.num e da education são quase idênticas
adult["education.num"].value_counts().plot(kind="pie")
adult['education'].value_counts().plot(kind='pie')
adult['education'].value_counts()
# brincando um pouco com frequência relativa
adult['marital.status'].value_counts(normalize = True)
# frequência relativa não interfere no gráfico em pizza, porque ele já mostra a frequência relativa...
adult['race'].value_counts(normalize = True).plot(kind='pie')
# já no gráfico em barras, afeta e é interessante para visualizar melhor a porcentagem
# note que mais de 70% da base trabalha no setor privado
adult['workclass'].value_counts(normalize = True).plot(kind='bar')
# mais de 80% da base é branco/a
adult['race'].value_counts(normalize = True).plot(kind='bar')
# mais de 65% tem pelo menos ensino méio (junto com some-college, bachaleros, masters e doctorates)
adult['education'].value_counts(normalize = True).plot(kind='bar')
# mais de 70% recebe menos do que 50k dólares anuais
adult['income'].value_counts(normalize = True).plot(kind='bar')
total = len(adult)
# variável que será utilizada para tratamento de missing data
# é um dict relacionando cada variável numérica com sua respectiva média na base
values = {'age': adult['age'].mean(skipna = True), 'fnlwgt': adult['fnlwgt'].mean(skipna = True), 'education.num': adult['education.num'].mean(skipna = True), 'capital.gain': adult['capital.gain'].mean(skipna = True), 'capital.loss': adult['capital.loss'].mean(skipna = True), 'hours.per.week': adult['hours.per.week'].mean(skipna = True), 'income': '<=50K'}
# a média é trabalhar um pouco mais do que 40h semanais
values
# tratamento de missing data com a substituição pela média
nAdult = adult.fillna(value = values)
nAdult
# fazendo um for até 100 para ver se é possível detectar o melhor indice
# o resultado individual é guardado na variável performance_neighbors
melhor = 0
indice = 0
performance_neighbors = []
Xadult = nAdult[['age', 'fnlwgt', 'education.num', 'capital.gain', 'hours.per.week']]
Yadult = nAdult.income
for i in range (1, 100):
    knn = KNeighborsClassifier(n_neighbors = i)
    scores = cross_val_score(knn, Xadult, Yadult, cv=20)
    performance_neighbors.append(sum(scores)/20)
    if sum(scores)/20 > melhor:
        melhor = sum(scores)/20
        indice = i
melhor
indice
# fazendo kNN com o melhor indice
knn = KNeighborsClassifier(n_neighbors=indice)
# utilizando fit nos dados
knn.fit(Xadult, Yadult)
# obtendo a base de testes
adultTest = pd.read_csv("../input/testes-com-knn/test_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
# procedimento padrão para uma forma alternativa de lidar com missing data
valuesTest = {'age': adultTest['age'].mean(skipna = True), 'fnlwgt': adultTest['fnlwgt'].mean(skipna = True), 'education.num': adultTest['education.num'].mean(skipna = True), 'capital.gain': adultTest['capital.gain'].mean(skipna = True), 'capital.loss': adultTest['capital.loss'].mean(skipna = True), 'hours.per.week': adultTest['hours.per.week'].mean(skipna = True)}
nAdultTest = adultTest.fillna(value = valuesTest)
# tentando prever apenas com variaveis numéricas
YtestPred = knn.predict(nAdultTest[['age', 'fnlwgt', 'education.num', 'capital.gain', 'hours.per.week']])
# checando os resultados
YtestPred
# transformando os resultados em uma planilha de submissão
ids= adultTest.iloc[:,0].values
ids = ids.ravel()
dataset = pd.DataFrame({'Id':ids[:],'income':YtestPred[:]})
dataset.to_csv("submition.csv", index = False)
