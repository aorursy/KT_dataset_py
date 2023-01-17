#imports das bibliotecas

import numpy as np
import pandas as pd

import sklearn 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

import matplotlib as plt
import seaborn as sns
#Leitura dos dados de treinamento

adult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",
        na_values="?")
#verificação do cabeçalho

adult.head(20)
#Identificando os tipos de dados recebidos

adult.info()
#identificação dos percentis dos dados, média e desvio-padrão também

adult.describe(percentiles=[.05, .25, .5, .75, .95], include=['int', 'float'], exclude=[np.object])
#identificação de dados não numéricos

adult.describe(percentiles=None, include=[np.object])
#plotagem de gráficos com dados numéricos
#Pode-se ver que a maioria das pessoas com salário > 50K/ano possuem idade próxima a 40 anos
#E que as pessoas com salários menores do que 50K/ano são mais novas, com concentração entre 20 e 30 anos

fig, axes = plt.pyplot.subplots(nrows = 2, ncols = 1)
adult.loc[adult['income'] == '>50K']['age'].plot(kind = 'density', title = 'More than 50K', ax = axes[0], color = 'black', xlim = (10, 100))
adult.loc[adult['income'] == '<=50K']['age'].plot(kind = 'density', title = 'Less than 50K', ax = axes[1], color = 'red', xlim = (10, 100))
fig.subplots_adjust(bottom = -0.3)
#Horas de trabalho por densidade da concentração do salário
#Há correlação de que, quanto mais uma pessoa trabalha, maior é o salário (apesar desta correlação não ser muito alta)

fig, axes = plt.pyplot.subplots(nrows = 2, ncols = 1)
adult.loc[adult['income'] == '>50K']['hours.per.week'].plot(kind = 'density', title = 'More than 50K', ax = axes[0], color = 'black', xlim = (0,100))
adult.loc[adult['income'] == '<=50K']['hours.per.week'].plot(kind = 'density', title = 'Less than 50K', ax = axes[1], color = 'red', xlim = (0,100))
fig.subplots_adjust(bottom = -0.3)
fig, axes = plt.pyplot.subplots(nrows = 2, ncols = 1)
adult.loc[adult['income'] == '>50K']['education.num'].plot(kind = 'density', title = 'More than 50K', ax = axes[0], color = 'black', xlim = (0,20))
adult.loc[adult['income'] == '<=50K']['education.num'].plot(kind = 'density', title = 'Less than 50K', ax = axes[1], color = 'red', xlim = (0,20))
fig.subplots_adjust(bottom = -0.3)
#Peso final (fnlwgt) por densidade
#Pela formas semelhantes dos gráficos não há diferenças significativas

fig, axes = plt.pyplot.subplots(nrows = 2, ncols = 1)
adult.loc[adult['income'] == '>50K']['fnlwgt'].plot(kind = 'density', title = 'More than 50K', ax = axes[0], color = 'black', xlim = (0,10**6))
adult.loc[adult['income'] == '<=50K']['fnlwgt'].plot(kind = 'density', title = 'Less than 50K', ax = axes[1], color = 'red', xlim = (0,10**6))
fig.subplots_adjust(bottom = -0.3)
#Ganho de capital
#Maiores ganhos de capital são observados com maiores salários

fig, axes = plt.pyplot.subplots(nrows = 2, ncols = 1)
adult.loc[adult['income'] == '>50K']['capital.gain'].plot(kind = 'density', title = 'More than 50K', ax = axes[0], color = 'black', xlim = (0,10000))
adult.loc[adult['income'] == '<=50K']['capital.gain'].plot(kind = 'density', title = 'Less than 50K', ax = axes[1], color = 'red', xlim = (0,10000))
fig.subplots_adjust(bottom = -0.3)
#Perda de capital
#Apesar de não ser muito notória, a maior perda de capital também acompanha maiores salários, no geral

fig, axes = plt.pyplot.subplots(nrows = 2, ncols = 1)
adult.loc[adult['income'] == '>50K']['capital.loss'].plot(kind = 'density', title = 'More than 50K', ax = axes[0], color = 'black', xlim = (0,5000))
adult.loc[adult['income'] == '<=50K']['capital.loss'].plot(kind = 'density', title = 'Less than 50K', ax = axes[1], color = 'red', xlim = (0,5000))
fig.subplots_adjust(bottom = -0.3)
#Mapa de correlação
#Pode-se ver a correlação entre as variáveis numéricas
#No geral, todas as variáveis são correlacionadas de forma fraca
#As mais fortemente correlacionadas são número de anos de educação com horas de trabalho por semana / ganho de capital

sns.heatmap(adult.corr(), annot=True, cmap=plt.cm.Greens)
#remove valores não identificados da base de treinamento

nadult = adult.dropna()
#Correlação entre salário e dados discretos (não numéricos)
#Salário e sexo: pode-se notar que, no geral, homens tendem a ganhar salários maiores

a1 = nadult.groupby(['sex', 'income']).size().unstack()
a1['sum'] = nadult.groupby('sex').size()
a1 = a1.sort_values('sum', ascending = False)[['<=50K', '>50K']]
a1.plot(kind = 'bar', stacked = True)
#Salário e classe de trabalho: maiores proporções são observadas para trabalhadores dos setores privado, local-gov, self-empl-inc e self-empl-not-inc
#O mais baixo são o que não recebem pagamento..

a2 = nadult.groupby(['workclass', 'income']).size().unstack()
a2['sum'] = nadult.groupby('workclass').size()
a2 = a2.sort_values('workclass', ascending = False)[['<=50K', '>50K']]
a2.plot(kind = 'bar', stacked = True)
#Salário e situação de relacionamento: maridos e esposas recebem, proporcionalmente, os maiores salários de todos. Também pessoas que não estão numa família tendem a receber alto salário.

a3 = nadult.groupby(['relationship', 'income']).size().unstack()
a3['sum'] = nadult.groupby('relationship').size()
a3 = a3.sort_values('relationship', ascending = False)[['<=50K', '>50K']]
a3.plot(kind = 'bar', stacked = True)
#Status de casamento: nitidamente, a maior proporção salarial é para pessoas que estejam casadas

a4 = nadult.groupby(['marital.status', 'income']).size().unstack()
a4['sum'] = nadult.groupby('marital.status').size()
a4 = a4.sort_values('marital.status', ascending = False)[['<=50K', '>50K']]
a4.plot(kind = 'bar', stacked = True)
#Salário e ocupação

a5 = nadult.groupby(['occupation', 'income']).size().unstack()
a5['sum'] = nadult.groupby('occupation').size()
a5 = a5.sort_values('occupation', ascending = False)[['<=50K', '>50K']]
a5.plot(kind = 'bar', stacked = True)
#Salário e educação (contabilizada de forma não numérica): novamente, quanto maior o tempo de educação, maior tende a ser o salário

a6 = nadult.groupby(['education', 'income']).size().unstack()
a6['sum'] = nadult.groupby('education').size()
a6 = a6.sort_values('education', ascending = False)[['<=50K', '>50K']]
a6.plot(kind = 'bar', stacked = True)
#Não foi plotado gráfico devida a falta de proporção absurda das colunas
#Salário e etnia: no geral, asiáticos e ilheus do pacífico tendem a receber proporcionalmente os maiores salários
#seguidos por brancos, negros, latinos e outros

nadult.loc[adult['income'] == '>50K']["race"].value_counts(), nadult["race"].value_counts()
#País de origem: no geral, não dá para tirar conclusões devido a falta de dados dos outros países

nadult.loc[adult['income'] == '>50K']["native.country"].value_counts(), nadult["native.country"].value_counts()
#Encontrando o KNN por iterações

Xadult = nadult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
Yadult = nadult[['income']]
metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
weights = ['uniform', 'distance']
K_melhor = 0
media_melhor = 0

for i in range (20, 35):
    for j in range (2):
        for k in range (4):
            knn = KNeighborsClassifier(n_neighbors = i,  weights = weights[j], metric = metrics[k])
            scores = cross_val_score(knn, Xadult, Yadult.values.ravel(), cv=10)
            if scores.mean() > media_melhor:
                media_melhor = scores.mean()
                K_melhor = i
                parametros = [weights[j], metrics[k]]
                print(K_melhor, media_melhor, parametros)
                
print(K_melhor, media_melhor, parametros)
#Após encontrar os parâmetros com melhor perfomance, os parâmetros do algoritmo são ajustados

Xadult = nadult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
Yadult = nadult[["income"]]

knn = KNeighborsClassifier(n_neighbors=31, metric='chebyshev', weights='uniform')
knn.fit(Xadult, Yadult.values.ravel())
#Treinamento da base de dados

testAdult = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", na_values = "?")
numXtestAdult = testAdult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
YtestPred = knn.predict(numXtestAdult)
testAdult.shape
pd.DataFrame({'Id' : list(range(len(YtestPred)))})
resultado = pd.DataFrame({'income' : YtestPred})
print(resultado)
resultado.to_csv("submissao.csv", index = True, index_label = 'Id')