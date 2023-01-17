import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

from sklearn.preprocessing import MaxAbsScaler

from sklearn.preprocessing import StandardScaler

import seaborn as sb



%matplotlib inline

dados = pd.read_csv('../input/SEMANAL_BRASIL-DESDE_2013.csv', decimal=",")
dados.shape #Linhas x Colunas
dados.index # Descrição Index
dados.columns #Colunas presentes
dados.count() #Total dados não-nulos
numeroPostosPesquisados = dados.iloc[0:1824, 3]

semanaInicial = dados.iloc[0:1824, 0]

semanaFinal = dados.iloc[0:1824, 1]
tipoCombustivel = dados.set_index('PRODUTO')

tipoCombustivel.head()
gasolina = tipoCombustivel.loc['GASOLINA COMUM']
gasolina

numeroPP = gasolina.iloc[0:304 , 2]
valMax = numeroPP.max() 

print(valMax,"----> Número maximo de postos onde foram pesquisados o valor em R$, da gasolina")
valMin = numeroPP.min() 

print(valMin,"----> Número mínimo de postos onde foram pesquisados o valor em R$, da gasolina")
media = numeroPP.mean() 
print("{0:.0f}".format(round(media)),"----> Média de postos onde foram pesquisados o valor em R$, da gasolina ") 
histograma = numeroPP.hist()

print(histograma, "@@@ Esse é o gráfico de número de postos pesquisados em função do nº de postos pesquisados")
desvioPadrao = numeroPP.std()

print("{0:.0f}".format(round(desvioPadrao)),"----> Valor de desvio padrão ") 
mTruncada = valMax + valMin - media

print("{0:.0f}".format(round(mTruncada)),"----> Valor da Média Truncada ") 
gasolina.describe()
dadosKM = dados.iloc[0:1824, [8,14]]

dadosKM.replace(".",",")

dadosKM_array = dadosKM.values

kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=1234)

dadosKM["clusters"] = kmeans.fit_predict(dadosKM_array)

dadosKM.groupby("clusters").aggregate("mean").plot.bar(figsize=(10,7.5))

plt.title("(R$ Máximo) Revenda x Distribuição")
dadosKM
gasolinaKM = gasolina.iloc[0:1824, [7,13]]

gasolinaKM_array = gasolinaKM.values

kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=1234)

gasolinaKM["clusters"] = kmeans.fit_predict(gasolinaKM_array)

gasolinaKM.groupby("clusters").aggregate("mean").plot.bar(figsize=(10,7.5))

plt.title("R$ REVENDA X DISITRIBUIÇÃO (GASOLINA) ")
gasolinaKM
x = gasolinaKM.iloc[:, [0,1]].values

y = gasolinaKM.iloc[:, 2].values
from sklearn.model_selection import train_test_split  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)
from sklearn.preprocessing import StandardScaler  

scaler = StandardScaler()  

scaler.fit(x_train)



x_train = scaler.transform(x_train)  

x_test = scaler.transform(x_test) 
from sklearn.neighbors import KNeighborsClassifier  # import do KNN

classifier = KNeighborsClassifier(n_neighbors = 5)  # parametro = 5

classifier.fit(x_train, y_train)  
y_pred = classifier.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix  # import da matriz de confusão e report

print(confusion_matrix(y_test, y_pred)) # imprime a matriz de confusão

print(classification_report(y_test, y_pred)) # imprime as métricas
classifier = KNeighborsClassifier(n_neighbors = 18)

classifier.fit(x_train, y_train)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))