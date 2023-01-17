# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Primeiramente, irei baixar as bibliotecas necessárias:

import pandas as pd 
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA

#Agora, importarei o arquivo do presente estudo - Covid_19_data.csv

Corona = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

Corona.head()
#O estudo sobre a evolução do Coronavírus será relativo apenas ao Brasil. Para isso, aplicarei um filtro na variável 'Country/Region'

Corona_Brazil = Corona[Corona['Country/Region'] == 'Brazil']

Corona_Brazil.head(20)
#Vou identificar quais colunas tem no Dataframe:

Corona_Brazil.columns

#Agora vou excluir algumas colunas que não farão parte do presente estudo.

Corona_Brazil = Corona_Brazil.drop(columns=['SNo'])
Corona_Brazil = Corona_Brazil.drop(columns=['Province/State'])
Corona_Brazil = Corona_Brazil.drop(columns=['Last Update'])
Corona_Brazil.head()

# Vou alterar os nomes das colunas da tabela:

Corona_Brazil.columns = ['Data', 'País', 'Confirmados', 'Mortos', 'Recuperados']
Corona_Brazil.head()
#Vou criar uma variável para identificar quem são os efetivamente doentes. A lógica é: Doentes = Confirmados - Mortos - Recuperados. 

Corona_Brazil['Doentes'] = Corona_Brazil['Confirmados'] - Corona_Brazil['Mortos'] - Corona_Brazil['Recuperados']
Corona_Brazil.head()
#Identificando o tamanho da tabela e os tipos de variáveis

Corona_Brazil.info()

display(Corona_Brazil.shape)
#Alterando o formato da Data

Corona_Brazil['Data'] = pd.to_datetime(Corona_Brazil['Data'])
Corona_Brazil.info()
#Selecionando as colunas que informam a Data e os Doentes:

Corona_Brazil = Corona_Brazil.iloc[:,[0,5]]
Corona_Brazil.head()
#Mostrando as variáveis 'Data' e 'Doentes' em gráfico

plt.figure(figsize=(20,10))
plt.title('Coronavírus no Brasil')
plt.plot(Corona_Brazil['Data'], Corona_Brazil['Doentes'])
#Predizendo os valores

X = Corona_Brazil.iloc[:,0].map(dt.datetime.toordinal).values 
y = Corona_Brazil.iloc[:,1].values
X = X.reshape(-1,1) 
lr = LinearRegression() 
lr.fit(X, y) 
display(lr.coef_) 
display(lr.intercept_) 
plt.figure(figsize=(20,5)) 
plt.title('Coronavírus no Brasil')
plt.plot(X, lr.predict(X))
plt.plot(Corona_Brazil['Data'], Corona_Brazil['Doentes'])
#Série temporal - Naive
y = np.asarray(Corona_Brazil['Doentes'], dtype='int')
nb_b = BernoulliNB()
nb_m = MultinomialNB()
nb_g = GaussianNB()
nb_b.fit(X,y)
nb_m.fit(X,y)
nb_g.fit(X,y)
# Mostra eixo Y
y
# mostra eito X
X
#Série temporal - projeção

plt.figure(figsize=(20,5)) 
plt.plot(Corona_Brazil['Data'], Corona_Brazil['Doentes'])
plt.plot(X, nb_g.predict(X))
plt.plot(X, nb_b.predict(X))
plt.plot(X, nb_m.predict(X))
#Utilizando o Naive

Corona_Brazil = pd.concat([Corona_Brazil['Doentes'],Corona_Brazil['Doentes'].shift(1)], axis=1, keys=['Doentes_Reais', 'Doentes_Preditos'])
Corona_Brazil.head()
Corona_Brazil_2 = Corona_Brazil[1:]

Corona_Brazil_2.Mse_baseline = np.sqrt(mean_squared_error(Corona_Brazil_2.Doentes_Reais, Corona_Brazil_2.Doentes_Preditos))
Corona_Brazil_2.r2_baseline = r2_score(Corona_Brazil_2.Doentes_Reais, Corona_Brazil_2.Doentes_Preditos)

print(Corona_Brazil_2.Mse_baseline)
print(Corona_Brazil_2.r2_baseline)
Corona_Brazil_2.columns
plt.figure(figsize=(20,10))
plt.title('Coronavírus no Brasil')
plt.grid()
plt.plot(Corona_Brazil_2.Doentes_Reais, color='Green')
plt.plot(Corona.Brazil_2.Doentes_Preditos, color='Blue')
plt.legend(['Doentes_Reais', 'Doentes_Preditos'])
plt.show()
#Criando a base de treino e a base de teste

X = np.reshape([i for i in range(0, len(Corona_Brazil_2['Doentes_Reais']))],(-1, 1))
y = Corona_Brazil_2['Doentes_Reais']
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.20, random_state=42)
#Regressão Linear

LR = LinearRegression(n_jobs=-1)
LR.fit(X_treino, y_treino)
LR_pred = lr.predict(X_teste)
plt.figure(figsize=(10,10))

plt.title('Coronavírus no Brasil')
plt.grid()
plt.plot(X, y, linewidth=2)
plt.plot(X, LR.predict(X), linestyle='--', linewidth=3, color='red')
plt.xlabel('Dias')
plt.ylabel('Doentes')
plt.legend(['Doentes_Reais','Doentes_Preditos'])
plt.show()
#Agora utilizando o Modelo ARIMA

#Esta metodologia consiste em ajustar modelos autorregressivos integrados de médias móveis, ARIMA(p,d,q), a um conjunto de dados.

#Preparando a base

plot_acf(Corona_Brazil_2.Doentes_Reais)
plt.show()

plot_pacf(Corona_Brazil_2.Doentes_Reais)
plt.show()
#Criando a base de treino e teste:

train_size = int(len(Corona_Brazil_2) * 0.70)
Corona_Brazil_2_train = Corona_Brazil_2.Doentes_Reais[:train_size].values.reshape(-1,1)
Corona_Brazil_2_test = Corona_Brazil_2.Doentes_Reais[train_size:].values.reshape(-1,1)

print(Corona_Brazil_2_train)
print(Corona_Brazil_2_test)
#ARIMA:

df_teste_arima = pd.DataFrame([])
rmse = pd.Series([])
ordem = pd.Series([])
r2 = pd.Series([])
aic = pd.Series([])
predicoes = pd.DataFrame([])
c = 0

for i in range(0, 4):
    for j in range(0, 4):
        for k in range(0, 2):
            try:
#Ordenando o modelo
                Corona_Brazil_2_modelo = ARIMA(Corona_Brazil_2_train, order=(i,k,j))
#Ajustando o modelo
                Corona_Brazil_2_modelo_fit = Corona_Brazil_2_modelo.fit()
#Calculando o critério aic - Critério de Informação de Akaike
                aic[c] = Corona_Brazil_2_modelo_fit.aic
#Realizando a predição
                Corona_Brazil_2_modelo_fit_forecast = Corona_Brazil_2_modelo_fit.forecast(steps=(len(Corona_Brazil_2)-train_size))[0]
#Colocando a ordem que será salva
                ordem[c] = '({}, {}, {})'.format(i,j,k)
#Salvando o R2
                r2[c] = r2_score(Corona_Brazil_2_test, Corona_Brazil_2_modelo_fit_forecast)
#Salvando o RMSE
                rmse[c] = np.sqrt(mean_squared_error(Corona_Brazil_2_test, Corona_Brazil_2_modelo_fit_forecast))
#Salvando as predições do modelo
                predicoes.insert(c, ordem[c] , Corona_Brazil_2_modelo_fit_forecast)

                c += 1
            except:
                continue

df_teste_arima = pd.concat([ordem, r2, rmse, aic], axis=1, keys=['Order', 'R2_score', 'RSME', 'AIC']).sort_values(by=['R2_score','RSME', 'AIC'], ascending=False)
df_teste_arima
#Monstrando os dados em gráfico

plt.figure(figsize=(20,15))
if len(df_teste_arima) % 2 == 0:
    for c in range(len(df_teste_arima)):
        plt.subplot(int(len(df_teste_arima)/4),4,c+1)
        plt.title('Ordem do Arima: {}'.format(predicoes.columns[c]))
        plt.grid()
        plt.plot(Corona_Brazil_2_train, linewidth=2)
        plt.plot([None for i in Corona_Brazil_2_train] + [j for j in Corona_Brazil_2_test], linewidth=2)
        plt.plot([None for i in Corona_Braizl_2_train] + [j for j in predicoes[predicoes.columns[c]]], linestyle='--', color='green', linewidth=3)
        plt.legend(['Treino', 'Teste', 'Predito'])
else: 
    for c in range(len(df_teste_arima)):
        plt.subplot(np.ceil(int(len(df_teste_arima)/3))+1,3,c+1)
        plt.title('Ordem do Arima: {}'.format(predicoes.columns[c]))
        plt.grid()
        plt.plot(Corona_Brazil_2_train, linewidth=2)
        plt.plot([None for i in Corona_Brazil_2_train] + [j for j in Corona_Brazil_2_test], linewidth=2)
        plt.plot([None for i in Corona_Brazil_2_train] + [j for j in predicoes[predicoes.columns[c]]], linestyle='--', color='green', linewidth=3)
        plt.legend(['Treino', 'Teste', 'Predito'])
plt.tight_layout() 
plt.show()
#Resultado final

Corona_Brazil_2.plot(kind='kde')