import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle, islice
import statsmodels.api as sm

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
%matplotlib inline
#importando os dados
df= pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
#Verificando os dados
df.head()
#Criando um dataset com apenas os dados do Brazil
df = df[df['Country/Region'] =='Brazil']
#Verificando os tipos e quatidade de dados
df.info()
#Removendo colunas que não serão usadas nos modelos

df.drop(columns=['SNo','Province/State','Country/Region','Last Update'],axis=1,inplace=True)
df.head()
#Criando coluna contaminados
df['Contaminados']= df.Confirmed.astype(int) - df.Deaths.astype(int) - df.Recovered.astype(int)
df.sample(5)
#Transformando o tipo da variável para datetime
df['periodo'] = pd.to_datetime(df['ObservationDate'])

#Separando a variável 
df['dia'] = df['periodo'].dt.strftime('%d')
df['mes'] = df['periodo'].dt.strftime('%m')
df['ano'] = df['periodo'].dt.strftime('%Y')
df.head()
#verificando a transformação dos dados
df.info()
#Agora vou transformar o variável periodo no index da base.
df.index = df['periodo']
df.head()
#buscando registros aleatórios
df.sample(5)
plt.figure( figsize=(15, 8))
plt.bar(df.ObservationDate,df.Confirmed,label='Confirmados')
plt.bar(df.ObservationDate,df.Recovered,label='Recuperados')
plt.bar(df.ObservationDate,df.Deaths,label='Mortos',color='red')
plt.xticks(range(2,55,7),rotation='vertical')
plt.title("Evolução dos Confirmados x Recuperados x Mortos")
plt.ylabel('Contaminados x Recuperados x Mortos')
plt.xlabel('Período')
plt.show()

#plotando a evolução da contaminação
plt.rcParams['figure.figsize'] = (9,5)
sns.lineplot(x=df.mes,y=df.Contaminados,data=df,label='Contaminados')
sns.lineplot(x=df.mes,y=df.Recovered,data=df,label='Recuperados',color='green',linewidth=3)
sns.lineplot(x=df.mes,y=df.Deaths,data=df,label='Mortos',color='red')
plt.title("Evolução dos contaminados x Recuperados x Mortos por mês")
plt.ylabel('Contaminados x Recuperados x Mortos')
plt.xlabel('Mês')
plt.show()
my_colors = list(islice(cycle(['b','y','r', 'g']), None, len(df)))
df[['Confirmed','Contaminados','Deaths','Recovered']].sum().plot(kind='bar',color=my_colors)
plt.title("Quantidativos dos contaminados x Recuperados x Mortos")
plt.ylabel('Totais')
plt.show()


#Removendo a variável ObservationDate e preparando a base para rodar o modelo

df.drop('ObservationDate',inplace=True, axis=1)
df_contaminados = pd.DataFrame(df,columns=['Contaminados'])
df_contaminados.head()
#filtrando os contaminados que forem maior que zero
df_contaminados = df_contaminados[df_contaminados['Contaminados'] > 0]
#normalizando os dados
df_log = np.log(df_contaminados)
df_log.head()
# Estatísticas do modelo ARMA
modelo_arma = sm.tsa.ARMA(df_log, (3,0)).fit(disp=False)
print(modelo_arma.summary())

#Previssão para os próximos 4 meses
fig, ax = plt.subplots(figsize=(12,8))
fig = modelo_arma.plot_predict(start='2020-02-26', end='2020-06-20', ax=ax)
plt.title('Previssão para os próximos 4 meses - Modelo Arma')
plt.ylabel('Contaminados')
plt.xlabel('Periodo')
legend = ax.legend(loc='upper left')
modelo_arima = sm.tsa.ARIMA(df_log, order=(1, 1, 2)).fit()
print(modelo_arima.summary())
#Previssão do preço para 4 meses
fig, ax = plt.subplots(figsize=(10,8))
fig = modelo_arima.plot_predict(start='2020-02-27', end='2020-06-20', ax=ax)
plt.title('Previssão para os próximos 4 meses - Modelo Arima')
plt.ylabel('Contaminados')
plt.xlabel('Periodo')
legend = ax.legend(loc='upper left')
arima_ajustes = sm.tsa.ARIMA(df_log, order=(0, 1, 0))
arima_ajustes_treinado = arima_ajustes.fit()
print(arima_ajustes_treinado.summary())
#Previssão do preço para 4 meses
fig, ax = plt.subplots(figsize=(8,6))
fig = arima_ajustes_treinado.plot_predict(start='2020-02-27', end='2020-06-20', ax=ax)
plt.title('Previssão para os próximos 4 meses - Modelo Arima')
plt.ylabel('Contaminados')
plt.xlabel('Periodo')
legend = ax.legend(loc='upper left')
plt.show()
# Estatísticas do modelo SARIMA
modelo_sarima = sm.tsa.statespace.SARIMAX(df_log, freq='D',order=(7,1,7),seasonal_order=(0,0,0,0),
                                 enforce_stationarity=False, enforce_invertibility=False,).fit()

print(modelo_sarima.summary())
#Previssão do preço para 4 meses
fig, ax = plt.subplots(figsize=(8,6))
fig = modelo_sarima.predict(start='2020-02-27', end='2020-06-20', ax=ax).plot()
plt.title('Previssão para os próximos 4 meses - Modelo Sarima')
plt.ylabel('Contaminados')
plt.xlabel('Periodo')
legend = ax.legend(loc='upper left')
print('ARMA AIC: %5.2f\nARMA BIC %5.2f' %(modelo_arma.aic,modelo_arima.bic))
print('ARIMA AIC: %5.2f\nARIMA BIC %5.2f' %(modelo_arima.aic,modelo_arima.bic))
print('ARIMA COM AJUSTES AIC: %5.2f\nARIMA COM AJUSTES BIC %5.2f' %(arima_ajustes_treinado.aic,arima_ajustes_treinado.bic))
print('SARIMA AIC: %5.2f\nSARIMA BIC %5.2f' %(modelo_sarima.aic,modelo_sarima.bic))