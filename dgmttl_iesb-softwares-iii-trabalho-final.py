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
# Importando arquivo csv para o dataframe

df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")



# Conferindo os tipos de dados 

df.info()
# Conferindo a massa de dados

df.sample(15).T
# Filtrando a base de dados com informações do Brasil

df_brazil = df.copy()

df_brazil = df[df_brazil['Country/Region'] == 'Brazil']



df_brazil.info()
# Removendo colunas desnecessárias

df_brazil = df_brazil.drop('Last Update', axis=1)

df_brazil = df_brazil.drop('Province/State', axis=1)

df_brazil = df_brazil.drop('SNo', axis=1)



# Modificando tipos de dados

df_brazil['ObservationDate'] = pd.to_datetime(df_brazil.ObservationDate)



df_brazil = df_brazil.sort_values('ObservationDate')



df_brazil.info()
# Ajustando os índices

df_brazil = df_brazil.reset_index(drop=True)



df_brazil.head()
# Calculando novos casos e total de infectados



for index, row in df_brazil.iterrows():

    df_brazil['NovosCasos'] = df_brazil['Confirmed'] - df_brazil['Confirmed'].shift(1)

    

df_brazil['NovosCasos'] = df_brazil['NovosCasos'].fillna(0)





for index, row in df_brazil.iterrows():

    df_brazil['TotalInfectados'] = df_brazil['Confirmed'] - df_brazil['Recovered']

       

df_brazil.head(10)
df_brazil = df_brazil[df_brazil["Confirmed"] > 0]
df_brazil_lp = pd.melt(df_brazil, id_vars=['ObservationDate'], 

                      value_vars=['Confirmed', 'Deaths', 'Recovered', 'NovosCasos', 'TotalInfectados'])



# Importando biblioteca

import matplotlib.pyplot as plt

import seaborn as sns







# Plotando os números do Brasil

plt.figure(figsize=(15,5))

sns.lineplot(x='ObservationDate', y='value', data=df_brazil_lp, hue='variable')

plt.title("Números do COVID-19 no Brasil")

plt.xticks(rotation=90)



# Plotando a quantidade de casos confirmados no Brasil

plt.figure(figsize=(15,5))

sns.lineplot(x='ObservationDate', y='Confirmed', data=df_brazil)

plt.title("Quantidade de casos confirmados do COVID-19 no Brasil")

plt.xticks(rotation=90)
#Agrupando a base por data

df_brazil_data=df_brazil.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum',

                                                     'NovosCasos':'sum', 'TotalInfectados': 'sum'})



model_scores=[]



treino=df_brazil_data.iloc[:int(df_brazil_data.shape[0]*0.90)]

teste=df_brazil_data.iloc[int(df_brazil_data.shape[0]*0.90):]



from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing

from sklearn.metrics import mean_squared_error,r2_score



holt=Holt(np.asarray(treino["Confirmed"])).fit(smoothing_level=1.8, smoothing_slope=0.4)

y_pred=teste.copy()







y_pred["Holt"]=holt.forecast(len(teste))

model_scores.append(np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["Holt"])))

print("Erro Médio Quadrático de Holt: ",np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["Holt"])))
plt.figure(figsize=(10,5))

plt.plot(treino.Confirmed,label="Treino",marker='o')

teste.Confirmed.plot(label="Validação",marker='*')

y_pred.Holt.plot(label="Predição Holt",marker='^')

plt.ylabel("Quantidade de Infectados")

plt.xlabel("Dias")

plt.title("Total de Casos Ativos")

plt.xticks(rotation=90)

plt.legend()
treino=df_brazil_data.iloc[:int(df_brazil_data.shape[0]*0.90)]

teste=df_brazil_data.iloc[int(df_brazil_data.shape[0]*0.90):]

log_series=np.log(treino["Confirmed"])

y_pred=teste.copy()



from statsmodels.tsa.arima_model import ARIMA



arima=ARIMA(log_series,(3,2,4))

arima_fit = arima.fit()
predicao_arima=arima_fit.forecast(len(teste))[0]

y_pred["Predição ARIMA"]=list(np.exp(predicao_arima))



model_scores.append(np.sqrt(mean_squared_error(list(teste["Confirmed"]),np.exp(predicao_arima))))

print("Erro Médio Quadrático de ARIMA: ",np.sqrt(mean_squared_error(list(teste["Confirmed"]),np.exp(predicao_arima))))
plt.figure(figsize=(10,5))

plt.plot(treino.index,treino["Confirmed"],label="Treino",marker='o')

plt.plot(teste.index,teste["Confirmed"],label="Teste",marker='*')

plt.plot(y_pred["Predição ARIMA"],label="Predição ARIMA",marker='^')

plt.legend()

plt.xlabel("Date Time")

plt.ylabel('Confirmed Cases')

plt.title("Modelo de Previsão ARIMA")

plt.xticks(rotation=90)
model_scores