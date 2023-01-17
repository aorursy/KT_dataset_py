import pandas as pd

import numpy as np

from datetime import datetime

import plotly.express as px

import plotly.graph_objects as go
covid = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates = ['ObservationDate', 'Last Update'])

covid
import re

def corrige_colunas(col_name):

    return re.sub(r'[/| ]', '', col_name).lower()
covid.columns = [corrige_colunas(col) for col in covid.columns]

covid
covid_br = covid.loc[covid.countryregion == 'Brazil']

covid_br
br_sum = covid_br.groupby('observationdate').agg({'confirmed': 'sum', 'deaths': 'sum', 'recovered': 'sum'}).reset_index()

br_sum
px.line(br_sum, "observationdate", "confirmed", title = 'Quadro Evolutivo do Número de Confirmados')
br_sum['novoscasos'] = list(map(

lambda x: 0 if (x == 0) else br_sum['confirmed'].iloc[x] - br_sum['confirmed'].iloc[x-1],

    np.arange(br_sum.shape[0])

))
br_sum
px.line(br_sum, x = "observationdate", y ="novoscasos", title = 'Quadro Evolutivo de Novos Casos Diários')
fig = go.Figure()

fig.add_trace(go.Scatter(x = br_sum.observationdate, y = br_sum.deaths, 

                         name = 'Número de Mortos', mode = 'lines', 

                         line = {'color':'red'}))

fig.update_layout(title = 'Quadro Evolutivo do Número de Mortes')

fig.show()
br_sum['novasmortes'] = list(map(

lambda x: 0 if (x == 0) else br_sum['deaths'].iloc[x] - br_sum['deaths'].iloc[x-1],

    np.arange(br_sum.shape[0])

))



br_sum
fig = go.Figure()

fig.add_trace(go.Scatter(x = br_sum.observationdate, y = br_sum.novasmortes, 

                         name = 'Mortes Diárias', mode = 'lines+markers', 

                         line = {'color':'blue'}))

fig.update_layout(title = 'Quadro Evolutivo do Número de Mortes Diárias')

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x = br_sum.observationdate, y = br_sum.recovered, 

                         name = 'Número de Recuperados', mode = 'lines', 

                         line = {'color':'orange'}))

fig.update_layout(title = 'Quadro Evolutivo do Número de Recuperados')

fig.show()
br_sum['novosrecuperados'] = list(map(

lambda x: 0 if (x == 0) else br_sum['recovered'].iloc[x] - br_sum['recovered'].iloc[x-1],

    np.arange(br_sum.shape[0])

))



br_sum
fig = go.Figure()

fig.add_trace(go.Scatter(x = br_sum.observationdate, y = br_sum.novosrecuperados, 

                         name = 'Número de Recuperados', mode = 'lines', 

                         line = {'color':'green'}))

fig.update_layout(title = 'Quadro Evolutivo do Número de Recuperados Por Dia')

fig.show()
def taxa_crescimento(data, variable, data_inicio = None, data_fim = None):

    if data_inicio == None:

        data_inicio = data.observationdate.loc[data[variable] > 0].min()

    else:

        data_inicio = pd.to_datetime(data_inicio)

        

    if data_fim == None:

        data_fim = data.observationdate.iloc[-1]

    else:

        data_fim = pd.to_datetime(data_fim)

    

    passado = data.loc[data.observationdate == data_inicio, variable].values[0]

    presente = data.loc[data.observationdate == data_fim, variable].values[0]

    

    n = (data_fim - data_inicio).days

    

    taxa = (presente/passado)**(1/n) - 1

    

    return taxa*100
taxa_crescimento(br_sum, 'confirmed')
def taxa_crescimento_diario(data, variable, data_inicio = None):

    if data_inicio == None:

        data_inicio = data.observationdate.loc[data[variable] > 0].min()

    else:

        data_inicio = pd.to_datetime(data_inicio)

        

    data_fim = data.observationdate.max()

    

    n = (data_fim - data_inicio).days

    

    taxa = list(map(lambda x: (data[variable].iloc[x] - data[variable].iloc[x-1]) / data[variable].iloc[x-1],

                   range(1, n+1)))

    

    return np.array(taxa)*100
tx = taxa_crescimento_diario(br_sum, 'confirmed')

tx
primeiro_dia = br_sum.observationdate.loc[br_sum.confirmed > 0 ].min()

px.line(x = pd.date_range(primeiro_dia, br_sum.observationdate.max())[1:], y = tx, title = 'Taxa de Crescimento diário da COVID-19 no Brasil')
confirmados = br_sum.confirmed

confirmados.index = br_sum.observationdate

confirmados
!pip install pmdarima
from pmdarima.arima import auto_arima

modelo = auto_arima(confirmados)
fig = go.Figure(go.Scatter(x = confirmados.index, y = confirmados, name = 'Observados'))

fig.add_trace(go.Scatter(x = confirmados.index, y = modelo.predict_in_sample(), name = 'Previsores'))

fig.add_trace(go.Scatter(x = pd.date_range('2020-09-16', '2020-10-01'), y = modelo.predict(18), name = 'Previsões'))

fig.update_layout(title = 'Previsão para o número de confirmados para os próximos 15 dias')

fig.show()
from statsmodels.tsa.seasonal import seasonal_decompose

import matplotlib.pyplot as plt
#confirmados = confirmados.sort_index(inplace= True)
decom = seasonal_decompose(x = confirmados, period = 101)
fig, (ax1,ax2,ax3, ax4) = plt.subplots(4,1, figsize=(15,8))

decom.observed.plot(ax=ax1)

decom.trend.plot(ax=ax2)

decom.seasonal.plot(ax=ax3)

decom.resid.plot(ax=ax4)

plt.tight_layout()
#!conda install -c conda-forge fbprophet -y
from fbprophet import Prophet
train = confirmados.reset_index()[:-5]

test = confirmados.reset_index()[-5: ]
train.rename(columns = {'observationdate': 'ds', 'confirmed': 'y'}, inplace = True)

test.rename(columns = {'observationdate': 'ds', 'confirmed': 'y'}, inplace = True)
pred = Prophet(growth = 'logistic', changepoints = ['2020-04-15', '2020-06-01', '2020-07-25'])
pop = 70683616

train['cap'] = pop
pred.fit(train)
datas_futuras = pred.make_future_dataframe(periods = 830)

datas_futuras['cap'] = pop

forecast = pred.predict(datas_futuras)
fig = go.Figure()

fig.add_trace(go.Scatter(x = forecast.ds, y = forecast.yhat, name = 'Previsões Futuras'))

fig.add_trace(go.Scatter(x = train.ds, y = train.y, name = 'Treino do Modelo'))

fig.update_layout(title = 'Previsão para o ponto de viarada dos quadros da COVID-19 no Brasil')

fig.show()
mortes = br_sum.deaths

mortes.index = br_sum.observationdate

mortes
modelo1 = auto_arima(mortes)
fig = go.Figure(go.Scatter(x = mortes.index, y = mortes, name = 'Observados'))

fig.add_trace(go.Scatter(x = mortes.index, y = modelo1.predict_in_sample(), name = 'Previsores'))

fig.add_trace(go.Scatter(x = pd.date_range('2020-09-16', '2020-10-01'), y = modelo1.predict(18), name = 'Previsões'))

fig.update_layout(title = 'Previsão para o número de Mortos para os próximos 15 dias')

fig.show()
recuperados = br_sum.recovered

recuperados.index = br_sum.observationdate

recuperados
modelo2 = auto_arima(recuperados)
fig = go.Figure(go.Scatter(x = recuperados.index, y = recuperados, name = 'Observados'))

fig.add_trace(go.Scatter(x = recuperados.index, y = modelo2.predict_in_sample(), name = 'Previsores'))

fig.add_trace(go.Scatter(x = pd.date_range('2020-09-16', '2020-10-01'), y = modelo2.predict(18), name = 'Previsões'))

fig.update_layout(title = 'Previsão para o número de Recuperados para os próximos 15 dias')

fig.show()
fig = go.Figure(go.Scatter(x = pd.date_range('2020-09-16', '2020-10-01'), y = modelo.predict(18), name = 'Previsões Confirmados'))

fig.add_trace(go.Scatter(x = pd.date_range('2020-09-16', '2020-10-01'), y = modelo1.predict(18), name = 'Previsões Mortos'))

fig.add_trace(go.Scatter(x = pd.date_range('2020-09-16', '2020-10-01'), y = modelo2.predict(18), name = 'Previsões Recuperados'))

fig.update_layout(title = 'Previsão geral dos números da COVID-19 para os próximos 15 dias')

fig.show()