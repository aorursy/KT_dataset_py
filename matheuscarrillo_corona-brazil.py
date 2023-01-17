import pandas as pd 

import numpy as np 

import seaborn as sns 

import matplotlib.pyplot as plt 

%matplotlib inline 



import warnings

warnings.filterwarnings("ignore")
corona_brazil = pd.read_csv("../input/corona-virus-brazil/brazil_covid19.csv")
last_day = list(corona_brazil['date'])[-1]
corona_brazil.info()
corona_brazil.columns
corona_brazil[corona_brazil['date']=='2020-03-22']['cases'].sum()
corona_brazil[corona_brazil['date']=='2020-03-23']['cases'].sum()
corona_brazil['chave'] = corona_brazil.apply(lambda row: row['date']+row['region']+row['state']+str(row['cases'])+str(row['deaths']), axis=1)
corona_brazil = corona_brazil.drop_duplicates(['chave'], keep='last' )
def acumulado(infectado):

    if len(lista_infectado)>0:

        infectado = infectado + lista_infectado[-1]

        lista_infectado.append(infectado)

    return infectado
corona_sp = corona_brazil[corona_brazil['state']=="São Paulo"]
def analise_dia(dia, what):

    infectados_dia = corona_brazil[corona_brazil['date']==dia][what].sum()

    return infectados_dia
corona_dia = pd.DataFrame()

corona_dia['date'] = corona_brazil['date'].unique()
corona_dia['Infectados_acumulado'] = corona_dia.apply(lambda row: analise_dia(row['date'], 'cases'), axis=1)

corona_dia['Mortos_acumulado'] = corona_dia.apply(lambda row: analise_dia(row['date'], 'deaths'), axis=1)
def casos_dia(row):

    casos = row-lista_casos[-1]

    lista_casos.append(row)

    return casos
lista_casos = [0]

corona_sp['cases_day'] = corona_sp.apply(lambda row: casos_dia(row['cases']), axis=1 )
lista_casos = [0]

corona_sp['deaths_day'] = corona_sp.apply(lambda row: casos_dia(row['deaths']), axis=1 )
lista_casos = [0]

corona_dia['cases_day'] = corona_dia.apply(lambda row: casos_dia(row['Infectados_acumulado']), axis=1 )
lista_casos = [0]

corona_dia['death_day'] = corona_dia.apply(lambda row: casos_dia(row['Mortos_acumulado']), axis=1 )
plt.figure(figsize=(20,10))

sns.lineplot(x='date', y='cases_day', data=corona_sp,palette = "Blues")

plt.xticks(rotation=-90)

for ponto in list(corona_sp.index):

    plt.text(x=corona_sp['date'][ponto], y=corona_sp['cases_day'][ponto], s=corona_sp['cases_day'][ponto], fontsize=15)

plt.title('Gráfico Evolução de Infectados por dia em São Paulo')
plt.figure(figsize=(20,10))

sns.barplot(x='date', y='cases', data=corona_sp,palette = "Blues")

plt.xticks(rotation=-90)

plt.title('Gráfico Evolução de Infectados Acumulado em São Paulo')
plt.figure(figsize=(20,10))

sns.lineplot(x='date', y='cases_day', data=corona_dia,palette = "Blues")

plt.xticks(rotation=-90)

for ponto in list(corona_dia.index):

    plt.text(x=corona_dia['date'][ponto], y=corona_dia['cases_day'][ponto], s=corona_dia['cases_day'][ponto], fontsize=15)

plt.title('Gráfico Evolução de Infectados por dia no Brasil')
plt.figure(figsize=(20,10))

sns.barplot(x='date', y='Infectados_acumulado', data=corona_dia,palette = "Blues")

plt.xticks(rotation=-90)

plt.title('Gráfico Evolução de Infectados Acumulado no Brasil')
estados = list(corona_brazil['state'].unique())

corona_estado = pd.DataFrame(columns=['Estado','Suspeitos','Infectados','Liberados','Mortos'])
Infectados = []

Mortos = []

corona_filter = corona_brazil[corona_brazil['date']==last_day]

for estado in estados:    

    Infectados.append(corona_filter[corona_filter['state']==estado]['cases'].sum())

    Mortos.append(corona_filter[corona_filter['state']==estado]['deaths'].sum())

corona_estado['Estado'] = estados

corona_estado['Infectados'] = Infectados

corona_estado['Mortos'] = Mortos
ordernation = corona_estado.groupby(["Estado"])['Infectados'].aggregate(np.median).reset_index().sort_values('Infectados')



order = []

lista_estados_ord = list(ordernation['Estado'])

contagem = len(lista_estados_ord)



for i in range(contagem-1, -1, -1):

    order.append(lista_estados_ord[i])

    
plt.figure(figsize=(10,10))

sns.barplot(x="Infectados", y="Estado", data=corona_estado, color="b", order=order)

print('Gráfico de casos por estados')
plt.figure(figsize=(20,10))

sns.lineplot(x='date', y='deaths_day', data=corona_sp,palette = "Blues")

plt.xticks(rotation=-90)

for ponto in list(corona_sp.index):

    plt.text(x=corona_sp['date'][ponto], y=corona_sp['deaths_day'][ponto], s=corona_sp['deaths_day'][ponto], fontsize=15)

plt.title('Gráfico Evolução de Mortos por dia em São Paulo')
plt.figure(figsize=(10,10))

sns.barplot(x="Mortos", y="Estado", data=corona_estado, color="b", order=order)

plt.title('Gráfico de mortes por estados')
plt.figure(figsize=(20,10))

sns.lineplot(x='date', y='death_day', data=corona_dia,palette = "Blues")

plt.xticks(rotation=-90)

for ponto in list(corona_dia.index):

    plt.text(x=corona_dia['date'][ponto], y=corona_dia['death_day'][ponto], s=corona_dia['death_day'][ponto], fontsize=15)

plt.title('Gráfico Evolução de Mortos por dia no Brasil')
plt.figure(figsize=(20,10))

sns.barplot(x='date', y='Mortos_acumulado', data=corona_dia,palette = "Blues")

plt.xticks(rotation=-90)

plt.title('Gráfico Evolução de Mortos Acumulado no Brasil')