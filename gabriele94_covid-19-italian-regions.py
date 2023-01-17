# Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.express as px
data = pd.read_excel('../input/covid19-italian-regions/covid-19 20 marzo.xlsx')
data.info()
data.set_index('data', inplace=True)
data.info()
data.describe()
data.head(20)
latest= data.tail(21)
latest
px.bar(latest.sort_values('terapia_intensiva',ascending=False), x='denominazione_regione',y='terapia_intensiva',color='denominazione_regione',title='Total number of intensive care patients per region')
px.bar(latest.sort_values('deceduti',ascending=False), x='denominazione_regione',y='deceduti',color='denominazione_regione',title='Total number of deaths per region')
px.bar(latest.sort_values('tamponi',ascending=False),x='denominazione_regione',y='tamponi',color='denominazione_regione',title='Total number of tests per region')
px.bar(latest.sort_values('totale_attualmente_positivi',ascending=False), x='denominazione_regione',y='totale_attualmente_positivi',color='denominazione_regione',title='Total number of patients currently positive to the virus per region')
# calculate number of daily deaths per region

data['shift']=data.groupby('denominazione_regione')['deceduti'].shift(1)

data['deceduti_giorno']= data['deceduti'] - data['shift'] 
px.bar(data,x='denominazione_regione',y='deceduti_giorno',color=data.index,title='Daily deaths per region')
# calculate pct_change in daily deaths

data['deceduti_pct']=data.groupby('denominazione_regione')['deceduti'].pct_change()
px.bar(data,x='denominazione_regione',y='deceduti_pct',color=data.index,title='Pct_change in daily deaths per region')
# calculate daily intensive care patients per region

data['shift']=data.groupby('denominazione_regione')['terapia_intensiva'].shift(1)

data['terapia_giorno']=data['terapia_intensiva']- data['shift']
# calculate pct_change in intensive care patients per region

data['terapia_pct']=data.groupby('denominazione_regione')['terapia_intensiva'].pct_change()
px.bar(data,x='denominazione_regione',y='terapia_giorno',color=data.index, title='Daily intensive care admissions per region')
px.bar(data,x='denominazione_regione',y='terapia_pct',color=data.index, title='Pct_change in daily intensive care patients')
px.bar(data,x='denominazione_regione',y='nuovi_attualmente_positivi',color=data.index, title='New daily positive cases')
# calculate pct_change in new positive patients per day

data['nuovi_pct']=data.groupby('denominazione_regione')['nuovi_attualmente_positivi'].pct_change()
px.bar(data,x='denominazione_regione',y='nuovi_pct',color=data.index,title='Pct_change in new positive cases')
# calculate (new) daily hospitalised patients

data['shift']=data.groupby('denominazione_regione')['totale_ospedalizzati'].shift(1)

data['ospedalizzati_giorno']=data['totale_ospedalizzati']-data['shift']
# filtered tot hospitalised and new daily hospitalised for Piedmont

data[['ospedalizzati_giorno','totale_ospedalizzati','dimessi_guariti']][data['denominazione_regione']=='Piemonte']
#calculate pct_change in new hospitalised

data['ospedalizzati_pct']=data.groupby('denominazione_regione')['totale_ospedalizzati'].pct_change()
# filtered for Piedmont

data[['ospedalizzati_giorno','totale_ospedalizzati','ospedalizzati_pct']][data['denominazione_regione']=='Piemonte']
px.bar(data,x='denominazione_regione',y='ospedalizzati_giorno',color=data.index, title='New hospitalised patients (daily)')
px.bar(data,x='denominazione_regione',y='ospedalizzati_pct',color=data.index, title='Pct_change in hospitalised patients')
# virus mortality rate per region (mortality=deaths/tot cases)

data['tasso_letalita']=data['deceduti'].div(data['totale_casi'])
# filtered for Emilia Romagna region

data[['tasso_letalita','deceduti','totale_casi']][data['codice_regione']==8]
data['tasso_letalita'].fillna(0,inplace=True)
# calculate deaths per intensive care patients (many values are >1, a possible explanation could be the overcrowded hospitals)

data['deaths_per_intensive']=data['deceduti'].div(data['terapia_intensiva'])
data['deaths_per_intensive'].fillna(0, inplace=True)
px.bar(data, x=data.index, y='deceduti', color='denominazione_regione')
px.bar(data, x=data.index, y='tasso_letalita', color='denominazione_regione')
px.line(data, x=data.index, y='deaths_per_intensive', color='denominazione_regione', title='ratio deaths per intensive care patients')
px.bar(data, x=data.index, y='terapia_intensiva', color='denominazione_regione')
px.bar(data, x=data.index, y='terapia_giorno', color='denominazione_regione')
px.bar(data, x=data.index, y='nuovi_pct', color='denominazione_regione')
px.bar(data, x=data.index, y='deceduti_pct', color='denominazione_regione')
px.line(data, x=data.index, y='tasso_letalita', color='denominazione_regione')
px.line(data, x=data.index, y='nuovi_pct', color='denominazione_regione')
px.line(data, x=data.index, y='nuovi_attualmente_positivi', color='denominazione_regione')
px.bar(data, x=data.index, y='nuovi_attualmente_positivi', color='denominazione_regione')