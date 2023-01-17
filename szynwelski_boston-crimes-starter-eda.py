# Carregando bibliotecas

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns

import folium

from folium.plugins import HeatMap



# Importando dados

data = pd.read_csv('../input/crime.csv', encoding='latin-1')



# Olhadinha

data.head()
# somente dados de anos completos (2016, 2017)

data = data.loc[data['YEAR'].isin([2016,2017])]

# Manter apenas dados sobre ofensas da Parte 1 do UCR

data = data.loc[data['UCR_PART'] == 'Part One']



# Remover colunas não utilizadas

data = data.drop(['INCIDENT_NUMBER','OFFENSE_CODE','UCR_PART','Location'], axis=1)



# Converter OCCURED_ON_DATE em data e hora

data['OCCURRED_ON_DATE'] = pd.to_datetime(data['OCCURRED_ON_DATE'])



# Preencha nans na coluna SHOOTING

data.SHOOTING.fillna('N', inplace=True)



# Converter DAY_OF_WEEK em uma categoria ordenada

data.DAY_OF_WEEK = pd.Categorical(data.DAY_OF_WEEK, 

              categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],

              ordered=True)



# Substitua -1 valores em Lat / Long por Nan

data.Lat.replace(-1, None, inplace=True)

data.Long.replace(-1, None, inplace=True)



# Renomeie as colunas para algo mais fácil de digitar (as letras maiúsculas são irritantes!)

rename = {'OFFENSE_CODE_GROUP':'Group',

         'OFFENSE_DESCRIPTION':'Description',

         'DISTRICT':'District',

         'REPORTING_AREA':'Area',

         'SHOOTING':'Shooting',

         'OCCURRED_ON_DATE':'Date',

         'YEAR':'Year',

         'MONTH':'Month',

         'DAY_OF_WEEK':'Day',

         'HOUR':'Hour',

         'STREET':'Street'}

data.rename(index=str, columns=rename, inplace=True)



# Check

data.head()
# Mais algumas verificações de dados

data.dtypes

data.isnull().sum()

data.shape

# Gráfico de contagem para tipos de crime

sns.catplot(y='Group',

           kind='count',

            height=8, 

            aspect=1.5,

            order=data.Group.value_counts().index,

           data=data)
# Crimes por hora do dia

sns.catplot(x='Hour',

           kind='count',

            height=8.27, 

            aspect=3,

            color='black',

           data=data)

plt.xticks(size=30)

plt.yticks(size=30)

plt.xlabel('Hour', fontsize=40)

plt.ylabel('Count', fontsize=40)
# Crimes por dia da semana

sns.catplot(x='Day',

           kind='count',

            height=8, 

            aspect=3,

           data=data)

plt.xticks(size=30)

plt.yticks(size=30)

plt.xlabel('')

plt.ylabel('Count', fontsize=40)
# Crimes por mês do ano

months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

sns.catplot(x='Month',

           kind='count',

            height=8, 

            aspect=3,

            color='gray',

           data=data)

plt.xticks(np.arange(12), months, size=30)

plt.yticks(size=30)

plt.xlabel('')

plt.ylabel('Count', fontsize=40)
# Criar dados para plotagem

data['Day_of_year'] = data.Date.dt.dayofyear

data_holidays = data[data.Year == 2017].groupby(['Day_of_year']).size().reset_index(name='counts')



# Datas dos principais feriados nos EUA em 2017

holidays = pd.Series(['2017-01-01', # New Years Day

                     '2017-01-16', # MLK Day

                     '2017-03-17', # St. Patrick's Day

                     '2017-04-17', # Boston marathon

                     '2017-05-29', # Memorial Day

                     '2017-07-04', # Independence Day

                     '2017-09-04', # Labor Day

                     '2017-10-10', # Veterans Day

                     '2017-11-23', # Thanksgiving

                     '2017-12-25']) # Christmas

holidays = pd.to_datetime(holidays).dt.dayofyear

holidays_names = ['NY',

                 'MLK',

                 'St Pats',

                 'Marathon',

                 'Mem',

                 'July 4',

                 'Labor',

                 'Vets',

                 'Thnx',

                 'Xmas']



import datetime as dt

# Crimes de conspiração e feriados

fig, ax = plt.subplots(figsize=(11,6))

sns.lineplot(x='Day_of_year',

            y='counts',

            ax=ax,

            data=data_holidays)

plt.xlabel('Day of the year')

plt.vlines(holidays, 20, 80, alpha=0.5, color ='r')

for i in range(len(holidays)):

    plt.text(x=holidays[i], y=82, s=holidays_names[i])
#Gráfico de dispersão simples

sns.scatterplot(x='Lat',

               y='Long',

                alpha=0.01,

               data=data)
# Distritos da trama

sns.scatterplot(x='Lat',

               y='Long',

                hue='District',

                alpha=0.01,

               data=data)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
#Criar mapa básico do crime Folium

crime_map = folium.Map(location=[42.3125,-71.0875], 

                       tiles = "Stamen Toner",

                      zoom_start = 11)



# Adicionar dados para o mapa de calor

data_heatmap = data[data.Year == 2017]

data_heatmap = data[['Lat','Long']]

data_heatmap = data.dropna(axis=0, subset=['Lat','Long'])

data_heatmap = [[row['Lat'],row['Long']] for index, row in data_heatmap.iterrows()]

HeatMap(data_heatmap, radius=10).add_to(crime_map)



# Enredo!

crime_map