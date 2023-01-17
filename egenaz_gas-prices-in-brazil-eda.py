import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data1 = '/kaggle/input/gas-prices-in-brazil/2004-2019.tsv'
gp = pd.read_csv(data1, delimiter = '\t')
gp.rename(

    columns={

        "DATA INICIAL": "start_date",

        "DATA FINAL": "end_date",

        "REGIÃO": "region",

        "ESTADO": "state",

        "PRODUTO": "fuel",

        "NÚMERO DE POSTOS PESQUISADOS": "n_gas_stations",

        "UNIDADE DE MEDIDA": "unit",

        "PREÇO MÉDIO REVENDA": "avg_price",

        "DESVIO PADRÃO REVENDA": "sd_price",

        "PREÇO MÍNIMO REVENDA": "min_price",

        "PREÇO MÁXIMO REVENDA": "max_price",

        "MARGEM MÉDIA REVENDA": "avg_price_margin",

        "ANO": "year",

        "MÊS": "month",

        "COEF DE VARIAÇÃO DISTRIBUIÇÃO": "coef_dist",

        "PREÇO MÁXIMO DISTRIBUIÇÃO": "dist_max_price",

        "PREÇO MÍNIMO DISTRIBUIÇÃO": "dist_min_price",

        "DESVIO PADRÃO DISTRIBUIÇÃO": "dist_sd_price",

        "PREÇO MÉDIO DISTRIBUIÇÃO": "dist_avg_price",

        "COEF DE VARIAÇÃO REVENDA": "coef_price"

    },

    inplace=True

)





regions = {"SUL":"SOUTH", "SUDESTE":"SOUTHEAST", "CENTRO OESTE":"MIDWEST", 

            "NORTE":"NORTH", "NORDESTE":"NORTHEAST"}

gp["region"] = gp.region.map(regions)





fuels = {"ÓLEO DIESEL":"DIESEL", "GASOLINA COMUM":"REGULAR GASOLINE", "GLP":"LPG", 

            "ETANOL HIDRATADO":"HYDROUS ETHANOL", "GNV":"NATURAL GAS", "ÓLEO DIESEL S10":"DIESEL S10"}

gp["fuel"] = gp.fuel.map(fuels)
event_dictionary_units ={'R$/l ' : 'R$/l', 'R$/13Kg' : 'R$/l', 'R$/m3' : 'R$/l'}

event_dictionary_conversion ={'R$/l' : 1 , 'R$/13Kg' : 0.00006153, 'R$/m3' : 0.001}



gp['conversion'] = gp['unit'].map(event_dictionary_conversion)

gp['avg_price'] = gp.avg_price*gp.conversion

gp['unit'] = gp['unit'].map(event_dictionary_units)



gp.tail()
gp = gp.drop("unit", axis = 1)
gp.info()
for col in ['sd_price','min_price',

            'max_price','avg_price_margin','coef_price',

            'dist_avg_price', 'dist_sd_price', 'dist_min_price', 

            'dist_max_price', 'coef_dist']:

  

  gp[col] = pd.to_numeric(gp[col], errors='coerce')

    

gp.info()
from sklearn.preprocessing import LabelEncoder



labelencoder_region = LabelEncoder()

gp.region = labelencoder_region.fit_transform(gp.region)



labelencoder_state = LabelEncoder()

gp.state = labelencoder_state.fit_transform(gp.state)



labelencoder_fuel = LabelEncoder()

gp.fuel = labelencoder_fuel.fit_transform(gp.fuel)
gp.isnull().sum()
gp['avg_price_margin'].fillna(gp['avg_price_margin'].median(), inplace = True)

gp['dist_avg_price'].fillna(gp['dist_avg_price'].median(), inplace = True)

gp['dist_sd_price'].fillna(gp['dist_sd_price'].median(), inplace = True)

gp['dist_min_price'].fillna(gp['dist_min_price'].median(), inplace = True)

gp['dist_max_price'].fillna(gp['dist_max_price'].median(), inplace = True)

gp['coef_dist'].fillna(gp['coef_dist'].median(), inplace = True)
fig, ax = plt.subplots(figsize=(15,7))

gp.query('year != 2019 & fuel in ["3","4"]').groupby(['year','region']).sum()['avg_price'].unstack().plot(ax=ax)

plt.legend( labels = ['Midwest', 'North', 'Northeast', 'South', 'Southeast'])

plt.grid(True)
fig, ax = plt.subplots(figsize=(15,7))

gp.query('year != 2019 & fuel in ["3","4"]').groupby(['year','region']).sum()['avg_price'].pct_change().unstack().plot(ax=ax)

plt.legend(labels = ['Midwest', 'North', 'Northeast', 'South', 'Southeast'])

plt.grid(True)
fig, ax = plt.subplots(figsize=(15,7))

gp.query('year != 2019 & region in ["2"]').groupby(['year','state']).sum()['avg_price'].unstack().plot(ax=ax)

plt.legend(labels = ['Alagoas', 'Bahia', 'Ceara', 'Maranhao', 'Paraiba', 'Pernambuco', 'Piaui', 'Rio Gramde Do Nome', 'Sergipe'])

plt.grid(True)
fig, ax = plt.subplots(figsize=(15,7))

gp.query('year != 2019 & region in ["2"]').groupby(['year','state']).sum()['avg_price'].pct_change().unstack().plot(ax=ax)

plt.legend(labels = ['Alagoas', 'Bahia', 'Ceara', 'Maranhao', 'Paraiba', 'Pernambuco', 'Piaui', 'Rio Gramde Do Nome', 'Sergipe'])

plt.grid(True)
fig, ax = plt.subplots(figsize=(15,7))

gp.query('year != 2019 & region in ["1"]').groupby(['year','state']).sum()['avg_price'].unstack().plot(ax=ax)

plt.legend(labels = ['Acre', 'Amapa', 'Amazonas', 'Para', 'Rondonia', 'Roraima', 'Tokantis'])

plt.grid(True)
fig, ax = plt.subplots(figsize=(15,7))

gp.query('year != 2019 & region in ["1"]').groupby(['year','state']).sum()['avg_price'].pct_change().unstack().plot(ax=ax)

plt.legend(labels = ['Acre', 'Amapa', 'Amazonas', 'Para', 'Rondonia', 'Roraima', 'Tokantis'])

plt.grid(True)
fig, ax = plt.subplots(figsize=(20,14))

gp.query('year!=2019').groupby(['state','fuel'])['avg_price'].agg('sum').unstack().plot(kind='bar'

                                                                                         ,ax=ax)

plt.legend(labels = ['Diesel', 'Diesel S10', 'Hydrous Ethanol',

                     'LPG', 'Natural Gas', 'Regular Gasoline'])



positions = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26)

labels = ('Acre', 'Alagoas', 'Amapa', 'Amazonas', 'Bahia', 'Ceara', 'Distrito Federal',

          'Espirito Santo', 'Gaias', 'Maranhao', 'Mato Grosso', 'Mato Grosso do Sull',

         'Minas Gerais', 'Para', 'Paraiba', 'Parona', 'Pernambuco', 'Piaui',

         'Rio de Janeiro', 'Rio Grande Do Norte', 'Rio Grande Do Sul', 'Rondonia', 'Roraima', 'Santa Caterina',

         'Sao Paulo', 'Sergipe', 'Tokantins')

plt.xticks(positions, labels)

plt.grid(True)
fig, ax = plt.subplots(figsize=(20,14))

gp.query('year!=2019').groupby(['state','fuel'])['min_price'].agg('sum').unstack().plot(kind='bar'

                                                                                         ,ax=ax)

plt.legend(labels = ['Diesel', 'Diesel S10', 'Hydrous Ethanol',

                     'LPG', 'Natural Gas', 'Regular Gasoline'])



positions = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26)

labels = ('Acre', 'Alagoas', 'Amapa', 'Amazonas', 'Bahia', 'Ceara', 'Distrito Federal',

          'Espirito Santo', 'Gaias', 'Maranhao', 'Mato Grosso', 'Mato Grosso do Sull',

         'Minas Gerais', 'Para', 'Paraiba', 'Parona', 'Pernambuco', 'Piaui',

         'Rio de Janeiro', 'Rio Grande Do Norte', 'Rio Grande Do Sul', 'Rondonia', 'Roraima', 'Santa Caterina',

         'Sao Paulo', 'Sergipe', 'Tokantins')

plt.xticks(positions, labels)

plt.grid(True)
fig, ax = plt.subplots(figsize=(20,14))

gp.query('year!=2019').groupby(['state','fuel'])['max_price'].agg('sum').unstack().plot(kind='bar'

                                                                                         ,ax=ax)

plt.legend(labels = ['Diesel', 'Diesel S10', 'Hydrous Ethanol',

                     'LPG', 'Natural Gas', 'Regular Gasoline'])



positions = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26)

labels = ('Acre', 'Alagoas', 'Amapa', 'Amazonas', 'Bahia', 'Ceara', 'Distrito Federal',

          'Espirito Santo', 'Gaias', 'Maranhao', 'Mato Grosso', 'Mato Grosso do Sull',

         'Minas Gerais', 'Para', 'Paraiba', 'Parona', 'Pernambuco', 'Piaui',

         'Rio de Janeiro', 'Rio Grande Do Norte', 'Rio Grande Do Sul', 'Rondonia', 'Roraima', 'Santa Caterina',

         'Sao Paulo', 'Sergipe', 'Tokantins')

plt.xticks(positions, labels)

plt.grid(True)