# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # seaborn package

import matplotlib.pyplot as plt # matplotlib library



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
os.chdir('../input')

df = pd.read_csv('2004-2019.tsv', sep='\t',parse_dates=[1,2])
df.shape
df2 = df.drop("Unnamed: 0", axis=1)
df2.rename(

    columns={

        "DATA INICIAL": "start_date",

        "DATA FINAL": "end_date",

        "REGIÃO": "region",

        "ESTADO": "state",

        "PRODUTO": "product",

        "NÚMERO DE POSTOS PESQUISADOS": "no_gas_stations",

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

df2.dtypes
for col in ['avg_price_margin', 'dist_avg_price', 'dist_sd_price', 'dist_min_price', 'dist_max_price', 'coef_dist']:

    df2[col] = pd.to_numeric(df2[col], errors='coerce')

df2.dtypes
df2.query('year!=2019 & product in ["GLV","GNV"]').groupby(['year', 'region'])['avg_price'].agg('sum')
# plot Regionwise Yearwise Average Price data

fig, ax = plt.subplots(figsize=(15,7))

df2.query('year!=2019 & product in ["GLV","GNV"]').groupby(['year','region']).sum()['avg_price'].unstack().plot(ax=ax)

plt.grid(True)
# plot Regionwise Yearwise Average Price Changes (% wise) data 

fig, ax = plt.subplots(figsize=(15,7))

df2.query('year!=2019 & product in ["GLV","GNV"]').groupby(['year','region']).sum()['avg_price'].pct_change().unstack().plot(ax=ax)

plt.grid(True)
df2.query('year!=2019 & region in ["NORDESTE"]').groupby(['year', 'state'])['avg_price'].agg('sum')
# plot Regionwise Yearwise Average Price data

fig, ax = plt.subplots(figsize=(15,7))

df2.query('year!=2019 & region in ["NORDESTE"]').groupby(['year', 'state'])['avg_price'].agg('sum').unstack().plot(ax=ax)

plt.grid(True)
# plot Regionwise Yearwise Average Price Changes (% wise) data 

fig, ax = plt.subplots(figsize=(15,7))

df2.query('year!=2019 & region in ["NORDESTE"]').groupby(['year', 'state'])['avg_price'].agg('sum').pct_change().unstack().plot(ax=ax)

plt.grid(True)
df2.query('year!=2019 & region in ["CENTRO OESTE"]').groupby(['year', 'state'])['avg_price'].agg('sum')
# plot Regionwise Yearwise Average Price data

fig, ax = plt.subplots(figsize=(15,7))

df2.query('year!=2019 & region in ["CENTRO OESTE"]').groupby(['year', 'state'])['avg_price'].agg('sum').unstack().plot(ax=ax)

plt.grid(True)
# plot Regionwise Yearwise Average Price Changes (% wise) data 

fig, ax = plt.subplots(figsize=(15,7))

df2.query('year!=2019 & region in ["CENTRO OESTE"]').groupby(['year', 'state'])['avg_price'].agg('sum').pct_change().unstack().plot(ax=ax)

plt.grid(True)
# plot Regionwise Yearwise Average Price Changes (% wise) data 

fig, ax = plt.subplots(figsize=(15,7))

df2.query('year!=2019').groupby(['year','region']).sum()['avg_price'].pct_change().unstack().plot(ax=ax)

plt.grid(True)
df2.query('year!=2019').groupby(['year', 'state', 'product'])['avg_price'].agg('sum')
# plot Statewise Yearwise Most Expensive Product (GLP) Price data 

fig, ax = plt.subplots(figsize=(15,7))

df2.query('year!=2019').groupby(['state','product'])['max_price'].agg('sum').unstack().plot(kind='bar',ax=ax)

plt.grid(True)
# plot Statewise Yearwise Most Expensive Product (GLP) Price data 

fig, ax = plt.subplots(figsize=(15,7))

df2.query('year!=2019').groupby(['state','product'])['min_price'].agg('sum').unstack().plot(kind='bar',ax=ax)

plt.grid(True)