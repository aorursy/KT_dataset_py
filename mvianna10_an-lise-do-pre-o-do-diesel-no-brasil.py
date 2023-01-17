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



import seaborn as sns # seaborn package

import matplotlib.pyplot as plt # matplotlib library

# abre dataset com preços de combustíveis

os.chdir('/kaggle/input/gas-prices-in-brazil/')

df = pd.read_csv('2004-2019.tsv', sep='\t',parse_dates=[1,2])

df.sample(5)

# verifica tipos de campo

df.info()
# Remoção de coluna indesejada

df2 = df.drop("Unnamed: 0", axis=1)



# Conversão de colunas numéricas que constam como object para os tipos adequados

col_numeros = ['UNIDADE DE MEDIDA', 'MARGEM MÉDIA REVENDA', 'PREÇO MÉDIO DISTRIBUIÇÃO', 

            'DESVIO PADRÃO DISTRIBUIÇÃO', 'PREÇO MÍNIMO DISTRIBUIÇÃO', 'PREÇO MÁXIMO DISTRIBUIÇÃO',

            'COEF DE VARIAÇÃO DISTRIBUIÇÃO']

for col in col_numeros:

    df2[col] = pd.to_numeric(df2[col], errors='coerce')

df2.dtypes
# Lista produtos

produtos = df2['PRODUTO'].unique()

for p in produtos: print(p)
# Identifica faixa de tempo

def ano_mes(data):  # formata para mês-ano

    am = str(data%100) + "-"+ str(data//100)

    return am



meses = df2['MÊS'] + df2['ANO']*100 

meses = meses.unique()

meses.sort()



print(ano_mes(meses[0]) + ' até '+ ano_mes(meses[-1]))

# Verifica valores Null

df2.info()
df2.sample(5)
# Carrega dataset de preços de óleo bruto

tea_oil = pd.read_csv('/kaggle/input/tea-and-oil-price-data-gem-database/export_dt.csv', index_col=0, parse_dates=True)

tea_oil['ANO'] = tea_oil.index.year

tea_oil['MÊS'] = tea_oil.index.month

tea_oil.head()

#cor = 'white' # eu uso um notebook com fundo escuro. Para fundo claro, remova o comentário abaixo

cor = 'black'



def muda_cor(graf,titulo):  # muda cores e título para um gráfico

    graf.xaxis.label.set_color(cor)

    graf.xaxis.label.set_color(cor)

    graf.tick_params(axis='x', colors=cor)

    graf.tick_params(axis='y', colors=cor)

    graf.set_title(titulo, color=cor)



    graf.grid(True)

    return 



# plot Regionwise Yearwise Average Price data

#plt.figsize=(39,15)

fig, axes  = plt.subplots(2, 2, sharey=True,figsize=(15,13))



df2.query('2012<=ANO<=2018 & PRODUTO in ["ÓLEO DIESEL"]').groupby(['ANO', 'REGIÃO'])['PREÇO MÉDIO REVENDA'].agg('mean').unstack().plot(ax=axes[0,0])

df2.query('2012<=ANO<=2018 & PRODUTO in ["ÓLEO DIESEL S10"]').groupby(['ANO', 'REGIÃO'])['PREÇO MÉDIO REVENDA'].agg('mean').unstack().plot(ax=axes[0,1])

df2.query('ANO!=2019 & PRODUTO in ["ÓLEO DIESEL"]').groupby(['ANO', 'REGIÃO'])['PREÇO MÉDIO REVENDA'].agg('mean').unstack().plot(ax=axes[1,0])

df2.query('ANO!=2019 & PRODUTO in ["ÓLEO DIESEL S10"]').groupby(['ANO'])['PREÇO MÉDIO REVENDA'].agg('mean').plot(ax=axes[1,1])

df2.query('ANO!=2019 & PRODUTO in ["ÓLEO DIESEL"]').groupby(['ANO'])['PREÇO MÉDIO REVENDA'].agg('mean').plot(ax=axes[1,1])



#configura cada gráfico

muda_cor(axes[0,0], 'ÓLEO DIESEL 2012-2018')

muda_cor(axes[0,1], 'ÓLEO DIESEL S10 2012-2018')

muda_cor(axes[1,0], 'ÓLEO DIESEL 2004-2018')

muda_cor(axes[1,1], 'ÓLEO DIESEL S10 x ÓLEO DIESEL 2004-2018')



#plt.grid(True)

#ax1.grid(True)

#for axis in axes: axis.grid(True)

#axes
fig, ax1  = plt.subplots(figsize=(15,7))





df2.query('2004<=ANO<=2018 & PRODUTO in ["ÓLEO DIESEL"]').groupby(['ANO'])['PREÇO MÉDIO REVENDA'].agg('mean').plot(ax=ax1, color ='green')

ax2 = ax1.twinx()

df2.query('2004<=ANO<=2018 & PRODUTO in ["ÓLEO DIESEL"]').groupby(['ANO'])['PREÇO MÉDIO DISTRIBUIÇÃO'].agg('mean').plot(ax=ax1, color ='blue')

ax2 = ax1.twinx()

tea_oil.query('2004<=ANO<=2018').groupby(['ANO'])['OIL'].agg('mean').plot(ax=ax2, color ='red')





muda_cor(ax1, 'DIESEL x ÓLEO BRUTO')

muda_cor(ax2, '')