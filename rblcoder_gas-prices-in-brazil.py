import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import seaborn as sns   

import matplotlib.pyplot as plt
df_gas_price = pd.read_csv('/kaggle/input/gas-prices-in-brazil/2004-2019.tsv', sep='\t')
df_gas_price.info()
df_gas_price.head(2).T
plt.subplots(figsize=(16, 7))

ax = sns.boxplot(x="REGIÃO", y="PREÇO MÉDIO REVENDA", data=df_gas_price)

ax.set_yscale('log')
plt.subplots(figsize=(16, 7))

ax = sns.boxplot(x="PRODUTO", y="PREÇO MÉDIO REVENDA", data=df_gas_price)

ax.set_yscale('log')
plt.subplots(figsize=(16, 7))

ax = sns.boxplot(x="ANO", y="PREÇO MÉDIO REVENDA", data=df_gas_price)

ax.set_yscale('log')
plt.subplots(figsize=(16, 7))

ax = sns.boxplot(x="PRODUTO", y="PREÇO MÉDIO REVENDA",hue="MÊS", data=df_gas_price)

ax.set_yscale('log')
import plotly.express as px

fig = px.scatter(df_gas_price, x="ANO", y="PREÇO MÉDIO REVENDA", facet_row="PRODUTO",

                width=1000, height=1300)



fig.show()
import plotly.express as px

fig = px.scatter(df_gas_price, x="ANO", y="PREÇO MÉDIO REVENDA", facet_row="PRODUTO", facet_col='REGIÃO',

                width=1000, height=1300)



fig.show()