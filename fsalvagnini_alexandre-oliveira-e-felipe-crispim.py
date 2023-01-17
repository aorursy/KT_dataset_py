# Importa as libs necessárias

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns
df_par = pd.read_csv(r'../input/cota_parlamentar_sp.csv', delimiter = ';')
df_par.head()
df_par_partido = df_par.groupby('sgpartido').sum().reset_index()
gastos_5 = sorted(list(df_par_partido['vlrdocumento']), reverse = True)[:5]

partidos_5 = ['PT', 'PSDB', 'PP', 'DEM', 'PR']
plt.pie(np.array(gastos_5), labels = partidos_5)
sns.set(rc = {'figure.figsize' : (10, 10)})

sns.violinplot(x = df_par['sgpartido'], y = df_par['vlrdocumento'], order = partidos_5)

# Coloca informações no gŕafico para facilitar a compreenssão

plt.title('Distribuição dos gastos')

plt.xlabel('Partido')

plt.ylabel('Valor gasto')
df_PT = df_par.query('sgpartido == "PT"')
df_PT_grouped = df_PT.groupby('numano').sum().reset_index()

sns.set(rc = {'figure.figsize' : (10, 10)})

sns.lineplot(x="numano", y="vlrdocumento", data=df_PT_grouped)

plt.title('Gasto anual PT')

plt.xlabel('Ano')

plt.ylabel('Valor gasto')
df_PSDB = df_par.query('sgpartido == "PSDB"')
df_PSDB_grouped = df_PSDB.groupby('numano').sum().reset_index()

sns.set(rc = {'figure.figsize' : (10, 10)})

sns.lineplot(x="numano", y="vlrdocumento", data=df_PSDB_grouped)

plt.title('Gasto anual PSDB')

plt.xlabel('Ano')

plt.ylabel('Valor gasto')
df_PP = df_par.query('sgpartido == "PP"')
df_PP_grouped = df_PP.groupby('numano').sum().reset_index()

sns.set(rc = {'figure.figsize' : (10, 10)})

sns.lineplot(x="numano", y="vlrdocumento", data=df_PP_grouped)

plt.title('Gasto anual PP')

plt.xlabel('Ano')

plt.ylabel('Valor gasto')
df_DEM = df_par.query('sgpartido == "DEM"')
df_DEM_grouped = df_DEM.groupby('numano').sum().reset_index()

sns.set(rc = {'figure.figsize' : (10, 10)})

sns.lineplot(x="numano", y="vlrdocumento", data=df_DEM_grouped)

plt.title('Gasto anual DEM')

plt.xlabel('Ano')

plt.ylabel('Valor gasto')
df_PR = df_par.query('sgpartido == "PR"')
df_PR_grouped = df_PR.groupby('numano').sum().reset_index()

sns.set(rc = {'figure.figsize' : (10, 10)})

sns.lineplot(x="numano", y="vlrdocumento", data=df_PR_grouped)

plt.title('Gasto anual PR')

plt.xlabel('Ano')

plt.ylabel('Valor gasto')
df_par_nummes = df_par.groupby('nummes').sum().reset_index()
df_par_nummes
sns.set(rc = {'figure.figsize' : (10, 10)})

sns.barplot(x = 'nummes', y = 'vlrdocumento', data = df_par_nummes)

# Coloca informações no gŕafico para facilitar a compreenssão

plt.title('Gastos de acordo com o número de vezes que se elegeu')

plt.xlabel('Nº de vezes que se elegeu')

plt.ylabel('Valor gasto')
df_fornecedor = df_par.groupby('txtfornecedor').sum().reset_index()
values = list(df_fornecedor['vlrdocumento'].values)

sorted_values = sorted(values, reverse = True)

array_values = []

for v in sorted_values[:5]:

    array_values.append(df_fornecedor.iloc[values.index(v)].values)

    

df_5_fornecedores = pd.DataFrame(np.array(array_values))
df_5_fornecedores
sns.set(rc = {'figure.figsize' : (10, 5)})

sns.barplot(x = 0, y = 5, data = df_5_fornecedores)

# Coloca informações no gŕafico para facilitar a compreenssão

plt.title('Gastos com os 5 maiores fornecedores')

plt.xlabel('Fornecedor')

plt.ylabel('Valor gasto')