# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/tcc-facens-telhado-verde/"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns
#Dados obtidos no intervalo entre os dias 02/06/2019 e 09/06/2019

telhado = pd.read_csv('../input/telhado2/telhado 2.6 ao 9.6_2.csv', delimiter=';')

#telhado = pd.read_csv('../input/tcc-facens-telhado-verde/telhado_2.6_ao_9.6.csv', delimiter=';')

telhado.head(10)
telhado_pv = telhado.pivot_table('temperatura', ['data'], 'sensor')
telhado = pd.DataFrame(telhado_pv.to_records())

telhado.rename(columns={'1': 'temp_sensor1', 

                        '2': 'temp_sensor2',

                        '3': 'temp_sensor3',

                        '4': 'temp_sensor4',

                        '5': 'temp_sensor5',

                        '6': 'temp_sensor6'}, inplace=True)
telhado
telhado['data'] = pd.to_datetime(telhado['data'])

# Coluna 'Data'

telhado['Data'] = telhado['data'].dt.strftime('%Y-%m-%d')

# Coluna 'Hora'

telhado['Hora'] = telhado['data'].dt.strftime('%H:%M:%S')

telhado['hora'] = telhado['data'].dt.strftime('%H')

telhado['minuto'] = telhado['data'].dt.strftime('%M')

telhado['segundo'] = telhado['data'].dt.strftime('%S')

telhado = telhado.drop_duplicates(keep='first')

telhado.head(10)
#Dados obtidos no intervalo entre os dias 02/06/2019 e 09/06/2019

clima = pd.read_csv('../input/tcc-facens-telhado-verde/estacao_2.6_ao_9.6.csv', delimiter=';')

clima.head(10)
clima['data'] = pd.to_datetime(clima['data'])

# Coluna 'Data'

clima['Data'] = clima['data'].dt.strftime('%Y-%m-%d')

# Coluna 'Hora'

clima['Hora'] = clima['data'].dt.strftime('%H:%M:%S')

clima['hora'] = clima['data'].dt.strftime('%H')

clima['minuto'] = clima['data'].dt.strftime('%M')

clima['segundo'] = clima['data'].dt.strftime('%S')

clima = clima.drop_duplicates(keep='first')

clima.tail(10)
pd.DataFrame(telhado.dtypes)
pd.DataFrame(clima.dtypes)
# Verificando as estatisticas de cada coluna

telhado.describe()
# Verificando as estatisticas de cada coluna

clima.describe()
telhado.shape
nRow, nCol = telhado.shape

print(f'Dataset Telhado Verde - Linhas: {nRow} / Colunas: {nCol}')
clima.shape
nRow, nCol = clima.shape

print(f'Dataset Estação Metereológica - Linhas: {nRow} / Colunas: {nCol}')
# Verificando a quantidade de valores faltantes por coluna

telhado.isnull().sum().sort_values(ascending=False)
#pd.DataFrame(telhado.isnull().groupby(['sensor']).sum())

#pd.DataFrame(telhado.groupby(['sensor'])['temperatura'].count())
# Verificando a quantidade de valores faltantes por coluna

clima.isnull().sum().sort_values(ascending=False)
# Verificando a porcetagem de valores faltantes por coluna para definir como contorna-los

atributos_missing = []



for f in telhado.columns:

    missings = telhado[f].isnull().sum()

    if missings > 0:

        atributos_missing.append(f)

        missings_perc = missings/telhado.shape[0]

        

        print('Atributo {} tem {} amostras ({:.2%}) com valores faltantes'.format(f, missings, missings_perc))



print('No total, há {} atributos com valores faltantes'.format(len(atributos_missing)))
corr_telhado = telhado

corr_te = corr_telhado.corr()

sns.heatmap(corr_te, 

            xticklabels=corr_te.columns.values,

            yticklabels=corr_te.columns.values,

            annot=True, fmt=".1f",

           cmap=sns.diverging_palette(220, 10, as_cmap=True))

plt.show()
corr_clima = clima

corr_cli = corr_clima.corr()

fig, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(corr_cli, 

            xticklabels=corr_cli.columns.values,

            yticklabels=corr_cli.columns.values,

            annot=True, fmt=".1f",

           cmap=sns.diverging_palette(220, 10, as_cmap=True))

plt.show()
#corr_clima = clima[['deficit_vapor_vmin','deficit_vapor_vmedio','radiacao_solar_valor','temperatura_ar_vmin','temperatura_ar_vmedio','temperatura_ar_vmax',]]

corr_clima = clima[['deficit_vapor_vmedio','ponto_orvalho_vmedio', 'radiacao_solar_valor','temperatura_ar_vmedio', 'umidade_ar_vmedio',]]

corr_cli = corr_clima.corr()

fig, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(corr_cli, 

            xticklabels=corr_cli.columns.values,

            yticklabels=corr_cli.columns.values,

            annot=True, fmt=".1f",

           cmap=sns.diverging_palette(220, 10, as_cmap=True))

plt.show()
def plot_corr(corr):

    # Cortaremos a metade de cima pois é o espelho da metade de baixo

    fig, ax = plt.subplots(figsize=(10, 10))

    mask = np.zeros_like(corr, dtype=np.bool)

    mask[np.triu_indices_from(mask, 1)] = True



    sns.heatmap(corr, mask=mask, cmap='RdBu', square=True, linewidths=.5, annot=True, fmt=".1f")



# Calculando a correlação

corr = corr_clima.corr() 

plot_corr(corr)
#ax = sns.lineplot(x="Data", y="temperatura", hue='sensor', data=telhado.head(10))
#media_telhado = telhado[['sensor','Data','temperatura']].groupby(['sensor','Data']).mean().reset_index()

#media_telhado.dropna(inplace=True)

#media_telhado
media_clima = clima[['Data','radiacao_solar_valor']].groupby(['Data']).mean().reset_index()

media_clima.dropna(inplace=True)

media_clima
sns.set()

fig, ax = plt.subplots(figsize=(10, 5))

ax = sns.lineplot(x="Data", y="radiacao_solar_valor", markers=True, dashes=False,data=media_clima)
media_clima2 = clima[['Data','temperatura_ar_vmedio']].groupby(['Data']).mean().reset_index()

media_clima2.dropna(inplace=True)

media_clima2
sns.set()

fig, ax = plt.subplots(figsize=(10, 5))

ax = sns.lineplot(x="Data", y="temperatura_ar_vmedio", markers=True, dashes=False,data=media_clima2)
sns.set()

#x = clima['radiacao_solar_valor']

x1 = clima['temperatura_ar_vmedio']

ax = sns.distplot(x1)
sns.set()

x2 = clima['radiacao_solar_valor']

ax = sns.distplot(x2, bins=10)
#sns.set()

#telhado2 = telhado.dropna(inplace=False)

#x = telhado2['temperatura']

#ax = sns.distplot(x, bins=10)
sns.set()

x3 = clima['deficit_vapor_vmedio']

ax = sns.distplot(x3, bins = 10)
sns.set()

x4 = clima['ponto_orvalho_vmedio']

ax = sns.distplot(x4, bins = 10)
sns.set()

x = clima['temperatura_ar_vmedio']

ax = sns.distplot(x, bins = 10)
sns.set()

x5 = clima['umidade_ar_vmedio']

ax = sns.distplot(x5, bins = 10)
sns.set(style="white", palette="muted", color_codes=True)



# Set up the matplotlib figure

f, axes = plt.subplots(3, 2, figsize=(15, 15), sharex=True)

sns.despine(left=True)

x1 = clima['temperatura_ar_vmedio']

sns.distplot(x1, bins = 5, ax=axes[0, 0])



x2 = clima['radiacao_solar_valor']

sns.distplot(x2, bins=30, ax=axes[0, 1])



x3 = clima['deficit_vapor_vmedio']

sns.distplot(x3, bins = 30, ax=axes[1, 0])



x4 = clima['ponto_orvalho_vmedio']

sns.distplot(x4, bins = 30, ax=axes[1, 1])



x5 = clima['umidade_ar_vmedio']

sns.distplot(x5, bins = 30, ax=axes[2, 0])



#plt.setp(axes, yticks=[])

plt.tight_layout()
#result = pd.merge(media_telhado, media_clima2, on=['Data', 'Data'])

#result
media_clima3 = clima[['Data','deficit_vapor_vmedio','ponto_orvalho_vmedio', 'radiacao_solar_valor','temperatura_ar_vmedio', 'umidade_ar_vmedio']].groupby(['Data']).mean().reset_index()

media_clima3.dropna(inplace=True)

media_clima3
#result = pd.merge(media_telhado, media_clima3, on=['Data', 'Data'])

#result
#corr_clima = result[['temperatura','deficit_vapor_vmedio','ponto_orvalho_vmedio', 'radiacao_solar_valor','temperatura_ar_vmedio', 'umidade_ar_vmedio',]]

#corr_cli = corr_clima.corr()

#fig, ax = plt.subplots(figsize=(10, 10))

#sns.heatmap(corr_cli, 

#            xticklabels=corr_cli.columns.values,

#            yticklabels=corr_cli.columns.values,3

#            annot=True, fmt=".1f",

#           cmap=sns.diverging_palette(220, 10, as_cmap=True))

#plt.show()