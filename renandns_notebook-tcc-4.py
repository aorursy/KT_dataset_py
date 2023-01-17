# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns
#Dados obtidos no intervalo entre os dias 02/06/2019 e 09/06/2019(somente dados que possuem informações dos 6 sensores)

telhado = pd.read_csv("../input/telhado-verde-6-sensores/telhado_verde_6_sensores.csv")

telhado.head(10)
#Dados obtidos no intervalo entre os dias 02/06/2019 e 09/06/2019(somente dados que possuem informações dos 6 sensores)

telhado_new = pd.read_csv("../input/telhado-verde-6-sensores-temp-colunas/telhado_verde_6_sensores_temp_colunas.csv")

telhado_new.head(10)
nRow, nCol = telhado.shape

print(f'Dataset Telhado Verde - Linhas: {nRow} / Colunas: {nCol}')
nRow, nCol = telhado.shape

print(f'Dataset Telhado Verde Temp Colunas - Linhas: {nRow} / Colunas: {nCol}')
telha1 = telhado[['data','temperatura']].groupby(['data']).count().sort_values(['temperatura'],ascending=False)

telha1.head(60)
telhado.sort_values(['temperatura'],ascending=False)
telhado['temperatura'].sort_values(ascending=True)
telhado.duplicated(subset=['data', 'sensor'], keep='first').sum()
telhado = telhado.drop_duplicates(keep='first')
nRow, nCol = telhado.shape

print(f'Dataset Telhado Verde - Linhas: {nRow} / Colunas: {nCol}')
telhado = telhado.sort_values(['data','sensor'])
telhado.head(20)
pd.DataFrame(telhado.groupby(['sensor']).count())
telhado['data'] = pd.to_datetime(telhado['data'])

# Coluna 'Data'

telhado['Data'] = telhado['data'].dt.strftime('%Y-%m-%d')

# Coluna 'Hora'

telhado['Hora'] = telhado['data'].dt.strftime('%H:%M:%S')

telhado['hora'] = telhado['data'].dt.strftime('%H')

telhado['minuto'] = telhado['data'].dt.strftime('%M')

telhado['segundo'] = telhado['data'].dt.strftime('%S')

telhado.head(10)
telhado_new['data'] = pd.to_datetime(telhado_new['data'])

# Coluna 'Data'

telhado_new['Data'] = telhado_new['data'].dt.strftime('%Y-%m-%d')

# Coluna 'Hora'

telhado_new['Hora'] = telhado_new['data'].dt.strftime('%H:%M:%S')

telhado_new['hora'] = telhado_new['data'].dt.strftime('%H')

telhado_new['minuto'] = telhado_new['data'].dt.strftime('%M')

telhado_new['segundo'] = telhado_new['data'].dt.strftime('%S')

telhado_new.head(10)
telhado['Data'] = pd.to_datetime(telhado['Data'])

telhado['hora'] = pd.to_numeric(telhado['hora'], downcast='integer')

telhado['minuto'] = pd.to_numeric(telhado['minuto'], downcast='integer')

telhado['segundo'] = pd.to_numeric(telhado['segundo'], downcast='integer')
telhado_new['Data'] = pd.to_datetime(telhado_new['Data'])

telhado_new['hora'] = pd.to_numeric(telhado_new['hora'], downcast='integer')

telhado_new['minuto'] = pd.to_numeric(telhado_new['minuto'], downcast='integer')

telhado_new['segundo'] = pd.to_numeric(telhado_new['segundo'], downcast='integer')
pd.DataFrame(telhado.dtypes)
telhado.head(10)
from datetime import datetime



def categoriza_SF(s):

    if s == 1 or s == 2:

        return 1

    else:

        return 0

    

def categoriza_SM(s):

    if s == 3 or s == 4:

        return 1

    else:

        return 0

    

def categoriza_SMD(s):

    if s == 5 or s == 6:

        return 1

    else:

        return 0

    

def categoriza_Dia(hora):

    if hora > 6 and hora < 18:

        return 1

    else:

        return 0

    

def categoriza_Noite(hora):

    if (hora >= 0 and hora <= 6) or (hora >= 18 and hora <= 23):

        return 1

    else:

        return 0

    

def categoriza_Verao(data):

    ano = data.year

    dt_from_1 = datetime.strptime('2018-12-21', '%Y-%m-%d')

    dt_to_1 = datetime.strptime('2018-03-19', '%Y-%m-%d')

    dt_from_2 = datetime.strptime('2019-12-21', '%Y-%m-%d')

    dt_to_2 = datetime.strptime('2019-03-19', '%Y-%m-%d')

    if ano == 2018:

        if data >= dt_from_1:

            return 1

        else:

            if data <= dt_to_1:

                   return 1

            else:

                return 0

    else:

        if ano == 2019:

            if data >= dt_from_2:

                return 1

            else:

                if data <= dt_to_2:

                       return 1

                else:

                    return 0

        else:

            return 0

        

def categoriza_Outono(data):

    ano = data.year

    dt_from_1 = datetime.strptime('2018-03-20', '%Y-%m-%d')

    dt_to_1 = datetime.strptime('2018-06-20', '%Y-%m-%d')

    dt_from_2 = datetime.strptime('2019-03-20', '%Y-%m-%d')

    dt_to_2 = datetime.strptime('2019-06-20', '%Y-%m-%d')

    if ano == 2018:

        if data >= dt_from_1 and data <= dt_to_1:

            return 1

        else:

            return 0

    else:

        if ano == 2019:

            if data >= dt_from_2 and data <= dt_to_2:

                return 1

            else:

                return 0

        else:

            return 0

        

def categoriza_Inverno(data):

    ano = data.year

    dt_from_1 = datetime.strptime('2018-06-21', '%Y-%m-%d')

    dt_to_1 = datetime.strptime('2018-09-21', '%Y-%m-%d')

    dt_from_2 = datetime.strptime('2019-06-21', '%Y-%m-%d')

    dt_to_2 = datetime.strptime('2019-09-21', '%Y-%m-%d')

    if ano == 2018:

        if data >= dt_from_1 and data <= dt_to_1:

            return 1

        else:

            return 0

    else:

        if ano == 2019:

            if data >= dt_from_2 and data <= dt_to_2:

                return 1

            else:

                return 0

        else:

            return 0

        

def categoriza_Primavera(data):

    ano = data.year

    dt_from_1 = datetime.strptime('2018-09-22', '%Y-%m-%d')

    dt_to_1 = datetime.strptime('2018-12-20', '%Y-%m-%d')

    dt_from_2 = datetime.strptime('2019-09-22', '%Y-%m-%d')

    dt_to_2 = datetime.strptime('2019-12-20', '%Y-%m-%d')

    if ano == 2018:

        if data >= dt_from_1 and data <= dt_to_1:

            return 1

        else:

            return 0

    else:

        if ano == 2019:

            if data >= dt_from_2 and data <= dt_to_2:

                return 1

            else:

                return 0

        else:

            return 0
telhado['Sistema_FLAT'] = telhado['sensor'].apply(categoriza_SF)

telhado['Sistema_Modular'] = telhado['sensor'].apply(categoriza_SM)

telhado['Sistema_MacDrain'] = telhado['sensor'].apply(categoriza_SMD)

#telhado['Verao'] = telhado['data'].apply(categoriza_Verao)

#telhado['Outono'] = telhado['data'].apply(categoriza_Outono)

#telhado['Inverno'] = telhado['data'].apply(categoriza_Inverno)

#telhado['Primavera'] = telhado['data'].apply(categoriza_Primavera)

#telhado['Dia'] = telhado['hora'].apply(categoriza_Dia)

#telhado['Noite'] = telhado['hora'].apply(categoriza_Noite)

#telhado.loc[telhado['sensor'] == 1 or telhado['sensor'] == 2, 'Sistema_FLAT'] = 1

telhado
telhado_new['Verao'] = telhado_new['data'].apply(categoriza_Verao)

telhado_new['Outono'] = telhado_new['data'].apply(categoriza_Outono)

telhado_new['Inverno'] = telhado_new['data'].apply(categoriza_Inverno)

telhado_new['Primavera'] = telhado_new['data'].apply(categoriza_Primavera)

telhado_new['Dia'] = telhado_new['hora'].apply(categoriza_Dia)

telhado_new['Noite'] = telhado_new['hora'].apply(categoriza_Noite)
#telhado['Dia'] = telhado['hora'].apply(categoriza_Dia)

#telhado['Noite'] = telhado['hora'].apply(categoriza_Noite)

#telhado
#telhado['Sistema_FLAT'] = telhado['sensor'] == 1 or telhado['sensor'] == 2 = 1

#telhado.loc[telhado['sensor'] == 1 or  telhado['sensor'] == 2, 'Sistema_FLAT'] = 1

#telhado
#Dados obtidos no intervalo entre os dias 01/01/2018 e 13/09/2019

clima = pd.read_csv('../input/dataset-estacao-e-telhado-full1/Dataset_Estacao_01_01_2018_a_13_09_2019.csv', delimiter=',')

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
clima['Data'] = pd.to_datetime(clima['Data'])

clima['hora'] = pd.to_numeric(clima['hora'], downcast='integer')

clima['minuto'] = pd.to_numeric(clima['minuto'], downcast='integer')

clima['segundo'] = pd.to_numeric(clima['segundo'], downcast='integer')
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

pd.DataFrame(telhado.groupby(['sensor'])['temperatura'].count())
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
#corr_telhado = telhado

#corr_te = corr_telhado.corr()

#sns.heatmap(corr_te, 

#            xticklabels=corr_te.columns.values,

#            yticklabels=corr_te.columns.values,

#            annot=True, fmt=".1f",

#           cmap=sns.diverging_palette(220, 10, as_cmap=True))

#plt.show()
#corr_clima = clima

#corr_cli = corr_clima.corr()

#fig, ax = plt.subplots(figsize=(10, 10))

#sns.heatmap(corr_cli, 

#            xticklabels=corr_cli.columns.values,

#            yticklabels=corr_cli.columns.values,

#            annot=True, fmt=".1f",

#           cmap=sns.diverging_palette(220, 10, as_cmap=True))

#plt.show()
#corr_clima = clima[['deficit_vapor_vmin','deficit_vapor_vmedio','radiacao_solar_valor','temperatura_ar_vmin','temperatura_ar_vmedio','temperatura_ar_vmax',]]

#corr_clima = clima[['deficit_vapor_vmedio','ponto_orvalho_vmedio', 'radiacao_solar_valor','temperatura_ar_vmedio', 'umidade_ar_vmedio',]]

#corr_cli = corr_clima.corr()

#fig, ax = plt.subplots(figsize=(10, 10))

#sns.heatmap(corr_cli, 

#            xticklabels=corr_cli.columns.values,

#            yticklabels=corr_cli.columns.values,

#            annot=True, fmt=".1f",

#           cmap=sns.diverging_palette(220, 10, as_cmap=True))

#plt.show()
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
media_clima = clima[['Data','hora','radiacao_solar_valor']].groupby(['Data', 'hora']).mean().reset_index()

media_clima.dropna(inplace=True)

media_clima
media_clima = clima[['hora','radiacao_solar_valor', 'temperatura_ar_vmedio']].groupby(['hora']).mean().reset_index()

media_clima.dropna(inplace=True)

media_clima
#sns.set()

#fig, ax = plt.subplots(figsize=(10, 5))

#ax = sns.lineplot(x="Data", y="radiacao_solar_valor", markers=True, dashes=False,data=media_clima)
sns.set()

fig, ax = plt.subplots(figsize=(10, 5))

ax = sns.lineplot(x="hora", y="radiacao_solar_valor", markers=True, dashes=False,data=media_clima)
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
sns.set()

telhado2 = telhado.dropna(inplace=False)

x = telhado2['temperatura']

ax = sns.distplot(x, bins=10)
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
result = pd.merge(telhado_new, clima, on=['Data', 'Data', 'hora', 'hora', 'minuto', 'minuto'])

result
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