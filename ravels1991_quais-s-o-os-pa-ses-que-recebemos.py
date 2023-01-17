import time

start_time = time.time()
!pip install bar_chart_race



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

import requests

from bs4 import BeautifulSoup



import bar_chart_race as bcr

from IPython.display import Video



plt.style.use('ggplot')



%matplotlib inline
df = pd.read_excel('../input/brasil-solicitaes-de-refugiado/solicitacoes-de-reconhecimento-da-condicao-de-refugiado.xlsx',

                  encoding='latin1')

df.head()
df.shape
df.dtypes
df['Mês/Ano'] = pd.to_datetime(df['Mês/Ano'].str.replace('/', ''), format='%m%Y', errors='coerce')

df['Year'] = df['Mês/Ano'].dt.year
df.groupby('Year')['Quantidade'].sum().plot(kind='bar', rot=45, figsize=(18,6), color='skyblue', title='Números de Refugiados por ano');
df_paises = df.groupby(['Year', 'Nacionalidade'])['Quantidade'].sum().unstack()

top_10 = df.groupby('Nacionalidade')['Quantidade'].sum().sort_values(ascending=False)[:10]

df_paises[top_10.index.to_list()].plot(figsize=(18,8), linewidth=2, alpha=0.7)

plt.title('Países com maior número de refugiados')

plt.xticks(np.arange(2000, 2020, 1))

plt.xlim(2000, 2019);
df_paises[top_10.index.to_list()[1:-1]].plot(figsize=(18,8), linewidth=2, alpha=0.7)

plt.title('Países com maior número de refugiados(Sem Venezuela)')

plt.xticks(np.arange(2000, 2020, 1))

plt.xlim(2000, 2019);
df_paises[top_10.index.to_list()].plot(figsize=(20,8), linewidth=2, alpha=0.7)

plt.title('Países com maior número de refugiados')

plt.xticks(np.arange(2000, 2020, 1))

plt.xlim(2000, 2019)

plt.axvline(x=2010, linestyle='--', color='skyblue')

plt.axvline(x=2011, linestyle='--', color='red')

plt.axvline(x=2016, linestyle='--', color='violet')

plt.text(2007.8, 30000, 'Terremoto no Haiti', va='center', bbox=dict(facecolor='skyblue', alpha=0.5), fontsize=13)

plt.text(2011.1, 20000, 'Guerra na Síria', va='center', bbox=dict(facecolor='red', alpha=0.5), fontsize=13)

plt.text(2013.7, 40000, 'Crise na Venezuela', va='center', bbox=dict(facecolor='violet', alpha=0.5), fontsize=13);
df_paises[['HAITI', 'SÍRIA']].plot(figsize=(20,8), linewidth=2, alpha=0.7, rot=45)

plt.title('Refugiados Haiti e Síria')

plt.xticks(np.arange(2000, 2020, 1))

plt.xlim(2000, 2019)

plt.axvline(x=2010, linestyle='--', color='red')

plt.axvline(x=2011, linestyle='--', color='skyblue')

plt.text(2007.8, 3000, 'Terremoto no Haiti', va='center', bbox=dict(facecolor='red', alpha=0.5), fontsize=13)

plt.text(2011.1, 2000, 'Guerra na Síria', va='center', bbox=dict(facecolor='skyblue', alpha=0.5), fontsize=13);
SUL = ['SC', 'RS', 'PR']

SUDESTE = ['MG', 'RJ', 'SP', 'ES']

CENTRO_OESTE = ['DF', 'GO', 'MT', 'MS' ]

NORDESTE = ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE']

NORTE = ['AC', 'AP', 'AM', 'PA', 'RN', 'RR', 'TO']
def regiao(row):

    if row in SUL:

        return 'SUL'

    if row in SUDESTE:

        return 'SUDESTE'

    if row in CENTRO_OESTE:

        return 'CENTRO_OESTE'

    if row in NORDESTE:

        return 'NORDESTE'

    if row in NORTE:

        return 'NORTE'
df['REGIAO'] = df['UF'].apply(regiao)

df.head()
regiao_sul = df.query('REGIAO == "SUL"')

regiao_sudeste = df.query('REGIAO == "SUDESTE"')

regiao_norte = df.query('REGIAO == "NORTE"')

regiao_nordeste = df.query('REGIAO == "NORDESTE"')

regiao_centro_oeste = df.query('REGIAO == "CENTRO_OESTE"')
def df_plot(df, title='title'):

    regiao = df.groupby(['Nacionalidade', 'Year'])['Quantidade'].sum().unstack()

    mask = regiao.T.sum().sort_values(ascending=False)[:5]

    ax = regiao.T[mask.index].plot(figsize=(18,8), rot=45, linewidth=2, alpha=0.8, title=title)

    plt.xticks(np.arange(2000, 2020, 1))

    plt.xlim(2000, 2019)

    return ax
df_plot(regiao_sul, 'Países com mais refugiados no Sul');
df_plot(regiao_sudeste, 'Países com mais refugiados no Sudeste');
df_plot(regiao_norte, 'Países com mais refugiados no Norte');
df_plot(regiao_nordeste, 'Países com mais refugiados no Nordeste');
df_plot(regiao_centro_oeste, 'Países com mais refugiados no Centro Oeste');
wiki_url = 'https://pt.wikipedia.org/wiki/Compara%C3%A7%C3%A3o_entre_c%C3%B3digos_de_pa%C3%ADses_COI,_FIFA,_e_ISO_3166'

response = requests.get(wiki_url)

soup = BeautifulSoup(response.text, 'xml')



table = soup.find('table',{'class': "wikitable sortable"})

wiki = pd.read_html(str(table), header=0)[0]

wiki.head()
wiki['País'] = wiki['País'].str.upper()

wiki.drop('Bandeira', axis=1, inplace=True)

wiki.rename(columns={'País': 'Nacionalidade'}, inplace=True)

wiki.head()
merge_df = pd.DataFrame.merge(df, wiki, on='Nacionalidade')

merge_df.head()
quantidade = merge_df.groupby(["Nacionalidade", "Year", 'ISO-3'])['Quantidade'].sum().reset_index()

mundo = quantidade.sort_values(by='Year').reset_index()

sem_venezuela = mundo[mundo['Nacionalidade'] != 'VENEZUELA']
fig = px.choropleth(mundo, locations="ISO-3", color="Quantidade", hover_name="Nacionalidade", animation_frame="Year", 

                    projection="natural earth", title="Refugiados Globo")

fig.show()
fig = px.choropleth(sem_venezuela, locations="ISO-3", color="Quantidade", hover_name="Nacionalidade", animation_frame="Year", 

                    projection="natural earth", title='Refugiados Globo (Sem Venezuela)')

fig.show()
sem_venezuela = sem_venezuela.groupby(['Nacionalidade', 'ISO-3'])['Quantidade'].sum().reset_index()

fig = px.choropleth(sem_venezuela, locations="ISO-3", color="Quantidade", hover_name="Nacionalidade", 

                    projection="natural earth", title='Refugiados Globo (Sem Venezuela)' )

fig.show()
paises = df.groupby(['Mês/Ano', 'Nacionalidade'])['Quantidade'].sum().reset_index()

paises = paises.pivot_table(index='Mês/Ano', columns='Nacionalidade', values='Quantidade').fillna(0).cumsum()

top_50 = df.groupby('Nacionalidade')['Quantidade'].sum().sort_values(ascending=False)[:50]

paises.head()
paises_bar_chart = bcr.bar_chart_race(

    df=paises[top_50.index],

    filename=None,

    period_length=500,

    figsize=(7, 4),

    n_bars=10,

    bar_size=.5,

    filter_column_colors=True,

    period_fmt= '%B %d, %Y',

    title='Solicitação de Refúgio por País')
paises_bar_chart
print(f"This kernel took {(time.time() - start_time)/60:.2f} minutes to run")