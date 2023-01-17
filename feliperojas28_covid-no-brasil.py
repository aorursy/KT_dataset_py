import requests

from bs4 import BeautifulSoup

import pandas as pd

import matplotlib.pyplot as plt
page = requests.get('https://brasil.io/covid19/')
soup = BeautifulSoup(page.content, 'html.parser')
tabela = soup.find_all(class_='table mdl-data-table')

len(tabela)
tabela = tabela[0]
linhas = tabela.find_all('tr')

len(linhas)
lin = linhas[1].text

lin = lin.strip('\n ')

lin = lin.replace(' \n ', '\t')

lin = lin.replace('.', '')

lin = lin.replace(',', '.')

lin
def prep_linha(linha):

    linha = linha.get_text()

    linha = linha.strip('\n ')

    linha = linha.replace(' \n ', '\t')

    linha = linha.replace('.', '')

    linha = linha.replace(',', '.')

    return linha
tabela_final = 'Data\tCidade\tUF\tConfirmados\tConfirmados por 100k hab\tMortes\tLetalidade\tMortes por 100k hab\n'

for i in range(1,len(linhas)-1):

    linha = prep_linha(linhas[i])

    tabela_final += linha

    tabela_final += '\n'
arquivo = open('COVID por município', 'w')

arquivo.write(tabela_final)

arquivo.close()
df = pd.read_csv('COVID por município', sep = '\t')

df.head()
df_estados = df.groupby('UF').sum().drop(['Confirmados por 100k hab', 'Mortes por 100k hab'], inplace = False, axis = 1).sort_values(by = 'Confirmados', ascending = False)

df_estados
plt.rcParams['figure.figsize'] = (12,10)

plt.style.use('ggplot')

df_estados1 = df_estados.sort_values(by = 'Confirmados')

df_estados1.plot(kind = 'barh', y = 'Confirmados', legend = False)
plt.rcParams['figure.figsize'] = (12,10)

plt.style.use('ggplot')

df_estados1 = df_estados.sort_values(by = 'Mortes')

df_estados1.plot(kind = 'barh', y = 'Mortes', legend = False)
plt.rcParams['figure.figsize'] = (10,7)

plt.style.use('ggplot')

df_sp = df.loc[df['UF'] == 'SP']

df_sp = df_sp.drop('Data', inplace = False, axis = 1)

df_sp = df_sp.loc[df_sp['Mortes'] >= 100]

df_sp = df_sp.sort_values(by = 'Mortes por 100k hab')

df_sp.plot(kind = 'barh', x = 'Cidade', y = 'Mortes por 100k hab', grid = True, title = 'Óbitos por 100 mil habitantes nos municípios paulistas a partir de 100 mortes', legend = False)

plt.show()
df_sp.sort_values(by = 'Mortes', ascending = False).drop(['UF', 'Confirmados', 'Confirmados por 100k hab'], inplace = False, axis = 1)
plt.rcParams['figure.figsize'] = (10,7)

plt.style.use('ggplot')

df_rj = df.loc[df['UF'] == 'RJ']

df_rj = df_rj.loc[df_rj['Mortes'] >= 100]

df_rj = df_rj.sort_values(by = 'Mortes por 100k hab')

df_rj.plot(kind = 'barh', x = 'Cidade', y = 'Mortes por 100k hab', grid = True, title = 'Óbitos por 100 mil habitantes nos municípios cariocas a partir de 100 mortes', legend = False)

plt.show()
df_rj.sort_values(by = 'Mortes', ascending = False).drop(['UF', 'Confirmados', 'Confirmados por 100k hab'], inplace = False, axis = 1)