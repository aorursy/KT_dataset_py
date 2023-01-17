# Download das páginas utilizadas para raspagem

import requests



# Responsável por nos mostrar o conteúdo HTML

from bs4 import BeautifulSoup



# Responsável por ler o arquivo .csv e nos fornecer determinados tipos de análise

import pandas as pd



# Expressões regulares necessárias para tratamento dos dados

import re



# Biblioteca que compreende a construção de gráficos

import matplotlib.pyplot as plt
# URL detentora dos dados de EBOLA

local = 'https://github.com/imdevskp/ebola_outbreak_dataset/blob/master/ebola_2014_2016_clean.csv'



# A biblioteca requests realiza o download

html_pagina = requests.get(local)
# soup nos disponiliza a obtenção do conteúdo HTML

soup = BeautifulSoup(html_pagina.content, 'html.parser')
# Com a inspeção dos elementos da página, a classe que contém a tabela é encontrada pelo soup

tabela = soup.find_all(class_='js-csv-data csv-data js-file-line-container')



# O comprimento da lista nos confirma que há só um elemento, ou seja, a tabela preterida deve ter sido corretamente fornecida

len(tabela)
# Como há uma lista, precisamos confirmar qual elemento será usado. Neste caso, como só há um, utilizaremos o elemento [0]

tabela = tabela[0]
# Procurando as tags 'tr' na tabela, achamos cada linha da tabela

linhas = tabela.find_all('tr')



# O comprimento de 2485 nos confirma o sucesso

len(linhas)
linha = linhas[7].text.strip('\n')

linha = linha.replace('\n', '\t')

linha

type(linha)


def prep_linha(linha):

    linha = linha.get_text()

    linha = linha.strip('\n')

    linha = linha.replace('\n', '\t')

    return linha
tabela_final = 'País ou região\tData\tTotal de casos\tTotal de mortes\n'

for i in range(1,len(linhas)-1):

    linha = prep_linha(linhas[i])

    tabela_final += linha

    tabela_final += '\n'
arquivo = open('Lista Ebola.csv', 'w')

arquivo.write(tabela_final)

arquivo.close()
ebola_df = pd.read_csv('Lista Ebola.csv', sep = '\t')

ebola_df.head()
# URL detentora dos dados de H1N1

local_h1n1 = "https://en.wikipedia.org/wiki/2009_swine_flu_pandemic_by_country"



# A biblioteca requests realiza o download

html_h1n1 = requests.get(local_h1n1)
# soup_h1n1 nos disponiliza a obtenção do conteúdo HTML

soup_h1n1 =  BeautifulSoup(html_h1n1.content, 'html.parser')
# Com a inspeção dos elementos da página, a classe que contém a tabela é encontrada pelo soup_h1n1

tabela_h1n1 = soup_h1n1.find_all(class_="navbox")



# O comprimento da lista nos traz 3, já que há três tabelas na página com esta identificação.

len(tabela_h1n1)
# Analisando os três elementos através da inspeção, vemos que a primeira tabela demonstrada é a que utilizaremos

tabela_h1n1 = tabela_h1n1[0]
# Procurando as tags 'tr' na tabela, achamos cada linha da tabela

linhas_h1n1 = tabela_h1n1.find_all('tr')



# O comprimento de 223 nos confirma o sucesso

len(linhas_h1n1)
def prep_linha_h1n1(linha_h1n1):

    linha_h1n1 = linha_h1n1.get_text()

    linha_h1n1 = linha_h1n1.strip('\n')

    linha_h1n1 = linha_h1n1.replace('\n', '\t')

    linha_h1n1 = linha_h1n1.replace(',', '')

    linha_h1n1 = linha_h1n1.replace('(', '').replace(')', '')

    linha_h1n1 = linha_h1n1.replace('\tR', '').replace('\tW', '').replace('\tS', '').replace('\tL', '').replace('\tN', '')

    linha_h1n1 = linha_h1n1.replace('\t***', '').replace('\t**', '').replace('\t*', '')

    linha_h1n1 = linha_h1n1.replace('\tlow2','').replace('\tlow', '').replace('\tmod','')

    linha_h1n1 = linha_h1n1.replace('\t=','').replace('\t-','').replace('\t+', "")

    linha_h1n1 = linha_h1n1.replace('~','').replace('^','').replace('#','').replace('+','')

    linha_h1n1 = linha_h1n1.replace('\t'*8, '\t').replace('\t'*7, '\t').replace('\t'*6, '\t').replace('\t'*5,'\t').replace('\t'*4, '\t').replace('\t'*3,'\t').replace('\t'*2, '\t')

    linha_h1n1 = re.sub(r'\[...\]', '', linha_h1n1)

    linha_h1n1 = re.sub(r'\[..\]', '', linha_h1n1)

    linha_h1n1 = re.sub(r'\[.\]', '', linha_h1n1)

    linha_h1n1 = linha_h1n1.replace('\t\t', '\t')



    return linha_h1n1

prep_linha_h1n1(linhas_h1n1[61])
tabela_final_h1n1 = 'País ou região\tCasos\tMortes\n'

for i in range(4,62):

    linha_h1n1 = prep_linha_h1n1(linhas_h1n1[i])

    tabela_final_h1n1 += linha_h1n1

    tabela_final_h1n1 += '\n'
arquivo_h1n1 = open('Lista H1N1.csv', 'w')

arquivo_h1n1.write(tabela_final_h1n1)

arquivo_h1n1.close()
h1n1_df = pd.read_csv('Lista H1N1.csv', sep = '\t')

h1n1_df.head()
# URL detentora dos dados de SARS

local_sars = 'https://en.wikipedia.org/wiki/2002%E2%80%932004_SARS_outbreak'



# A biblioteca requests realiza o download

html_sars = requests.get(local_sars)
# soup_sars nos disponiliza a obtenção do conteúdo HTML

soup_sars = BeautifulSoup(html_sars.content, 'html.parser')
# Com a inspeção dos elementos da página, a classe que contém a tabela é encontrada pelo soup_sars

tabela_sars = soup_sars.find_all(class_="wikitable sortable")



# O comprimento da lista nos confirma que há só um elemento, ou seja, a tabela preterida deve ter sido corretamente fornecida

len(tabela_sars)
# Como há uma lista, precisamos confirmar qual elemento será usado. Neste caso, como só há um, utilizaremos o elemento [0]

tabela_sars = tabela_sars[0]
# Procurando as tags 'tr' na tabela, achamos cada linha da tabela

linhas_sars = tabela_sars.find_all('tr')



# O comprimento de 35 nos confirma o sucesso

len(linhas_sars)
lin = linhas_sars[1].text

lin = lin.strip('\n').strip('\xa0')

lin = lin.replace('\n', '\t')

lin = re.sub(r'\[.*\]', '', lin)

lin = lin.replace(',', '')

lin
def prep_linha_sars(linha_sars):

    linha_sars = linha_sars.text

    linha_sars = linha_sars.strip('\n').strip('\xa0')

    linha_sars = linha_sars.replace('\n', '\t')

    linha_sars = re.sub(r'\[.*\]', '', linha_sars)

    linha_sars = linha_sars.replace(',', '')

    return linha_sars

prep_linha_sars(linhas_sars[3])
tabela_final_sars = 'País\tCasos\tMortes\tCurados\n'

for i in range(1,len(linhas_sars)-3):

    linha_sars = prep_linha_sars(linhas_sars[i])

    tabela_final_sars += linha_sars

    tabela_final_sars += '\n'
arquivo_sars = open('Lista SARS.csv', 'w')

arquivo_sars.write(tabela_final_sars)

arquivo_sars.close()
sars_df = pd.read_csv('Lista SARS.csv', '\t')

sars_df = sars_df.drop(columns=['Curados'])

sars_df.head()
# URL detentora dos dados de COVID-19

local_covid = 'https://github.com/datasets/covid-19/blob/master/data/countries-aggregated.csv'



# A biblioteca requests realiza o download

html_covid = requests.get(local_covid)
# soup_covid nos disponiliza a obtenção do conteúdo HTML

soup =  BeautifulSoup(html_covid.content, 'html.parser')
# Com a inspeção dos elementos da página, a classe que contém a tabela é encontrada pelo soup

tabela_covid = soup.find_all(class_='highlight tab-size js-file-line-container')



# O comprimento da lista nos confirma que há só um elemento, ou seja, a tabela preterida deve ter sido corretamente fornecida

len(tabela_covid)
# Como há uma lista, precisamos confirmar qual elemento será usado. Neste caso, como só há um, utilizaremos o elemento [0]

tabela_covid = tabela_covid[0]
# Procurando as tags 'tr' na tabela, achamos cada linha da tabela

linhas = tabela_covid.find_all('tr')



# O comprimento de 25381 nos confirma o sucesso

len(linhas)
linha = linhas[6].text.strip('\n')

linha = linha.replace('\n', '\t')

linha
def prep_linha(linha):

    linha = linha.get_text()

    linha = linha.strip('\n')

    linha = linha.replace('\n', '\t')

    return linha
tabela_final = 'Data,País,Confirmados,Recuperados,Mortos\n'

for i in range(1,len(linhas)):

    linha = prep_linha(linhas[i])

    tabela_final += linha

    tabela_final += '\n'
arquivo = open('lista-covid.csv', 'w')

arquivo.write(tabela_final)

arquivo.close()
covid_df = pd.read_csv('lista-covid.csv', sep = ',')

covid_df.head()
# Localizando os dados do último dia em que foram coletados

ebola_final = ebola_df.loc[ebola_df['Data'] == '2016-03-23']



# Somando os totais de casos e mortes e criando uma variável para armazená-los

totalcasos_ebola = int(ebola_final['Total de casos'].sum())

totalmortes_ebola = int(ebola_final['Total de mortes'].sum())



# Dividindo mortes por casos, para retornar a letalidade e armazenando-a também em uma variável

letalidade_ebola = float(totalmortes_ebola/totalcasos_ebola)
# Somando os totais de casos e mortes e criando uma variável para armazená-los

totalcasos_h1n1 = h1n1_df['Casos'].sum()

totalmortes_h1n1 = h1n1_df['Mortes'].sum()



# Dividindo mortes por casos, para retornar a letalidade e armazenando-a também em uma variável

letalidade_h1n1 = float(totalmortes_h1n1/totalcasos_h1n1)
# Somando os totais de casos e mortes e criando uma variável para armazená-los

totalcasos_sars = sars_df['Casos'].sum()

totalmortes_sars = sars_df['Mortes'].sum()



# Dividindo mortes por casos, para retornar a letalidade e armazenando-a também em uma variável

letalidade_sars = float(totalmortes_sars/totalcasos_sars)
# Localizando os dados do último dia em que foram coletados

covid_final = covid_df.loc[covid_df['Data'] == '2020-06-07']



# Somando os totais de casos e mortes e criando uma variável para armazená-los

totalcasos_covid = covid_final['Confirmados'].sum()

totalmortes_covid = covid_final['Mortos'].sum()



# Dividindo mortes por casos, para retornar a letalidade e armazenando-a também em uma variável

letalidade_covid = float(totalmortes_covid/totalcasos_covid)
tabela_epidemias = pd.DataFrame({

    'Epidemia': ['COVID-19', 'H1N1', 'Ebola', 'SARS-Cov-1'],

    'Total de casos': [totalcasos_covid, totalcasos_h1n1, totalcasos_ebola, totalcasos_sars],

    'Total de mortes': [totalmortes_covid, totalmortes_h1n1, totalmortes_ebola, totalmortes_sars],

    'Letalidade': [letalidade_covid, letalidade_h1n1, letalidade_ebola, letalidade_sars]

})

    

tabela_epidemias
# Esta tabela incluirá somente a letalidade das epidemias

tabela_letalidade = pd.DataFrame({

    'Epidemia': ['COVID-19', 'H1N1', 'Ebola', 'SARS-Cov-1'],

    'Letalidade': [letalidade_covid, letalidade_h1n1, letalidade_ebola, letalidade_sars]

})

    

tabela_letalidade
# Estilo do gráfico

plt.style.use('ggplot')

# Tamanho do gráfico

plt.rcParams['figure.figsize'] = (8,5)



# Utilizamos .set_index() para inverter linha e coluna do DataFrame original

# Utilizamos .plot() para que plt encontre os valores e os coloque em gráfico

# kind = 'bar' traz um gráfico de barras

# logy = True coloca o eixo y em escala logarítmica

# title = <input> aplica o título ao gráfico

tabela_epidemias.set_index('Epidemia').T.plot(kind = 'bar', logy = True, title = 'Casos, mortes e letalidade em escala logarítmica')
# Estilo do gráfico

plt.style.use('ggplot')

# Tamanho do gráfico

plt.rcParams['figure.figsize'] = (8,5)



# Utilizamos .set_index() para inverter linha e coluna do DataFrame original

# Utilizamos .plot() para que plt encontre os valores e os coloque em gráfico

# kind = 'bar' traz um gráfico de barras

# logy = True coloca o eixo y em escala logarítmica

# title = <input> aplica o título ao gráfico

tabela_letalidade.set_index('Epidemia').T.plot(kind = 'bar', title = 'Taxa de letalidade')
# Tamanho do gráfico

plt.rcParams['figure.figsize'] = (5,3)



# Determinando o DataFrame para mostrar apenas os dados do último dia de coleta

ebola_mortes = ebola_df.loc[ebola_df['Data'] == '2016-03-23']

# Removendo colunas desnecessárias para esta análise

ebola_mortes = ebola_mortes.drop('Data', inplace = False, axis = 1)

ebola_mortes = ebola_mortes.drop('Total de casos', inplace = False, axis = 1)

# Filtrando o DataFrame com a condição de haver mais de 1000 mortes

ebola_mortes = ebola_mortes.loc[ebola_mortes['Total de mortes'] >= 1000]

# Ordenando para que mostre de maneira crescente o total de mortes

ebola_mortes = ebola_mortes.sort_values(by = 'Total de mortes')



# Utilizamos .plot() para que plt encontre os valores e os coloque em gráfico

# kind = 'barh' nos traz um gráfico de barras horizontais

# Determinamos quais serão os eixos x e y

# legend = False remove a legenda

# title = <input> determina o título do gráfico

ebola_mortes.plot(kind = 'barh', x = 'País ou região', y = 'Total de mortes', legend = False, title = 'Países mais atingidos: Mortes por EBOLA')



# Determinando os nomes dos eixos

plt.xlabel('Mortes')

plt.ylabel('País')



# Execução do gráfico

plt.show()
# Tamanho do gráfico

plt.rcParams['figure.figsize'] = (11,8)



# Removendo colunas desnecessárias para esta análise

h1n1_mortes = h1n1_df.drop('Casos', inplace = False, axis = 1)

# Filtrando o DataFrame com a condição de haver mais de 200 mortes

h1n1_mortes = h1n1_mortes.loc[h1n1_mortes['Mortes'] >= 200]

# Ordenando para que mostre de maneira crescente o total de mortes

h1n1_mortes = h1n1_mortes.sort_values(by = 'Mortes')



# Utilizamos .plot() para que plt encontre os valores e os coloque em gráfico

# kind = 'barh' nos traz um gráfico de barras horizontais

# Determinamos quais serão os eixos x e y

# legend = False remove a legenda

# title = <input> determina o título do gráfico

h1n1_mortes.plot(kind = 'barh', legend = False, x = 'País ou região', y = 'Mortes', title = 'Países mais atingidos: Mortes por H1N1')



# Determinando os nomes dos eixos

plt.xlabel('Mortes')

plt.ylabel('País')



# Executando o gráfico

plt.show()
# Tamamho do gráfico

plt.rcParams['figure.figsize'] = (7,4)



# Removendo colunas desnecessárias para esta análise

sars_mortes = sars_df.drop('Casos', inplace = False, axis = 1)

# Filtrando o DataFrame com a condição de haver mais de 20 mortes

sars_mortes = sars_mortes.loc[sars_mortes['Mortes'] >= 20]

# Ordenando para que mostre de maneira crescente o total de mortes

sars_mortes = sars_mortes.sort_values(by = 'Mortes', ascending = True)



# Utilizamos .plot() para que plt encontre os valores e os coloque em gráfico

# kind = 'barh' nos traz um gráfico de barras horizontais

# Determinamos quais serão os eixos x e y

# legend = False remove a legenda

# title = <input> determina o título do gráfico

sars_mortes.plot(kind = 'barh', title = 'Países mais afetados: Mortes por SARS-CoV-1', legend = False, x = 'País', y = 'Mortes')



# Determinando os nomes dos eixos

plt.xlabel('Mortes')

plt.ylabel('País')



# Executando o gráfico

plt.show()
# Tamanho do gráfico

plt.rcParams['figure.figsize'] = (15,8)



# Removendo colunas desnecessárias para esta análise

covid_mortes = covid_df.drop('Confirmados', inplace = False, axis = 1)

covid_mortes = covid_df.drop('Recuperados', inplace = False, axis = 1)

# Filtrando o DataFrame com a condição de haver mais de 200 mortes

covid_mortes = covid_mortes.loc[covid_mortes['Data'] == '2020-06-08']

covid_mortes = covid_mortes.loc[covid_mortes['Mortos'] >= 5000]

# Ordenando para que mostre de maneira crescente o total de mortes

covid_mortes = covid_mortes.sort_values(by = 'Mortos')



# Utilizamos .plot() para que plt encontre os valores e os coloque em gráfico

# kind = 'barh' nos traz um gráfico de barras horizontais

# Determinamos quais serão os eixos x e y

# legend = False remove a legenda

# title = <input> determina o título do gráfico

covid_mortes.plot(kind = 'barh', legend = False, x = 'País', y = 'Mortos', title = 'Países mais atingidos: Mortes por COVID')



# Determinando os nomes dos eixos

plt.xlabel('Mortes')

plt.ylabel('País')



# Executando o gráfico

plt.show()
import folium



# Lendo o csv com as Latitudes de Longitudes dos países

csv_latlog_covid = pd.read_csv("../input/mapa-log-lat-covid/LatLog COVID-19.csv")

                           

# Criando o mapa para ser usado 

world_map_covid = folium.Map(location=[-16.1237611, -59.9219642], zoom_start=2, max_zoom = 10, min_zoom = 2)



# Criando repetição para incluir informações no mapa 

for i in range(0,len(csv_latlog_covid)):

    

    # Marcador do mapa

    folium.Marker(

        

        # Atribuindo as latitudes e longitudes de acordo com as colunas do csv

        location=[csv_latlog_covid.iloc[i]['Latitude'], csv_latlog_covid.iloc[i]['Longitude']],

        

        # Adicionando informação do pais ao marker 

        popup = csv_latlog_covid.iloc[i]['Pais'],

        

        #Atribuindo tìtulo ao marker

        title = 'País com caso de COVID-19',

        

         #Definido cor do marcador 

        icon = folium.Icon(color='red')

        

    # Tudo isso adicionado ao mapa

    ).add_to(world_map_covid)

    

world_map_covid
csv_latlog_ebola = pd.read_csv('../input/mapa-log-lat-ebola/LatLog Ebola.csv')





world_map_ebola = folium.Map(location=[14.497401, -14.452362], zoom_start=2, max_zoom = 10, min_zoom = 2)



for i in range(0,len(csv_latlog_ebola)):

    folium.Marker(

        location=[csv_latlog_ebola.iloc[i]['Latitude'], csv_latlog_ebola.iloc[i]['Longitude']],

        popup = csv_latlog_ebola.iloc[i]['Pais'],

        title = 'Pais com casos de Ebola',

        icon = folium.Icon(color='green')

        

    ).add_to(world_map_ebola)

    

world_map_ebola
csv_latlog_sars = pd.read_csv('../input/mapa-log-lat-sars/LatLog SARS.csv')
world_map_sars = folium.Map(location=[35.86166, 104.195397], zoom_start=2, max_zoom = 10, min_zoom = 2)



for i in range(0,len(csv_latlog_sars)):

    folium.Marker(

        location=[csv_latlog_sars.iloc[i]['Latitude'], csv_latlog_sars.iloc[i]['Longitude']],

        popup = csv_latlog_sars.iloc[i]['Pais'],

        title = 'Pais com casos de SARS',

        icon = folium.Icon(color='blue')

        

    ).add_to(world_map_sars)

    

world_map_sars
csv_latlog_h1n1 = pd.read_csv('../input/maploglath1n1/LatLog H1N1.csv')
world_map_h1n1 = folium.Map(location=[14.497401, -14.452362], zoom_start=2, max_zoom = 10, min_zoom = 2)



for i in range(0,len(csv_latlog_h1n1)):

    folium.Marker(

        location=[csv_latlog_h1n1.iloc[i]['Latitude'], csv_latlog_h1n1.iloc[i]['Longitude']],

        popup = csv_latlog_h1n1.iloc[i]['Pais'],

        title = 'Pais com casos de H1N1',

        icon = folium.Icon(color='purple')

        

    ).add_to(world_map_h1n1)

    

world_map_h1n1