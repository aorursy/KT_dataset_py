import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import csv
url_pagina = 'https://pt.wikipedia.org/wiki/Portal:COVID-19'
pagina_html = requests.get(url_pagina)
soup = BeautifulSoup(pagina_html.content, 'html.parser')
soup.prettify()
url_pagina2 = "https://www.br.undp.org/content/brazil/pt/home/idh0/rankings/idh-global.html"
pagina_html2 = requests.get(url_pagina2)
soup2 = BeautifulSoup(pagina_html2.content , "html.parser" )
soup2.prettify()
tbs_covid_pais = soup.find_all(class_='wikitable sortable mw-collapsible no-ref')
tb_covid_pais = tbs_covid_pais[0]
linhas_tb = tb_covid_pais.find_all('tr')
tbs_idh_pais = soup2.find_all(class_="tableizer-table")
tb_idh_pais = tbs_idh_pais[0]
linhas_idh = tb_idh_pais.find_all("tr")
def prep_linha(linha):
    #Utilizamos o strip para remover caracteres
    #Utilizamos o replace para substituir um caractere por outro
    #Utilizamos o regex para encontrar e/ou remover as notas
    linha = linha.get_text().strip("\n").strip(" ")
    linha = linha.replace("\n","\t")
    linha = linha.replace(u"\xa0", u"")
    linha = re.sub(r"\[.*]\t", "\t", linha)
    linha = re.sub(r"\[.*\]", "", linha)
    linha = linha.rstrip("\t")
    linha = linha.replace("–", "0")
    return linha
def prep_linha2(linha2):
    linha2 = linha2.get_text().strip("\n").strip(" ")
    linha2 = linha2.replace("\n","\t")
    linha2 = linha2.rstrip("\t")
    return linha2
ini_tb = 2
fim_tb = 222
ini_tb2 = 1 
fim_tb2 = 192
texto_final = "Países\tCasos\tMortes\tCurados\n"
for i in range(ini_tb, fim_tb+1):
    linha = prep_linha(linhas_tb[i])
    texto_final += linha
    texto_final += "\n"
print(texto_final)
texto_final2 = "Ranking IDH Global\tPaís\tIDH\n"
for i in range(ini_tb2, fim_tb2+1):
    linha2 = prep_linha2(linhas_idh[i])
    texto_final2 += linha2
    texto_final2 += "\n"
    texto_final2 = texto_final2.replace(',','.')
print(texto_final2)
arquivo = open("lista-paises.csv", "w")
arquivo.write(texto_final)
arquivo.close()
arquivo = open("lista-países_idh.csv","w")
arquivo.write(texto_final2)
arquivo.close()
df = pd.read_csv("lista-paises.csv", sep='\t')
df = df.iloc[::-1].reset_index(drop=True)
df = df.iloc[::-1].reset_index(drop=False)
df2 = pd.read_csv("lista-países_idh.csv",sep="\t")
df2.dropna(axis=0, how='any')
df2 = df2.loc[[1, 51, 108, 148]]
df2.plot(kind='bar', x='País', y='IDH');
print('A media de casos por país é de',round(df.Casos.mean()))
print('O numero total de casos confirmardos pelo mundo é de',df.Casos.sum())
print('A média de mortes por País é de',round(df.Mortes.mean()))
print('O numero total de mortes pelo mundo é de',df.Mortes.sum())
print('A média de casos recuperados por País é de',round(df.Curados.mean()))
print('O numero total de casos recuperados pelo mundo é de',df.Curados.sum())
df = pd.read_csv("lista-paises.csv", sep='\t')
df = df.head(10)
df.plot.bar(x='Países');
df3 = pd.read_csv("../input/listapaises/lista-paises_dados2.csv", sep=';', encoding='latin1')
df3 = df3.loc[[55, 22, 188, 93]]
df3 = df3.stack().str.replace(',','.').unstack()
df3['casos/milhao'] = df3['casos/milhao'].apply(float)
df3['mortes/milhao'] = df3['mortes/milhao'].apply(float)
df3.plot.bar(x='pais',y='casos/milhao');
df3.plot.bar(x='pais',y='mortes/milhao');
df4 = pd.read_csv('../input/csv-unif/CSV unificado.csv', sep=';', encoding='latin1')
df4 = df4.rename(columns={'C menos D':'IDH x Ranking Pandemia - casos'})
df4 = df4.rename(columns={'C menos E':'IDH x Ranking Pandemia - mortes'})
df4.plot.scatter(x='IDH',y='IDH x Ranking Pandemia - casos').invert_xaxis()
df4.plot.scatter(x='IDH',y='IDH x Ranking Pandemia - mortes').invert_xaxis()
