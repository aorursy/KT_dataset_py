import requests 

from bs4 import BeautifulSoup

import re 
url_pagina = "https://www.br.undp.org/content/brazil/pt/home/idh0/rankings/idh-global.html"
pagina_html = requests.get(url_pagina)

pagina_html
soup = BeautifulSoup(pagina_html.content , "html.parser" )

soup.prettify()
tbs_idh_pais = soup.find_all(class_="tableizer-table")



tb_idh_pais = tbs_idh_pais[0]

tbs_idh_pais
linhas_tb = tb_idh_pais.find_all("tr")

linhas_tb
lin = linhas_tb[2].get_text()

lin
print(lin)
lin.strip("\n")
lin = lin.replace("\n","\t")

lin
lin = lin.rstrip("\t")

lin
def prep_linha(linha):

    linha = linha.get_text().strip("\n").strip(" ")

    linha = linha.replace("\n","\t")

    linha = linha.rstrip("\t")

    return linha
ini_tb = 1 

fim_tb = 192
texto_final = "Ranking IDH Global\tPaís\tIDH 2014\n"

for i in range(ini_tb , fim_tb+1):

    linha = prep_linha(linhas_tb[i])

    texto_final += linha

    texto_final += "\n"

print(texto_final)
arquivo = open("lista-países.csv","w")

arquivo.write(texto_final)

arquivo.close()
import pandas as pd 

df = pd.read_csv("lista-países.csv",sep="\t")

df.head()