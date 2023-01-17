url_pagina_html='https://www.bbc.com/portuguese/internacional-51718755'
!pip install requests

!pip install pandas 

!pip install bs4
# Responsavel por se comunicar com o servidor e fazer o download da pagina

import requests

# Responsavel por extrair (raspagem) as informações da página que se fez o download

from bs4 import BeautifulSoup



# Responsavel por analisar as informações da página

import pandas as pd



import re



import seaborn as sns



import numpy as np



from matplotlib import pyplot as plt

pagina_html = requests.get(url_pagina_html)

pagina_html
soup = BeautifulSoup(pagina_html.content, 'html.parser')

lista_de_tabelas_html = soup.find_all(class_="core-table gel-brevier")

#buscando por todas as tags <table> retirando todas as tags como> <td> e <table>

tb_corona_por_país_tag = lista_de_tabelas_html[0]

type(tb_corona_por_país_tag)
linhas_tb = tb_corona_por_país_tag.find_all('tr')
lin = linhas_tb[0].get_text()

lin
lin = lin.strip("\n").strip(" ")

lin
lin=lin.replace(" ","")

lin
lin=lin.replace(".","")

lin
lin=lin.replace("\n","\t")

lin
lin = lin.replace(",",".")

lin
lin = lin.rstrip("\t")

lin
import re

def prep_linha(linha):

    # Remove os caracteres \n no começo e final da string

    # Em seguida, remove os espaços

    linha = linha.get_text().strip("\n").strip(" ")

    

    linha=linha.replace(" ","")



    

    # Substitui os caracteres \n no meio da string por \t

    linha = linha.replace("\n","\t")

    

    linha = linha.replace(".","")

    

    linha = linha.replace(",",".")



    

    # Remove o \t ao final da linha

    linha = linha.rstrip("\t")

    



    

    return linha
ini_tb = 0

fim_tb = 209
# Esta primeira linha define o título da tabela

texto_final = "País\t\t\tMortes\t\t\tMortalidade\t\t\tCasos\n"

for i in range(ini_tb, fim_tb+1):

    linha = prep_linha(linhas_tb[i])

    texto_final += linha

    texto_final += "\n"

print(texto_final)
arquivo = open("lista-paises.csv", "w")

arquivo.write(texto_final)

arquivo.close()

df = pd.read_csv("lista-paises.csv", sep='\t\t\t')

df.head()
df.info()
df.describe()
df.shape
df.sum()["Mortes"]
df.sum()["Casos"]