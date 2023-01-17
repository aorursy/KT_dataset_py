# download
import requests

# parsing do HTML
from bs4 import BeautifulSoup

# expressões regulares
import re
# URL de análise
url_pagina = 'https://pt.wikipedia.org/wiki/Portal:COVID-19'
# Utiliza a biblioteca requests para fazer o download da pagina HTML
pagina_html = requests.get(url_pagina)

# O Código 200 indica que o download foi bem sucedido
pagina_html
# O Beautiful Soup 
soup = BeautifulSoup(pagina_html.content, 'html.parser')
#soup.prettify()
tbs_covid_pais = soup.find_all(class_='wikitable sortable mw-collapsible no-ref')

# Lembre-se que o método find_all retorna uma lista. No caso deste exemplo, 
# só existe um elemento na lista
tb_covid_pais = tbs_covid_pais[0]
tb_covid_pais
# Retorna uma lista com todas as linhas da tabela
linhas_tb = tb_covid_pais.find_all('tr')
lin = linhas_tb[2].get_text()
lin
lin.strip("\n").strip(" ")
lin
lin = lin.strip("\n").strip(" ")
lin
lin = lin.replace(u"\xa0", u"")
lin
lin = lin.replace("\n","\t")
lin
import re
lin = re.sub(r"\[.*]\t", "\t", lin)
lin
lin = re.sub(r"\[.*\]", "", lin)
lin
lin = lin.rstrip("\t")
lin
import re
def prep_linha(linha):
    # Remove os caracteres \n no começo e final da string
    # Em seguida, remove os espaços
    linha = linha.get_text().strip("\n").strip(" ")
    
    # Substitui os caracteres \n no meio da string por \t
    linha = linha.replace("\n","\t")
    
    # Substitui os caracteres \xa0 pela string vazia, nada
    # Note que existe um u antes da string. Isso indica que a string está em unicode
    linha = linha.replace(u"\xa0", u"")
    
    # Utiliza regex para encontrar as notas depois do nome do pais
    linha = re.sub(r"\[.*]\t", "\t", linha)
    
    # Utiliza regex para remover as notas ao final do arquivo
    linha = re.sub(r"\[.*\]", "", linha)
    
    # Remove o \t ao final da linha
    linha = linha.rstrip("\t")
    
    # Substitui o — na coluna de curados pela string vazia
    linha = linha.replace("—", "")
    
    return linha
ini_tb = 2
fim_tb = 225
# Esta primeira linha define o título da tabela
texto_final = "pais\tcasos\tmortes\tcurados\n"
for i in range(ini_tb, fim_tb+1):
    linha = prep_linha(linhas_tb[i])
    texto_final += linha
    texto_final += "\n"
print(texto_final)
arquivo = open("lista-paises.csv", "w")
arquivo.write(texto_final)
arquivo.close()
import pandas as pd
df = pd.read_csv("lista-paises.csv", sep='\t')
df.head()
df.casos.mean()
df.casos.sum()