import requests

from bs4 import BeautifulSoup

import pandas as pd
url_pagina_html = 'https://g1.globo.com/bemestar/coronavirus/noticia/2020/03/22/casos-de-coronavirus-no-brasil-em-22-de-marco.ghtml'
pagina_html = requests.get(url_pagina_html)

print(pagina_html)
print(pagina_html.content)
soup = BeautifulSoup(pagina_html.content, 'html.parser')

soup.prettify()
lista_tabelas_html = soup.find_all('table')

lista_tabelas_html
tb_corona_por_UF_tag = lista_tabelas_html[0]
print(type(tb_corona_por_UF_tag))
tb_corona_por_UF_tag.get_text()
tb_corona_por_UF = tb_corona_por_UF_tag.get_text()

tb_corona_por_UF
tb_corona_por_UF = tb_corona_por_UF[52:332]

tb_corona_por_UF
lst_casos_por_estado = tb_corona_por_UF.split()

lst_casos_por_estado
estados = []

secretarias = []

ministerio = []

for i in range(0, len(lst_casos_por_estado), 3):

 estados.append(lst_casos_por_estado[i])

 secretarias.append(int(lst_casos_por_estado[i+1]))

 ministerio.append(int(lst_casos_por_estado[i+2]))

print(estados)

print(secretarias)

print(ministerio)
df_casos_corona_por_UF = pd.DataFrame({ 

    "Estado": estados,

    "Secretarias" : secretarias,

    "Ministerio" : ministerio,

})

df_casos_corona_por_UF
df_casos_corona_por_UF["Secretarias"].sum()
df_casos_corona_por_UF["Ministerio"].sum()