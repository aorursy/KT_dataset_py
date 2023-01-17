# reponsável por se comunicar com o servidor e fazer o download da página
import requests

# responsável por analisar os dados da página e trabalhar com dados tabulares
import pandas as pd

# responsável por fazer a raspagem dos dados
from bs4 import BeautifulSoup
covid = "https://www.bbc.com/portuguese/brasil-51713943/"
page = requests.get(covid)
soup = BeautifulSoup(page.content, 'html.parser')
# 'page.content' é o conteúdo da página e 'html.parser' é o que vai interpretar esse conteúdo
print(soup.find_all('p'))
sudeste = soup.find_all('p')[15]
print(sudeste)
sudeste = sudeste.get_text()
print(sudeste)
casos_por_UF = list(sudeste.split())
casos_por_UF.pop(0)
print(casos_por_UF)
estados = []
casos = []

for i in range(0, len(casos_por_UF), 2):
    estados.append(casos_por_UF[i])
    casos.append(casos_por_UF[i+1])
    
print(estados)
print(casos)
tabela_sudeste = pd.DataFrame({
    "Estados": estados,
    "Casos": casos
})

# primeiro escrevemos as chaves como nome das colunas da tabela, e seus valores 
# como as listas que representarão seu conteúdo

print(tabela_sudeste)

casos_num = tabela_sudeste["Casos"].str.extract("(?P<casos_num>\d+.\d+)", expand=False)
print(casos_num)
tabela_sudeste = pd.DataFrame({
    "Estados": estados,
    "Casos": casos_num
})

print(tabela_sudeste)