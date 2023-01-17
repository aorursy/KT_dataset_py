# Importando bibliotecas

import requests # Para fazer requisições
from bs4 import BeautifulSoup # Para trabalhar com dados vindos de paginas web
url = "http://books.toscrape.com/index.html"
response = requests.get(url)
html = response.content
scraped = BeautifulSoup(html, 'html.parser')
# Cada livro está num article com class product_pod
# assim selecionamos todos os articles para iterar por essa lista
books = scraped.find_all(class_='product_pod')

titles = [] # Lista que receberá os títulos de cada livro
prices = [] # Lista que recebera os preços de cada livro

list = {'Books': titles, 'Prices (£)': prices} # Lista que será usada para criar o dataframe do pandas

for item in books:
    title = item.h3.a['title'] # O Título está no atributo Title do <a> dentro do <h3> que está dentro do <article>
    
    # O preço está numa classe 'price_color' dentro de outra classe 'product_price'
    # O lstrip retira da esquerda o simbolo de Libra '£' e o float salva nosso dado como float
    price = float(item.find(class_='product_price').find(class_='price_color').text.lstrip('£'))
    
    titles.append(title) # O título é adicionado à lista de Títulos
    prices.append(price) # O Preço, já como float, é adicionado à lista de Preços
    #print(title)

print(list)
# Importa o pandas como pd
import pandas as pd
df = pd.DataFrame(list) # Cria um dataframe com nossa listade livros e preços
df
df.describe() # Descreve as colunas que contem numeros, nesse caso, apenas a coluna Prices contem numeros
df.info() # Mostra informações do dataframe e os tipos dos dados de cada coluna e quantas entradas que não são null
df.shape # Mostra quantas linhas e colunas tem o dataframe
df.Books # Mostra a coluna Books do dataframe
df['Prices (£)'] # Outra forma de chamar uma coluna é entre colchetes e aspas: ['Coluna']
df.iloc[:,1] 
# Localiza no dataframe pelo indice, sendo o primeiro valor antes da virgula para as linhas
# O valor apos a virgula é para o indice da coluna
# o : seleciona todas as linhas
# o 1 seleciona a segunda coluna (Lembre-se que o indice começa por 0)
df.iloc[:, 0]
# Aqui seleciona todas as linhas, mas a primeira coluna
df.loc[:, 'Books']
# Outra forma de selecionar todas as linhas da coluna de nome 'Books'
# o .loc seleciona pelo indice da linha, mas pela label = nome da coluna
df.loc[:, 'Prices (£)']
# aqui seleciona todas as linhas da coluna Prices
df.nlargest(5, 'Prices (£)')
# Comando muito utilizado para retornar os maiores valores, 
# nesse caso, selecionei os 5 registros com maiores preços, note que em ordem do maior para o menor
df.nsmallest(5, 'Prices (£)')
# Esse comando retorna os 5 menores preços, note que em ordem do menor para o maior
df.Books.value_counts()
# Comando para contagem de valores únicos, como não temos Livros duplicados, todos têm apenas 1 registro
df.tail()
df.sort_values('Prices (£)', ascending=False)