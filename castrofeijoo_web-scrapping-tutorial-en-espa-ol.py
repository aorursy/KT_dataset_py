from requests import get

from bs4 import BeautifulSoup
url_carteras = 'https://www.voguebags.cn/handbags-c-1/'
url_leido = get(url_carteras)
url_leido
# Veamos los primeros 500 caracteres que contiene la respuesta

url_leido.content[:1000]
# Que tipo de datos son?

type(url_leido.content)
html = BeautifulSoup(url_leido.content, 'html.parser')
# Que tipo de objeto es?

type(html)
# Que contiene?

html
logo = html.find(class_='logo')

logo
logo.find('a')
link_a_pagina_principal = logo.find('a')['href']

link_a_pagina_principal
logo.find('img')
logo.find('img')['src']
logo.find('img')['alt']
logo.find('img')['title']
logo.find('a').find('img')['src']
logo.find('img')['src']
categorias = html.find_all(class_= 'mu_category_name')
categorias
categorias[1].getText()
diseñadores = []

for cat in categorias:

    diseñadores.append(cat.getText())

diseñadores
guardar = [diseñador.getText() for diseñador in categorias]
guardar
html.find_all('a')
todos_links = html.find_all('a')
todos_links_texto = [ lnk['href']  for lnk in todos_links]
todos_links_texto
items_menu = html.find_all(class_='categoryListBoxContents')
items_menu[0].find('a')
items_menu[0].find('a')['href']
items_menu
items_menu[2].find('a')
items_menu[2].find('a')['href']
links = [item.find('a')['href'] for item in items_menu]

links
links_diseñadores = html.find_all(class_='category-products')
links_todos = [link.find('a')['href'] for link in links_diseñadores]
links_todos
arte = html.find_all(class_ = 'mu_nav_ico')
arte[2].find_all('a')
links_todos[3]
bottega_pagina = get(links_todos[3])

bottega = BeautifulSoup(bottega_pagina.content)
len(bottega.find_all(class_='centerBoxContentsProducts'))
bottega.find(id='productsListingTopNumber')
bottega.find(id='productsListingTopNumber').find_all('strong')
total = bottega.find(id='productsListingTopNumber').find_all('strong')[2].getText()

total
int(total)
int(total)/30
import math

math.ceil(int(total)/30)
links
links[3] + '?page=2&sort=20a'
links[3] + '?page=3&sort=20a'
pag = get(links[3]+'?page=3&sort=20a')
pag = BeautifulSoup(pag.content)
precios = pag.find_all(class_='musheji_price')

precios
precios = [p.getText() for p in precios]

precios
carteras = pag.find_all(class_='musheji_name')

carteras
carteras_texto = [c.getText() for c in carteras]

carteras_texto
precios
import pandas as pd

carteras_scrap = pd.DataFrame({'Carteras':carteras_texto, 'Precio':precios})
carteras_scrap
link_B = links[3]

link_B
pagina_0 = get(link_B)
pagina_0 = BeautifulSoup(pagina_0.content)
carteras_totales = pagina_0.find(id='productsListingTopNumber').find_all('strong')[2].getText()

carteras_totales
paginas_totales = math.ceil(int(carteras_totales)/30)

paginas_totales
for p in range(paginas_totales):

    print(links[3] + '?page='+ str(p+1) +'&sort=20a')
precios_totales = []

carteras_totales = []



for p in range(paginas_totales):

    pagina_individual = links[3] + '?page='+ str(p+1) +'&sort=20a'

    pagina_0 = get(pagina_individual)

    pagina_0 = BeautifulSoup(pagina_0.content)

    

    #Precios

    precios = pagina_0.find_all(class_='musheji_price')

    precios = [p.getText() for p in precios]

    precios_totales = precios_totales + precios



    #Nombres

    carteras = pagina_0.find_all(class_='musheji_name')

    carteras = [c.getText() for c in carteras]

    carteras_totales = carteras_totales + carteras
carteras_totales
len(carteras_totales)
len(precios_totales)
Bottega = pd.DataFrame({'Cartera': carteras_totales, 'Precio':precios_totales})
Bottega
Bottega.info()
Bottega['Precio'] = Bottega['Precio'].str.replace('$','')
Bottega['Precio'] = pd.to_numeric(Bottega.Precio)
Bottega.info()
Bottega.Precio.mean()
Bottega.Precio.max()
Bottega.Precio.min()
Bottega.describe()
for url in links:

    print(url)
url = links[1]

pagina_0 = get(url)

pagina_0 = BeautifulSoup(pagina_0.content)

carteras_totales = pagina_0.find(id='productsListingTopNumber').find_all('strong')[2].getText()

paginas_totales = math.ceil(int(carteras_totales)/30)



precios_totales = []

carteras_totales = []



for p in range(paginas_totales):

    pagina_individual = url + '?page='+ str(p+1) +'&sort=20a'

    pagina_0 = get(pagina_individual)

    pagina_0 = BeautifulSoup(pagina_0.content)

    

    #Precios

    precios = pagina_0.find_all(class_='musheji_price')

    precios = [p.getText() for p in precios]

    precios_totales = precios_totales + precios



    #Nombres

    carteras = pagina_0.find_all(class_='musheji_name')

    carteras = [c.getText() for c in carteras]

    carteras_totales = carteras_totales + carteras



Nuevo_scrap = pd.DataFrame({'Cartera': carteras_totales, 'Precio':precios_totales})
Nuevo_scrap['Precio'] = Nuevo_scrap['Precio'].str.replace('$','')

Nuevo_scrap['Precio'] = pd.to_numeric(Nuevo_scrap['Precio'])
Nuevo_scrap
link = 'http://www.infobae.com'

pag = get(link)

pag = BeautifulSoup(pag.content)



titulos = pag.find_all(class_='headline')
[t.getText() for t in titulos]