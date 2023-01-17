html = """
<html>
    <head>
        <title>Indicados e vencedores do Oscar em 2020</title>
    </head>
    <body>
        <h1>Oscar 2020</h1>
        <p>A maior premiação do cinema aconteceu em Fevereiro. Os destaques deste ano foram:</p>
        <ul class="destaques">
            <li>Coringa</li>
            <li>1917</li>
            <li>Era Uma Vez em Hollywood</li>
            <li>O Irlandês</li>
        </ul>
        <p>Veja abaixo os vencedores de 3 categorias.</p>
        <div>
            <div class="categoria-melhor-filme">
                <h2>Categoria: melhor filme</h2>
                <ul>
                    <li>Ford vs Ferrari</li>
                    <li>O Irlandês</li>
                    <li>JoJo Rabbit</li>
                    <li>Coringa</li>
                    <li>Adoráveis Mulheres</li>
                    <li>História de um Casamento</li>
                    <li>1917</li>
                    <li>Era Uma Vez Em Hollywood</li>
                    <li class="vencedor">Parasita <strong>[VENCEDOR]</strong></li>
                </ul>
            </div>
            <br>
            <div class="categoria-melhor-ator">
                <h2>Categoria: melhor ator</h2>
                <ul>
                    <li>Antonio Banderas - Dor e Glória</li>
                    <li>Leonardo DiCaprio - Era Uma Vez Em... Hollywood</li>
                    <li>Adam Driver - História de um Casamento</li>
                    <li class="vencedor">Joaquin Phoenix - Coringa <strong>[VENCEDOR]</strong></li>
                    <li>Jonathan Price - Dois Papas</li>
                </ul>
            </div>
            <br>
            <div class="categoria-melhor-atriz">
                <h2>Categoria: melhor atriz</h2>
                <ul>
                    <li>Cythia Erivo - Harriet</li>
                    <li>Scarlett Johansson - História de um Casamento</li>
                    <li>Saoirse Ronan - Adoráveis Mulheres</li>
                    <li>Charlize Theron - O Escândalo</li>
                    <li class="vencedor">Renée Zellweger - Judy: Muito Além do Arco-Íris <strong>[VENCEDOR]</strong></li>
                </ul>
            </div>
        </div>
    </body>
</html>
"""
from bs4 import BeautifulSoup
soup = BeautifulSoup(html, 'html.parser')
type(soup)
soup
print(soup.prettify())
soup.html.head.title
soup.title
soup.title.get_text()
soup.html.body.h1
soup.h1
soup.h1.get_text()
soup.p.get_text()
soup.find("p")
soup.find("p").get_text()
soup.find_all("p")
soup.find_all("p")[1]
soup.find_all("p")[1].get_text()
soup.find_all("h2")
soup.ul.find_all("li")
print("Lista de destaques:")

for filme in soup.ul.find_all("li"):
    print(filme.get_text())
# Insira a sua resposta aqui
soup.find_all('p', limit = 1)
soup.find_all('p')[0]
soup.find_all(["h2", "li"])
soup.find_all("li", text="Coringa")
soup.find_all("li", text="1917")
soup.find_all("li", text="O Irlandês")
import re
soup.find_all("h2", text=re.compile("^Categoria"))
soup.find_all("ul", {"class": "destaques"})
for item in soup.find_all("ul", {"class": "destaques"}):
    print(item.get_text())
soup.find_all("li", {"class": "vencedor"})
for item in soup.find_all("li", {"class": "vencedor"}):
    print(item.get_text())
for item in soup.find_all("li", {"class": "vencedor"}):
    print(item.get_text().replace(" [VENCEDOR]", ""))
soup.find("li", {"class": "vencedor"}).get_text()
soup.find("li", {"class": "vencedor"}).find_parent()
soup.find("li", {"class": "vencedor"}).find_parent('div')
soup.find("li", {"class": "vencedor"}).find_parent('div').find('h2').get_text()
soup.find('h2').get_text()
soup.find('h2').find_next_sibling()
soup.find('div', {'class': 'categoria-melhor-filme'}).find_next_siblings()
soup.find('li', {'class': 'vencedor'}).find_previous_sibling().get_text()
soup.find('div', {'class': 'categoria-melhor-atriz'}).find_previous_siblings()
soup.find('div', {'class': 'categoria-melhor-ator'}).find('li', {'class': 'vencedor'}).get_text()
soup.find('div', {'class': 'categoria-melhor-ator'}).find('li', {'class': 'vencedor'}).find_next('li').get_text()
soup.find('div', {'class': 'categoria-melhor-ator'}).find('li', {'class': 'vencedor'}).find_previous('li').get_text()
soup.find('div', {'class': 'categoria-melhor-ator'}).find('li', {'class': 'vencedor'}).find_all_next('li')
soup.find('div', {'class': 'categoria-melhor-ator'}).find('li', {'class': 'vencedor'}).find_all_previous('li')
# Insira a sua resposta aqui
import pandas as pd
soup.find_all("h2", text=re.compile("^Categoria"))
categorias = soup.find_all("h2", text=re.compile("^Categoria"))
soup.find_all("li", {"class": "vencedor"})
vencedores = soup.find_all("li", {"class": "vencedor"})
vencedores_categorias = {}
for cat, ven in zip(categorias, vencedores):
    categoria = cat.get_text().replace("Categoria: ", "")
    vencedor = ven.get_text().replace(" [VENCEDOR]", "")
    
    vencedores_categorias.update({categoria: [vencedor]})
vencedores_categorias
vencedores_categorias_df = pd.DataFrame.from_dict(vencedores_categorias, orient="index", columns=["Vencedores"])
vencedores_categorias_df
from urllib.request import urlopen
url = 'https://ocean-web-scraping.herokuapp.com/'
response = urlopen(url)
html_ocean_ws = response.read()
html_ocean_ws
soup_ocean_ws = BeautifulSoup(html_ocean_ws, 'html.parser')
soup_ocean_ws
print(soup_ocean_ws.prettify())
# Insira a sua resposta aqui