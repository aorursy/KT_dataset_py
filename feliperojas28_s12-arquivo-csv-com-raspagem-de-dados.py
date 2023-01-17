import requests
url = 'https://pt.wikipedia.org/wiki/Pandemia_de_COVID-19_no_Brasil'
pagina_html = requests.get(url)
from bs4 import BeautifulSoup
soup = BeautifulSoup(pagina_html.content, 'html.parser')
tabela = soup.find_all(class_='wikitable sortable')
len(tabela)
tabela = tabela[0]
linhas = tabela.find_all('tr')
len(linhas)
linha = linhas[1].text
linha
linha = linhas[1].text
linha = linha.strip('\n')
linha = linha.replace(u'\xa0', '')
linha = linha.replace('Sudeste\n\n60692\n\n', '')
linha = linha.replace('\n\n', '\t')
linha
def prep_linha(linha):
    linha = linha.get_text().strip('\n')
    linha = linha.replace(u'\xa0', '')
    linha = linha.replace('Sudeste\n\n60692\n\n', '')
    linha = linha.replace('Sul\n\n6867\n\n', '')
    linha = linha.replace('Nordeste\n\n42157\n\n', '')
    linha = linha.replace('Centro-Oeste\n\n4013\n\n', '')
    linha = linha.replace('Norte\n\n21377\n\n', '')
    linha = linha.replace('\n\n', '\t')
    return linha
texto_final = 'Estado\tCasos confirmados Min.Saúde\tCasos confirmados Sec.Saúde\tÓbitos Min.Saúde\tÓbitos Sec.Saúde\n'
for i in range(1,len(linhas)-1):
    linha = prep_linha(linhas[i])
    texto_final += linha
    texto_final += '\n'
print(texto_final)
arquivo = open('lista_18_estados.csv', 'w')
arquivo.write(texto_final)
arquivo.close()
import pandas as pd
df = pd.read_csv('lista_18_estados.csv', sep = '\t')
df