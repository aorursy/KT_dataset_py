import requests
url = 'https://pt.wikipedia.org/wiki/Campeonato_Brasileiro_de_Futebol_de_2004'
page = requests.get(url)
from bs4 import BeautifulSoup
soup = BeautifulSoup(page.content, 'html.parser')
tabela = soup.find_all('table')
len(tabela)
tabela = tabela[2]
linhas = tabela.find_all('tr')
len(linhas)
linha = linhas[2]
linha = linha.get_text()
linha = linha.strip('\n')
linha = linha.replace('\n\n', ',')
linha
def prep_linha(linha):
    linha = linha.get_text()
    linha = linha.strip('\n')
    linha = linha.replace('\n\n', ',')
    return linha
tabela_final = 'Equipe,Pontos,Jogos,Vitórias,Empates,Derrotas,Gols marcados,Gols sofridos,Saldo de Gols,Aproveitamento\n'
for i in range(2,len(linhas)-1):
    linha = prep_linha(linhas[i])
    tabela_final += linha
    tabela_final += '\n'
print(tabela_final)
arquivo = open('brasileirao2004.csv', 'w')
arquivo.write(tabela_final)
arquivo.close()
import pandas as pd
df = pd.read_csv('brasileirao2004.csv', sep = ',')
df
df.head(4)
df.tail(4)
df.sort_values('Gols sofridos')