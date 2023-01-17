import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) 
import csv
import networkx as nx
import matplotlib.pyplot as plt

#1 - Seguindo a estrutura de construção de redes semânticas tratadas na disciplina, utilizando a linguagem python suportada por um kaggle notebook: 

#a - Busque por uma base de dados.
#b - Faça sua importação no kaggle.
 
df = pd.read_csv(r'../input/herois/rede-de-herois.csv')#faz leitura do arquivo hero-network
print(df.head())

with open('../input/herois/rede-de-herois.csv', 'r') as heroisEm:
    heroisEm = csv.reader(heroisEm)
    headers = next(heroisEm)
    herois = [row for row in heroisEm]


#c - Crie e explique um procedimento para filtrar dados dessa base e prepará-los para criação de uma rede semântica.

unicosHerois = list(set([row[0] for row in herois])) #Pega as redundâncias do arquivo de super-heróis
 
id=list(enumerate(unicosHerois))#Cria uma lista de tuplas com ids únicos e seus nomes para cada super-herói na rede


 
keys = {nome: i for i, nome in enumerate(unicosHerois)} #cria um dicionário (mapa hash) que mapeia cada id para os nomes dos super-heróis
 
 
links = [] #Cria uma lista vazia
 
 
for row in herois: #Mapeia todos os nomes no arquivo csv para seus números de id
    try:
        links.append({keys[row[0]]: keys[row[1]]})
    except:
        links.append({row[0]: row[1]})
 
cluster = nx.Graph() #Cria um grafo

counter = 0

heroiNoId=[] #Pega a fonte e alvos
for row in id:
 heroiNoId.append(row[0])
        
 
cluster.add_nodes_from(heroiNoId)#Cria nós para o gráfico


 

#d - Construa e gere uma imagem de uma rede utilizando a biblioteca networkx.
nx.draw(cluster)

plt.axis('off')
plt.show(cluster)






