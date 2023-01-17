#Importando as bibliotecas necessárias para a análise

%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')
#Importando o dataset

dataset = pd.read_csv("../input/vgsales/vgsales.csv")
#Renomeando as Colunas

dataset.columns = ['Ranking', 'Nome', 'Plataforma', 'Ano', 'Gênero', 'Editora', 'Vendas América do Norte', 'Vendas EUA', 'Vendas Japão', 'Outras Vendas', 'Vendas Globais']
#Carregando as 10 primeiras linhas do Dataframe

dataset.head(10)
#Procurando dados nulos na coluna 'Ano'

dataset[dataset['Ano'].isnull()].head()
#Quantificando os jogos por gênero

dataset['Gênero'].value_counts()
#Cruzando os dados de Plataforma e Gênero em uma nova tabela

tabelacruzada = pd.crosstab(dataset['Plataforma'],dataset['Gênero'])
#Conferindo a nova tabela

tabelacruzada.head()
#Acrecentando a Coluna total no final da nova tabela e somando os valores em linha

tabelacruzada['Total'] = tabelacruzada.sum(axis=1)
#Organizando os dados da última coluna de forma decrescente

tabelacruzada = tabelacruzada.sort_values('Total', ascending=False)
#Criando uma nova tabela com valores da coluna 'Total' superiores a 1000

top10=tabelacruzada[tabelacruzada['Total']>1000]
top10.head()
#Gerando um gráfico 'heatmap' dos dados da tabela 'Top10'

sns.set(font_scale=1)

plt.figure(figsize=(18,9))

sns.heatmap(top10,annot=True, vmax=top10.loc[:'PS', :'Strategy'].values.max(), vmin=top10.loc[:, :'Strategy'].values.min(), fmt='d')

plt.xlabel('Gênero')

plt.ylabel('Plataforma')

plt.show()
#definindo o objeto de estudo PS2

PS2 = dataset[dataset[u'Plataforma']=='PS2']
#Gerando um gráfico com a quantidade de jogos de cada gênero feito para o PS2

PS2[u'Gênero'].value_counts().plot.bar(color='green')
#Top5 dos jogos de esportes vendidos nos EUA (PS2)

PS2.loc[(PS2['Gênero']=='Sports') & (PS2['Vendas EUA']>2)].head(5)