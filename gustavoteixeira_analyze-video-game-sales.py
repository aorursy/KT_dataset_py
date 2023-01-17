%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')
#Leitura do arquivo

videogames = pd.read_csv("../input/videogamesales/vgsales.csv")
#Exibindo as 10 primeiras linhas do Dataframe

videogames.head(10)
#Resumo de informações em todas as colunas

videogames.describe()
#Quantidade de linhas e colunas no Dataframe

videogames.shape
#Renomeando colunas

videogames.columns = ['Ranking', 'Nome', 'Plataforma', 'Ano', 'Gênero',

                     'Editora', 'Vendas América do Norte', 'Vendas EUA',

                     'Vendas Japão', 'Outras vendas', 'Vendas Global']
#Exibindo novamente as 10 primeiras linhas do arquivo, agora com os novos nomes nas colunas

videogames.head(10)
#Verificando linhas onde não há ano de lançamento definido

videogames[videogames['Ano'].isnull()].head()
#Verificando linhas onde não há editora definida

videogames[videogames['Editora'].isnull()].head()
#Visualizando a quantidade de jogos que foram lançados para cada plataforma

videogames['Plataforma'].value_counts()
#Visualizando gráficamente os dados anteriores 

videogames['Plataforma'].value_counts().head(10).plot(kind='bar', figsize=(11,5), grid = False, rot=0, color='green')



#Enfeitando o gráfico.

plt.title('Os 10 videogames com mais títulos lançados')

plt.xlabel('Videogame') #Nomeando o eixo X, onde fica o nome dos videogames

plt.ylabel('Quantidade de jogos lançados') #Nomeando o eixo Y, onde fica a quantidade de jogos

plt.show() #Exibindo o gráfico
#Visualizando a quantidade de jogos que foram lançados para cada gênero

videogames['Gênero'].value_counts()
#Visualizando gráficamente os dados anteriores 

videogames['Gênero'].value_counts().head(10).plot(kind='bar', figsize=(11,5), grid = False, rot=0, color='green')



#Enfeitando o gráfico.

plt.title('Os 10 gêneros que mais possuem jogos lançados')

plt.xlabel('Gênero') 

plt.ylabel('Quantidade de jogos lançados')

plt.show()
#Visualizando as 10 editoras que mais lançaram jogos

videogames['Editora'].value_counts().head(10)
#Visualizando gráficamente os dados anteriores 

videogames['Editora'].value_counts().head(5).plot(kind='bar', figsize=(11,5), grid = False, rot=0, color='green')



#Enfeitando o gráfico.

plt.title('As 5 editoras que mais lançaram jogos')

plt.xlabel('Editora') 

plt.ylabel('Quantidade de jogos lançados')

plt.show()
#Os 10 jogos mais vendidos da história

top_10_vendidos = videogames[['Nome', 'Vendas Global']].head(10).set_index('Nome').sort_values('Vendas Global', ascending=True)

top_10_vendidos.plot(kind='barh', figsize=(11,7), grid=False, color='darkred', legend=False)



plt.title('Top 10 jogos mais vendidos no mundo')

plt.xlabel('Total de vendas (em milhões de doláres)')

plt.ylabel('Jogo')

plt.show()
crosstab_vg = pd.crosstab(videogames['Plataforma'], videogames['Gênero'])

crosstab_vg.head()
crosstab_vg['Total'] = crosstab_vg.sum(axis=1)

crosstab_vg.head()
top10_platforms = crosstab_vg[crosstab_vg['Total']>1000].sort_values('Total', ascending = False)

top10_final = top10_platforms.append(pd.DataFrame(top10_platforms.sum(), columns=['total']).T, ignore_index=False)



sns.set(font_scale=1)

plt.figure(figsize=(18,9))

sns.heatmap(top10_final, annot=True, vmax=top10_final.loc[:'PS',:'Strategy'].values.max(), vmin=top10_final.loc[:,:'Strategy'].values.min(), fmt='d')

plt.xlabel('GÊNERO')

plt.ylabel('CONSOLE')

plt.title('QUANTIDADE DE TÍTULOS POR GÊNERO E CONSOLE')













plt.show()