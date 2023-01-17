# Vamos importar os pacotes necessários para esse tutorial:

# Link com referências de markdown para destaques no notebook do kaggle -> https://markdown-it.github.io/



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import requests



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Criando uma série com 4 números aleatórios, estamos chamando ela de "ações"



s = pd.Series(np.random.randn(4), name="ações")

s
# Os operadores podem ser aplicados a 'serie' de maneira similar aos arrays do NumPy



s * 100
# Podemos pegar o módulo dos números



np.abs(s)
# Existem diversos métodos que podemos aplicar (ler documentação oficial)

# .describe() nos entrega dados estatísticos referentes ao nosso dataset, como contagem, valor mínimo e máximo, etc



s.describe()
# Podemos rotular nossos dados

s.index = ['AMAZON', 'APPLE', 'MICROSOFT','GOOGLE']

s
# A sintaxe utilizada nos dicionários do `Python` pode ser utilizada nas `Series` do `Pandas`



s['AMAZON']
# Mais um exemplo



s['AMAZON'] = 0

s
# Mais um exemplo



'APPLE' in s
# Abrindo um .CSV dentro do meu Google Drive 



url = 'https://drive.google.com/file/d/1SOwu4GWBw_HljYFDD5K9sByp-smxFZ3P/view?usp=sharing'

path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]

df= pd.read_csv(path)

df
# Podemos utilizar cortes para ver apenas o conteúdo das linhas 2 a 5



df[2:5]
# Podemos pedir apenas colunas específicas



df[['country', 'tcgdp']]
# O atributo `iloc` é usado para selecionar linhas e colunas através de `integers` (números inteiros), ele segue o formato .iloc[rows, columns]

df.iloc[2:5, 0:4]
# O atributo `loc` é usado para selecionar linhas e colunas através de `integers` (núemros inteiros) e `labels` (rótulos)

df.loc[df.index[2:5], ['country', 'tcgdp']]
# O nosso DataFrame, por padrão segue o índice de acordo com o número da linha



df = df[['country', 'POP', 'tcgdp']]

df
# Para trocar o ´index´ para os países, ao invés do número da linha, podemos fazer o seguinte:



df = df.set_index('country')

df
# Podemos alterar o nome das colunas:



df.columns = 'population', 'total GDP'

df
# Podemos multiplicar todos os dados de população por 1000



df['population'] = df['population'] * 1e3

df
# Com uma linha de código é possível criar uma nova coluna que calcula o PIB per capita



df['GDP percap'] = df['total GDP'] * 1e6 / df['population']

df
# Também é possível plotar gráficos para análises exploratória dos dados



ax = df['GDP percap'].plot(kind='bar')

ax.set_xlabel('country', fontsize=12)

ax.set_ylabel('GDP per capita', fontsize=12)

plt.show()
# Podemos ordenar os dados pelo PIB per capita para melhorar a visualização do gráfico

df = df.sort_values(by='GDP percap', ascending=False)

df
ax = df['GDP percap'].plot(kind='bar')

ax.set_xlabel('country', fontsize=12)

ax.set_ylabel('GDP per capita', fontsize=12)

plt.show()
# Abrindo um CSV no meu Google Drive com filmes categorizados



movie_url = 'https://drive.google.com/file/d/1zQ1qmg6llqm29tpLa_sJT31umesJn6hn/view?usp=sharing'

movie_path = 'https://drive.google.com/uc?export=download&id='+movie_url.split('/')[-2]

movies = pd.read_csv(movie_path)

movies
# Abrindo um CSV no meu Google Drive com a nota do filme, dada por um usuário em um determinado timestamp



rating_url = 'https://drive.google.com/file/d/1i3habtRVO-C9T4fPk5elA6IgUBRtlEq4/view?usp=sharing'

rating_path = 'https://drive.google.com/uc?export=download&id='+rating_url.split('/')[-2]

ratings = pd.read_csv(rating_path)

ratings
movies.shape
ratings.shape
movies.columns
ratings.columns
movie_ratings = pd.merge(movies, ratings)

movie_ratings.columns
movie_ratings.head()
movie_ratings.shape