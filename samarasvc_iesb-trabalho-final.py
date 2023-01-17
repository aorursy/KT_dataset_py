# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Carregando os dados

df = pd.read_csv('../input/StudentsPerformance.csv')
# Verificando os dados

df.head()
# Verificando o nome das colunas

df.columns
# Características do conjunto de dados

df.info()
# Verificando valores nulos no conjunto de dados

df.isnull().sum()
# Verificando recursos descritos no conjunto de dados

df.describe()
f,ax = plt.subplots(figsize=(15,6))

sns.heatmap(df.corr(), annot=True, fmt='.2f', ax=ax, linecolor='black', lw=.7)
sns.pairplot(df, hue = 'gender')

plt.show()
# Visualizando o número de homens e mulheres no conjunto de dados



df['gender'].value_counts(normalize = True)

df['gender'].value_counts(dropna = False).plot.bar(color = 'magenta')

plt.title('Comparassão entre Homens e Mulheres')

plt.xlabel('Genero')

plt.ylabel('Quantidade')

plt.show()
# Visualizando os diferentes grupos no conjunto de dados



df['race/ethnicity'].value_counts(normalize = True)

df['race/ethnicity'].value_counts(dropna = False).plot.bar(color = 'cyan')

plt.title('Comparação entre vários grupos')

plt.xlabel('Grupos')

plt.ylabel('Qunatidade')

plt.show()
# Visualizando os diferentes níveis de educação dos pais



df['parental level of education'].value_counts(normalize = True)

df['parental level of education'].value_counts(dropna = False).plot.bar()

plt.title('Comparação entre edução dos pais')

plt.xlabel('Grau')

plt.ylabel('Quantidade')

plt.show()
# Distribuição das Notas de matemática

plt.figure(figsize=(8,5))

sns.distplot(df['math score'], kde = False, color='m', bins = 30)

plt.ylabel('Frequência')

plt.title('Distribuição das notas de matemática')

plt.show()
# Comparação entre Genêro vs Raça 



x = pd.crosstab(df['gender'], df['race/ethnicity'])

x.div(x.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (4, 4))
# Comparação de raça / etnia e nível de educação dos pais



x = pd.crosstab(df['race/ethnicity'], df['parental level of education'])

x.div(x.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = 'True', figsize = (7, 4) )
# Gráfico de pizza para representar a proporção de status de aprovação e reprovação entre os alunos



size = [960, 40]

colors = ['red', 'pink']

labels = "Aprovado", "Reprovado"

explode = [0, 0.2]



plt.pie(size, colors = colors, labels = labels, autopct = '%.2f%%', explode = explode, shadow = True)

plt.legend()

plt.show()
# Traçar um gráfico de pizza para a distribuição de várias notas entre os alunos



labels = ['Grade 0', 'Grade A', 'Grade B', 'Grade C', 'Grade D', 'Grade E']

sizes = [58, 156, 260, 252, 223, 51]

colors = ['yellow', 'gold', 'lightskyblue', 'lightcoral', 'pink', 'cyan']

explode = (0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001)



patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)

plt.legend(patches, labels)

plt.axis('equal')

plt.tight_layout()

plt.show()

# Número de alunos com pontuação máxima em todas as três categorias

perfect_writing = df['writing score'] == 100

perfect_reading = df['reading score'] == 100

perfect_math = df['math score'] == 100



perfect_score = df[(perfect_math) & (perfect_reading) & (perfect_writing)]

perfect_score
# Bar Plot de Notas de acordo com o gênero

plt.figure(figsize=(10,4))



plt.subplot(1,3,1)

sns.barplot(x = 'gender', y = 'reading score', data = df)



plt.subplot(1,3,2)

sns.barplot(x = 'gender', y = 'writing score', data = df)



plt.subplot(1,3,3)

sns.barplot(x = 'gender', y = 'math score', data = df)



plt.tight_layout()