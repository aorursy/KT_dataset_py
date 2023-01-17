# importar pacotes necessários

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

#from sklearn.utils import shuffle
# definir parâmetros extras

import warnings

warnings.filterwarnings("ignore")

sns.set(style="white", color_codes=True)
# carregar arquivo de dados de treino

data = pd.read_csv('../input/zoo-train.csv', index_col='animal_name')



# embaralhar linhas

#data = shuffle(data)

data = data.sample(frac=1)



# deixar coluna como categórica

data['class_type'] = data['class_type'].astype('category')



# mostrar alguns exemplos de registros

data.head()
# 1-7 is Mammal, Bird, Reptile, Fish, Amphibian, Bug and Invertebrate

animal_type = ['Mammal', 'Bird', 'Reptile', 'Fish', 'Amphibian', 'Bug', 'Invertebrate']



data['class_name'] = data['class_type'].map(lambda x: animal_type[x-1])



data.iloc[:,-2:].head()
# quantos registros existem de cada espécie?

data['class_type'].value_counts()
sns.countplot(data['class_name'])
data.legs.unique()
# just curious which animal has 5 legs

data.loc[data['legs'] == 5][['class_type', 'class_name']]
sns.countplot(data['legs'])
# gerar gráfico para analisar pares de características

#sns.pairplot(data, hue="class_type", size=3)
# gerar gráfico em pares com kde nas diagonais

#sns.pairplot(data, hue="class_type", size=3, diag_kind="kde")
data.groupby('class_name').mean()
g = sns.FacetGrid(data, col="class_name")

g.map(plt.hist, "legs")

plt.show()