import pandas as pd

import matplotlib.pyplot as plt

links = pd.read_csv("../input/links.csv")

filmes = pd.read_csv("../input/movies.csv")

notas = pd.read_csv("../input/ratings.csv")

tags = pd.read_csv("../input/tags.csv")
links.head()
filmes.head()
notas.head()
tags.head()
notas.shape
notas.columns = ["usuarioId", "filmeId", "nota", "momento"]

notas.head
notas['nota'].unique()
notas['nota'].value_counts()
print("media",notas['nota'].mean())

print("mediana",notas['nota'].median())
notas.nota.head()
notas.nota.plot(kind='hist')
notas.nota.describe()
import seaborn as sns



sns.boxplot(notas.nota)
filmes.columns = ["filmeId", "titulo", "generos"]

filmes.head()
notas.head()
notas.query("filmeId==1").nota.mean()
notas.query("filmeId==2").nota.mean()
medias_por_filme = notas.groupby("filmeId").mean().nota

medias_por_filme.head()
medias_por_filme.plot(kind='hist')
plt.figure(figsize=(5,8))

sns.boxplot(y=medias_por_filme)
medias_por_filme.describe()
sns.distplot(medias_por_filme)
plt.hist(medias_por_filme)

plt.title("Histograma das médias dos filmes")