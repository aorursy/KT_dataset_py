import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv('../input/movie_metadata.csv')

data.sample(3)
data.shape # Número de observações
col = ['title_year','duration','budget','num_critic_for_reviews','movie_facebook_likes','num_voted_users','gross','imdb_score']
data = data[col]

data.shape # Número de observações
data.sample(3)
# Dados Faltantes

total = data.isnull().sum().sort_values(ascending=False)

percent = (data.isnull().sum()*100/data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Porcentagem'])

missing_data.head(20)
# Remover Observacões com algum dado ausente

data = data.dropna()

print(data.isnull().sum().max())

data.shape
data.sample(5)
sum(data['movie_facebook_likes'] == 0)
import seaborn as sns

sns.boxplot(x=data['movie_facebook_likes'] )
import seaborn as sns

 

sns.set(style="ticks")

sns.pairplot(data[0:50], kind="scatter", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))

from numpy import array

from numpy import mean

from numpy import cov

from numpy.linalg import eig



A = data

M = mean(A.T, axis=1)

# center columns by subtracting column means

C = A - M

# calculate covariance matrix of centered matrix

V = cov(C.T)
# eigendecomposition of covariance matrix

values, vectors = eig(V)

print(values)
pd.DataFrame(data=[values], columns=[1,2,3,4,5,6,7,8], index=['CP'])
print(vectors.shape)

print(vectors)
pd.DataFrame(data=vectors, columns=[1,2,3,4,5,6,7,8], index=[1,2,3,4,5,6,7,8])
values_perc = values*100/sum(values)

pd.DataFrame(data=[values_perc], columns=[1,2,3,4,5,6,7,8], index=['CP %'])
# project data

import matplotlib.pyplot as plt

P = vectors.T.dot(C.T)

print(P.shape)

range = 50

plt.plot(P[0,0:range], P[1,0:range], 'o', markersize=6, color='blue', alpha=0.5, label='Classe 1')

plt.plot(P[0,range:range*2], P[1,range:range*2], '^', markersize=7, color='red', alpha=0.5, label='Classe 2')

plt.xlabel('CP1 91,1%')

plt.ylabel('CP2 8,8%')

plt.legend()

plt.title('Componentes Principais')



plt.show()