import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.cluster import KMeans

import seaborn as sns

import os

print(os.listdir("../input"))
df = pd.read_csv('../input/mortalidade-indigena-caner-mato-grosso/BANCO CANCER 10.06.csv')
df.head()
df['Idade'] = df['Idade'].apply(lambda x: int(x.replace('a','').replace('m','')))
def faixaEtaria(idade):

    if idade > 0 and idade < 5:

        return 1

    elif idade >= 5 and idade < 10:

        return 2

    elif idade >= 10 and idade < 15:

        return 3

    elif idade >= 15 and idade < 20:

        return 4

    elif idade >= 20 and idade < 30:

        return 5

    elif idade >= 30 and idade < 40:

        return 6

    elif idade >= 40 and idade < 50:

        return 7

    elif idade >= 50 and idade < 60:

        return 8

    elif idade >= 60 and idade < 70:

        return 9

    elif idade >= 70 and idade < 80:

        return 10

    elif idade >= 80 and idade < 90:

        return 11

    elif idade >= 90 and idade < 100:

        return 12

    elif idade >= 100:

        return 13



df['cdFaixaEtaria'] = df['Idade'].apply(faixaEtaria)

faixasEtariasDict = {

    1: '0 a 5',

    2: '5 a 10',

    3: '10 a 15',

    4: '15 a 20',

    5: '20 a 30',

    6: '30 a 40',

    7: '40 a 50',

    8: '50 a 60',

    9: '60 a 70',

    10: '70 a 80',

    11: '80 a 90',

    12: '90 a 100',

    13: '100+'

}



def codigoSexo(sexo):

    return 0 if sexo == 'Feminino' else 1



df['FaixaEtaria'] = df['cdFaixaEtaria'].apply(lambda x: faixasEtariasDict.get(int(x)))

df['cdSexo'] = df['Sexo'].apply(codigoSexo)
df['cdCausa'] = df['CausaCid103C'].apply(lambda x: int(x[1:3]))

df['CausaCodigo'] = df['CausaCid103C'].apply(lambda x: x[0:3])

df.head()
df.Idade.plot.hist()
df.cdFaixaEtaria.plot.hist()
df.cdSexo.plot.hist()
sns.pairplot(data=df)
kmeans = KMeans(n_clusters=3)

kmeans = kmeans.fit(df.Idade.values.reshape(-1,1))

labels = kmeans.predict(df.Idade.values.reshape(-1,1))



C = kmeans.cluster_centers_

print(labels,C)
dfIdadeGrupo = pd.concat([df,pd.DataFrame(labels, columns= ['Grupo'])], axis=1, join='inner')

dfIdadeGrupo.head()
cores = dfIdadeGrupo.Grupo.map({0:'b',1:'r',2:'y'})

dfIdadeGrupo.plot.scatter(x='Idade',y='cdCausa', c=cores)
kmeans = KMeans(n_clusters=3)

kmeans = kmeans.fit(df.cdCausa.values.reshape(-1,1))

labels = kmeans.predict(df.cdCausa.values.reshape(-1,1))



C = kmeans.cluster_centers_

print(labels,C)
dfCausaGrupo = pd.concat([df,pd.DataFrame(labels, columns= ['Grupo'])], axis=1, join='inner')

dfCausaGrupo.head()
cores = dfCausaGrupo.Grupo.map({0:'b',1:'r',2:'y'})

dfCausaGrupo.plot.scatter(x='Idade',y='cdCausa', c=cores)
plt.figure(figsize=(30,30))

sns.heatmap(df.corr(),annot=True,linewidths=0.01 ,cmap='coolwarm', cbar=False)

plt.show()
#adiciona coluna com count agrupado por causa e sexo, servirá para tamanho dos pontos no gráfico scatter

df['CasosPorCausa'] = df.groupby(['CausaCodigo', 'Sexo'])['CausaCodigo'].transform('count')

df.head()
agrupado = df.groupby(['CausaCodigo', 'Sexo', 'Idade'])

agrupadoCount = agrupado['Idade'].agg(['count'])

agrupadoCount
plt.scatter(df['Idade'], df['cdCausa'], s=df['CasosPorCausa']*10, c=df['cdSexo'], alpha=0.5)
plt.scatter(df['cdFaixaEtaria'], df['cdCausa'], s=df['CasosPorCausa']*10, c=df['cdSexo'], alpha=0.5)