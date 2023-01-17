# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importando as Bibliotecas

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



import matplotlib.ticker as ticker

from sklearn import datasets, linear_model

from sklearn import linear_model

from sklearn.linear_model import LinearRegression

from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler

from sklearn.manifold import TSNE
# Importando e apresentando o arquivo

arquivo = pd.read_csv('/kaggle/input/top-spotify-tracks-of-2018/top2018.csv', encoding='utf-8')

arquivo.head()
arquivo.info()
arquivo.corr()
# Histograma com a distribuição das variáveis

arquivo.hist()

plt.show()
#Plotar um gráfico Heatmap com a correlação entre as variáveis. 



plt.suptitle("Gráfico - Correlação entre variáveis")

sns.heatmap(arquivo.corr(),cmap='YlOrRd')

plt.show()
Correlacao=arquivo[['danceability','energy','valence','loudness','tempo']]

Correlacao
#Plotar um gráfico Heatmap com a correlação mais fortes entre as variáveis. 



plt.suptitle("Gráfico - Correlações mais fortes")

sns.heatmap(Correlacao.corr(), annot=True, cmap="viridis")

plt.show()
#Gráfico Joinplot usando como base os dados correlacionados e uma regressão linear entre as variáveis energy e loudness. 



sns.jointplot(data=Correlacao,y='energy',x='loudness',kind='reg', color='r')
#Análise descritiva

arquivo.describe().T
# A duração das músicas.

arquivo['Duration_min']=arquivo['duration_ms']/60000

print(arquivo)
#Gráfico com a distribuição da duração das músicas em milisegundos

plt.figure(figsize=(6,4))

sns.swarmplot(arquivo['duration_ms'], color='r')
plt.suptitle("Gráfico - Identificação dos artistas")

sns.barplot(data=arquivo.sort_values(by='key', ascending=False).head(15), x='key', y='artists');
#Postando os 10 primeiros artistas na lista das músicas mais tocadas:

arquivo['artists'].value_counts().head(10)
#Artista 1 = Post Malone

art1 = arquivo[arquivo['artists']=='Post Malone']

art1[['name','danceability','energy','loudness','valence','tempo']]
#Artista 2 = XXXTENTACION

art2 = arquivo[arquivo['artists']=='XXXTENTACION']

art2[['name','danceability','energy','loudness','valence','tempo']]
#Artista3 = Drake

art3 = arquivo[arquivo['artists']=='Drake']

art3[['name','danceability','energy','loudness','valence','tempo']]
# A um nível de significância de 0.5% quais as músicas mais dançantes:

mu=arquivo['danceability']>=0.75

re=(arquivo['danceability']>=0.5) & (arquivo['danceability']<0.75)

nd=arquivo['danceability']<0.5



data=[mu.sum(),re.sum(),nd.sum()]

Dance=pd.DataFrame(data,columns=['percent'],

                   index=['Muito','Regular','Ñ Dançante/Instrumental'])

Dance
#Plotando gráfico de distribuição de danceabilidade

plt.suptitle("Gráfico - Distribuição de Danceabilidade das músicas")

sns.distplot(arquivo['danceability'], color='y')
#Regressão linear entre danceabilidade e Valencia das músicas

x1 = arquivo["danceability"].values

y2 = arquivo["valence"].values



x = x1.reshape(x1.shape[0], 1)

y = y2.reshape(y2.shape[0], 1)



regressao = linear_model.LinearRegression()

regressao.fit(x, y)



fig= plt.figure(figsize=(7,5))

plt.suptitle("Correlação entre as variáveis Danceability e Valence")



ax = plt.subplot(1,1,1)

ax = plt.scatter(x1, y2, alpha=0.5, color='blue')

ax = plt.plot(x1, regressao.predict(x), color='red', linewidth=2)

plt.xticks(())

plt.yticks(())



plt.xlabel("danceability")

plt.ylabel("valence")



plt.show()
#Criar tabela com as 10 músicas mais dançantes

arquivo[['name','artists','danceability','valence','tempo']].sort_values(by='danceability', ascending=False).head(10)
#Montando tabela com análise de intensidade de energia a partir do nível de significância de 0.5%

mu=arquivo['energy']>=0.75

re=(arquivo['energy']>=0.5) & (arquivo['energy']<0.75)

nd=arquivo['energy']<0.5



data=[mu.sum(),re.sum(),nd.sum()]

energy=pd.DataFrame(data,columns=['Percentual'],

                   index=['Alta','Regular','Pouca energia'])

energy
#Plotando gráfico de distribuição de energia

plt.suptitle("Gráfico - Distribuição de energia")

sns.distplot(arquivo['energy'])
#Criar tabela com as 10 músicas maior energia

arquivo[['name','artists','energy','valence','tempo']].sort_values(by='energy', ascending=False).head(10)
#Gráfico Scatterplot com indicando a correlação e distribuição das variáveis.

plt.figure(figsize=(6,5))

arquivo.plot(title = 'Análise danceabilidade e energia de uma musica', kind='scatter', x='danceability',y='energy',color='red')

plt.figure(figsize=(6,5))

arquivo.plot(title = 'Análise', kind='scatter', x='loudness',y='speechiness',color='g')

plt.figure(figsize=(6,5))

arquivo.plot(title = 'Análise', kind='scatter', x='valence',y='liveness',color='b') 
# Criar um classificação entre os tempos das músicas para definir os ritmos.



arquivo.loc[arquivo['tempo']>168,'Ritmo']='Presto'

arquivo.loc[(arquivo['tempo']>=110) & (arquivo['tempo']<=168),'Ritmo']='Allegro'

arquivo.loc[(arquivo['tempo']>=76) & (arquivo['tempo']<=108),'Ritmo']='Andante'

arquivo.loc[(arquivo['tempo']>=66) & (arquivo['tempo']<=76),'Ritmo']='Adagio'

arquivo.loc[arquivo['tempo']<65,'Ritmo']='Lento'



arquivo['Ritmo'].value_counts()
sns.set_style(style='dark')

rit1 = arquivo['Ritmo'].value_counts()

rit2 = pd.DataFrame(rit1)



sns.barplot(x=rit1, y=rit2.index, data=arquivo)

plt.title("Gráfico - Classificação do Andamento musical")

plt.show()