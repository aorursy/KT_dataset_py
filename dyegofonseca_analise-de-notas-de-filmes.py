# Importanto as bibliotecas principais



import pandas as pd

import numpy as np

import seaborn as sns
# Importando a base de dados do TMDB

tmdb = pd.read_csv('../input/dados-de-filmes/tmdb_5000_movies.csv')
tmdb.head()
tmdb.shape
tmdb['budget'].describe()
sns.distplot(tmdb.vote_average);
ax = sns.distplot(tmdb.vote_average)

ax.set(xlabel='Nota média', ylabel="Desidade")

ax.set_title('Média de votos em filmes no TMDB 5000');
ax = sns.distplot(tmdb.vote_average, norm_hist=False, kde = False)

ax.set(xlabel='Nota média', ylabel="Frequencia")

ax.set_title('Média de votos em filmes no TMDB 5000');
sns.boxplot(tmdb.vote_average);

ax.set_title('Distribuição de nota média')
tmdb.query('vote_average == 0')
tmdb[tmdb.vote_average == 0]
tmdb[tmdb.vote_count > 10].describe()
tmdb_com_mais_dez_votos = tmdb[tmdb.vote_count > 10]
sns.boxplot(tmdb.vote_average);

ax.set_title('Distribuição de nota média')
sns.boxplot(tmdb_com_mais_dez_votos['vote_average']);
notas = pd.read_csv('../input/dados-de-filmes/ratings.csv')
notas.head()
notas.shape
notas.groupby('movieId').mean()['rating']
media_por_filme = notas.groupby('movieId').mean()['rating']

media_por_filme
ax = sns.distplot(media_por_filme.values)

ax.set(xlabel='Nota média', ylabel="Densidade")

ax.set_title('Média de votos em filmes no MOVIE LENS');
notas.head()
quantidade_de_votos = notas.groupby('movieId').count()

quantidade_de_votos
quantidade_de_votos[quantidade_de_votos['rating'] >= 10]
quantidade_de_votos[quantidade_de_votos['rating'] >= 10].index
filme_com_pelo_menos_dez_votos = quantidade_de_votos[quantidade_de_votos['rating'] >= 10].index.values

filme_com_pelo_menos_dez_votos
media_por_filme.head()
notas_medias_do_filme_com_pelo_menos_dez_votos = media_por_filme.loc[filme_com_pelo_menos_dez_votos]

notas_medias_do_filme_com_pelo_menos_dez_votos
ax = sns.distplot(notas_medias_do_filme_com_pelo_menos_dez_votos)
sns.boxplot(notas_medias_do_filme_com_pelo_menos_dez_votos);
ax = sns.distplot(notas_medias_do_filme_com_pelo_menos_dez_votos, 

                  hist_kws={'cumulative':True}, 

                  kde_kws={'cumulative':True})

ax.set(xlabel='Nota média', ylabel="% acumulado dos filmes")

ax.set_title('Média dos filmes no movielens 100k com 10 ou mais votos');
ax = sns.distplot(tmdb_com_mais_dez_votos.vote_count)

ax.set(xlabel='Número do votos', ylabel='Densidade')

ax.set_title('Número de votos em filmes no TMDB 5000 com 10 ou mais votos')
tmdb.query("budget > 0").budget
tmdb[tmdb['budget'] > 0].budget
ax = sns.distplot(tmdb.query('budget > 0').budget)

ax.set(xlabel = 'Budget (gastos)', ylabel = 'Densidade')

ax.set_title('Gastos em filmes no TMDB 5000')
tmdb.query('popularity == 0').popularity
ax = sns.distplot(tmdb.popularity)

ax.set(xlabel = 'Popularidade', ylabel = 'Densidade')

ax.set_title('Popularidade em filmes no TMDB 5000')
ax = sns.distplot(tmdb.runtime)

ax.set(xlabel='Tempo de duração', ylabel='Densidade')

#ax.title('Tempo de duração em filmes do TMDB 5000')
tmdb.runtime.isnull().sum()
#Não fincionou essa filtragem para tirar os nulos

runtime_sem_nulos = tmdb[tmdb['runtime'] != tmdb.runtime.isnull()].runtime

runtime_sem_nulos.isnull().sum()
#Chamando a função dropna

runtime_sem_nulos = tmdb.runtime.dropna()

runtime_sem_nulos.isnull().sum()
ax = sns.distplot(runtime_sem_nulos)

ax.set(xlabel='Minutos', ylabel='Densidade')

ax.set_title('Duração do filme em minutos');
ax = sns.distplot(runtime_sem_nulos[runtime_sem_nulos > 0])

ax.set(xlabel='Minutos', ylabel='Densidade')

ax.set_title('Duração do filme em minutos');
ax = sns.distplot(runtime_sem_nulos[runtime_sem_nulos > 0],

                  hist_kws={'cumulative': True},

                  kde_kws={'cumulative':True})

ax.set(xlabel='Minutos', ylabel='Densidade')

ax.set_title('Duração do filme em minutos');
notas_medias_do_filme_com_pelo_menos_dez_votos.mean()
import matplotlib.pyplot as plt

import numpy as np



medias = []

for i in range(len(notas_medias_do_filme_com_pelo_menos_dez_votos)):

  medias.append(notas_medias_do_filme_com_pelo_menos_dez_votos[0:i].mean())



plt.plot(medias)
np.random.seed(75243)

temp = notas_medias_do_filme_com_pelo_menos_dez_votos.sample(frac = 1)



'''medias = list()

for i in range(len(temp)):

  medias.append(temp[0:i].mean())'''



[temp[0:i].mean() for i in range(len(temp))]



plt.plot(medias)

# Importando a biblioteca do Statsmodels para utilizar o Zscore

# Zscore é utilizado para saber quanto que o dado varia em relação a média em termos de desvio padrão



from statsmodels.stats.weightstats import zconfint

from statsmodels.stats.weightstats import ztest
zconfint(notas_medias_do_filme_com_pelo_menos_dez_votos)
from statsmodels.stats.weightstats import DescrStatsW
descr_todos_com_10_votos = DescrStatsW(notas_medias_do_filme_com_pelo_menos_dez_votos)

descr_todos_com_10_votos.tconfint_mean()
! ls -la 
filmes = pd.read_csv('../input/dados-de-filmes/movies.csv')
filmes.head()
np.random.seed(75241)



temp = notas.sample(frac = 1).rating



[temp[0:i].mean() for i in range(1, 100)]



plt.plot(medias)
tmdb.head()
tmdb[tmdb['original_title'] == 'Toy Story'].id
notas.head()
notas[notas['movieId'] == 1]
notas1 = notas[notas['movieId'] == 1]
zconfint(notas1.rating)
# Aplicando o Zteste (score Z) para uma amostra de 1000 da base de dados, reduzindo o tempo de execução do loop. 

# O zteste está sendo aplicado considerando um média 3.2099981127791013.

# O Zteste é aplicado para bases de dados grandes, e o Tteste utilizado para base de dados pequenas.



np.random.seed(75241)

temp = notas.sample(frac = 1).rating



def calcula_teste(i):

    media = temp[0:i].mean()

    stat, p = ztest(temp[0:i], value = 3.2099981127791013)

    return(i, media, p)



medias = np.array([calcula_teste(i) for i in range(1, 1000)])

medias
#Teste



[temp[0:i].mean() for i in range(1, 100)]



plt.plot(medias)
# Gráfico da média e do p-value



plt.plot(medias[:,0], medias[:,1])

plt.plot(medias[:,0], medias[:,2])

plt.hlines(y= 0.05, xmin=2, xmax = 1000, colors='g')
# Diferença entre o intervalo de confiança do notas1 e notas

print(ztest(notas1.rating, notas.rating))

zconfint(notas1.rating, notas.rating)
from scipy.stats import ttest_ind
ttest_ind(notas.rating, notas1.rating)
descr_todas_notas = DescrStatsW(notas.rating)

descr_toystory = DescrStatsW(notas1.rating)

print(descr_todas_notas)

print(descr_toystory)
comparacao = descr_todas_notas.get_compare(descr_toystory)

comparacao.summary()
notas1 = notas1.reset_index()
notas1 = notas1.drop(['index'], axis=1)

notas1
notas1.rating.head()
import matplotlib.pyplot as plt



plt.boxplot([notas.rating, notas1.rating], labels=['Todas as notas', 'Nota do Toy Story' ]);

plt.title('Distribuição da nota de acordo com o filme');
notas1['rating'].describe()['mean']
notas['rating'].head(len(notas1['rating'])).describe()
plt.boxplot([notas.rating, notas1[3:12].rating], labels=['Todas as notas', 'Nota do Toy Story (do 3 ao 10)' ]);

plt.title('Distribuição da nota de acordo com o filme');
descr_todas_as_notas = DescrStatsW(notas.rating)

descr_toystory = DescrStatsW(notas1[3:12].rating)

comparacao = descr_todas_as_notas.get_compare(descr_toystory)

comparacao.summary(use_t=True)
filmes.query('movieId in [1,593,72226]')
notas1 = notas.query('movieId == 1')

notas593 = notas.query('movieId == 593')

notas72226 = notas.query('movieId == 72226')



plt.boxplot([notas1.rating, notas593.rating, notas72226.rating], labels=['Toy Story', 'The Silence of the Lambs',  'Fantastic Mr. Fox']);

plt.title('Distribuição da nota de acordo com o filme');
sns.boxplot(x = 'movieId', y= 'rating', data = notas.query('movieId in (1,593,72226)'))
descr1 = DescrStatsW(notas1.rating)

descr593 = DescrStatsW(notas593.rating)

comparacao = descr1.get_compare(descr593)

comparacao.summary()
descr72226 = DescrStatsW(notas72226.rating)

descr593 = DescrStatsW(notas593.rating)

comparacao = descr72226.get_compare(descr593)

comparacao.summary()
comparacao = descr1.get_compare(descr72226)

comparacao.summary()
notas.query('movieId in (1, 593, 72226)').groupby('movieId').count()
from scipy.stats import normaltest
# Se p value é menor que o intervalo de confiança de 5%, descartamos a hipotese nula. Ou seja, se p < 0.05 os dados não vieram de uma distribuição normal. 

# Portanto não poderia aplicar Z ou T teste, já que os dados não são paramétricos



_, p  = normaltest(notas1.rating)

p
# Para o ranksum, a hipotese nula diz que as duas bases pertecem a mesma distribuição. Já a hipotese alteranativa diz que os samples são de distribuições diferentes.



from scipy.stats import ranksums



_, p = ranksums(notas1.rating, notas593.rating)

p