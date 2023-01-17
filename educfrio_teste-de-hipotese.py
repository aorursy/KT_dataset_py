'''

As bibliotecas usadas são:

random

statistic

numpy.random

scipy.stats

pandas

matplotlib

statsmodels

pandas-profiling

'''



import numpy as np

import matplotlib.pyplot as plt

import math

import random

import pandas as pd

import scipy.stats as stat



import os



path = os.environ['PATH']



if path.startswith('C'):

    IN_KAGGLE = False

else:

    IN_KAGGLE = True
# Para uso com funções da biblioteca standard (ex random.randint)

random.seed(1)

# Para uso com funções da biblioteca numpy (ex np.random.randint)

np.random.seed(1)



# Quando for passada como parâmetro a seed

random_state = 1
# Permutação: possibilidades de colocação de n objetos em n posições = n!

def permutacao (n):

    return math.factorial(n)



# Arranjo: p objetos em n posições, ordem importa = n!/(n-p)!

def arranjo (n,p):

    return math.factorial(n)/math.factorial(n-p)



# Combinação: p objetos em n posições, ordem não importa = n!/(n-p)!p!

def combinacao (n,p):

    return math.factorial(n)/(math.factorial(n-p)*math.factorial(p))



# Variações possíveis havendo n slots e p possibilidades para cada um

def possibilidades(n,p):

    return p**n
from scipy.stats import norm



mean = 0

std = 1

rv = norm(loc=mean, scale=std)

print('Desvios que englobam 95% dos pontos {}'.format(rv.interval(0.95)    ))

#ponto = 1.959963984540054

ponto = 1.092

print('Probabilidade de ter um ponto além de {} desvios {}'.format(ponto,rv.sf(ponto)))



print('Probabilidade de ter um ponto além ou aquem de {} desvios {}'.format(ponto,2*rv.sf(ponto)))
from scipy.stats import norm



# desvios que contêm 95% dos pontos

mean = 0

std = 1

rvt = t(df=(100-1))

print('Desvios que englobam 95% dos pontos {}'.format(rvt.interval(0.95)    ))

#ponto = 1.959963984540054

ponto = 1.092

print('Probabilidade de ter um ponto além de {} desvios {}'.format(ponto,rvt.sf(ponto)))



print('Probabilidade de ter um ponto além ou aquem de {} desvios {}'.format(ponto,2*rvt.sf(ponto)))



#A população é normal com média 0 e desvio 1

#Criando amostra com 100 elementos



import numpy as np

from scipy.stats import t



amostra = rv.rvs(size=100)

#Cria a distribuição das amostras

desvio_distribuicao_amostras = std/np.sqrt(100)

dist_amostras = norm(loc=mean, scale=desvio_distribuicao_amostras)

dist_amostras_t = t(100-1,loc=mean, scale=desvio_distribuicao_amostras)



media_amostra = amostra.mean()



numero_desvios_amostra = np.abs( (media_amostra-0)/desvio_distribuicao_amostras )



print('Media {} numero_desvios_amostra {} amostra'.format(media_amostra,

                                                                    numero_desvios_amostra))
print('Probabilidade de ter um ponto além ou aquem de numero_desvios_amostra {}'.format(

    2*rv.sf(numero_desvios_amostra)))
print('Probabilidade de ter um ponto além ou aquem de numero_desvios_amostra {}'.format(

    2*rvt.sf(numero_desvios_amostra)))
from scipy import stats



# Two sided

stats.ttest_1samp(amostra,0)

from scipy.stats import binom

tentativas = 30

rv_honesta = binom(tentativas, 1/2)

populacao_honesta = rv_honesta.rvs(size=1000000, random_state=random_state)

print(rv_honesta.mean())

print(rv_honesta.std())

rv = binom(tentativas, 1/2.5)

resultado = rv.rvs(size=100, random_state=random_state)

print(np.mean(resultado))
amostras = 500

medias = np.zeros((amostras,1))

np.random.seed(1)

for i in range(0,amostras,1):

    medias[i]=np.mean(populacao_honesta[np.random.randint(0, len(populacao_honesta),100)])



print(medias.mean())

print(medias.std())    



fig, axs = plt.subplots(1, 1, figsize=(14,6))





axs.hist(medias, density=True, facecolor='g', alpha=0.75, bins=50)

axs.grid(True)

axs.set_title('Distribuição das médias')



plt.show()
rv_norm = stat.norm(loc=medias.mean(), scale=medias.std())
rv_norm.cdf(11.76)
if IN_KAGGLE:

    df = pd.read_csv("../input/2016.csv")

else:

    df = pd.read_csv("2016.csv")

df.head(20)
df.Region.unique()
dfWE = df.loc[df.Region == 'Western Europe',['Country', 'Region', 'Happiness Rank', 'Happiness Score',

       'Lower Confidence Interval', 'Upper Confidence Interval',

       'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',

       'Freedom', 'Trust (Government Corruption)', 'Generosity',

       'Dystopia Residual']]



dfLC = df.loc[df.Region == 'Latin America and Caribbean',['Country', 'Region', 'Happiness Rank', 'Happiness Score',

       'Lower Confidence Interval', 'Upper Confidence Interval',

       'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',

       'Freedom', 'Trust (Government Corruption)', 'Generosity',

       'Dystopia Residual']]
fig, axs = plt.subplots(1, 2, figsize=(14,6))



axs[0].hist(dfWE['Happiness Score'], density=True, facecolor='g', alpha=0.75)

axs[0].grid(True)

axs[0].set_title('Western Europe')



print(dfWE['Happiness Score'].mean())

print(dfWE['Happiness Score'].std())



axs[1].hist(dfLC['Happiness Score'], density=True, facecolor='g', alpha=0.75)

axs[1].grid(True)

axs[1].set_title('Latin America and Caribbean')



print(dfLC['Happiness Score'].mean())

print(dfLC['Happiness Score'].std())
dfWE
dfLC
from scipy import stats



stats.ttest_ind(dfWE['Happiness Score'].values,dfLC['Happiness Score'].values, equal_var=False)
from scipy import stats 



stats.ttest_ind(dfWE['Happiness Score'].values,dfLC['Happiness Score'].values, equal_var=True)

# Gerando int - biblioteca python standard

print(random.randrange(100, 1000, 2))

print(random.randint(100, 1000))



# Gerando int - biblioteca numpy

print(np.random.randint(100, 1000,2))



# Gerando float - biblioteca python standard

print(random.random())

print(random.uniform(100, 1000))

print(random.normalvariate(1, 1))



# Gerando float - biblioteca numpy

print(np.random.random(5))

print(np.random.randn(5))



np.random.random_sample(size=100)
print(np.linspace(0.0,1.0,11))

print(np.arange(0.0,10.0,3))

print(np.logspace(0.0,10.0,3))

# Escolha com reposição

# usando numpy np.random.choice(10,size=10,replace=True)





faces = list(range(1,7))

lancamentos = 600

pesos = [1/6,1/6,0.5/6,0.5/6,2/6,1/6]

resultados = random.choices(population=faces, weights=pesos, k=lancamentos)

#print(resultados)

for i in faces:

    print('Face {}, peso {}, vezes {}'.format(i,pesos[i-1],resultados.count(i)))
# Escolha sem reposição

# usando numpy np.random.choice(10,size=10,replace=False)





lista = list(range(1,7))

random.sample(population=lista, k=len(lista))

# Embaralhamento

# usando numpy np.random.choices



lista = list(range(1,7))

random.shuffle(lista)

lista