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
if IN_KAGGLE:

    df = pd.read_csv("../input/2017.csv")

else:

    df = pd.read_csv("2017.csv")

    



df.head(20)
df = df.loc[:,[  'Happiness.Score',  'Economy..GDP.per.Capita.', 'Family',

       'Health..Life.Expectancy.', 'Freedom', 'Generosity',

       'Trust..Government.Corruption.','Dystopia.Residual']]
plt.figure(figsize=(14,6))



_ = df['Happiness.Score'].hist( bins=50, density=True)



plt.xlabel('Índice de felicidade')

plt.ylabel('Probabilidade')

plt.title('Histogram do índice de felicidade')



plt.show()
df.corr()
# Aparentemente economia é fortemente correlacionada com felicidade



_ = df.plot(figsize=(14,6), kind='scatter', x='Happiness.Score', y='Economy..GDP.per.Capita.' )
# Já a generosidade não apresenta correlação significativa



plt.figure()



_ = df.plot(figsize=(14,6),kind='scatter', x='Happiness.Score', y='Generosity' )
from pandas.plotting import scatter_matrix



_ = scatter_matrix(df, figsize=(14,10), alpha=0.2, diagonal='kde')

#df.plot(figsize=(14,6), kind='scatter', x='Happiness.Score', y='Economy..GDP.per.Capita.' )

np.cov(df[['Happiness.Score','Economy..GDP.per.Capita.']].values.T)
#df.plot(figsize=(14,6), kind='scatter', x='Happiness.Score', y='Economy..GDP.per.Capita.' )

np.corrcoef(df[['Happiness.Score','Economy..GDP.per.Capita.']].values.T)
from scipy.stats import spearmanr



rho, pval = spearmanr(df[['Happiness.Score','Economy..GDP.per.Capita.']].values)

print(rho)

print(pval)
x=np.arange(0, 8.1, 0.05)

y = np.sin(np.pi*x)



fig, ax1 = plt.subplots(figsize=(14,8))

ax1.plot( range(len(x)),y)

plt.grid()

plt.tight_layout()

plt.show()



print(len(x))



print(np.corrcoef(x,y))
# https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/



from statsmodels.graphics.tsaplots import plot_acf



fig, ax = plt.subplots(1,1,figsize=(14,8))

plot_acf(y,ax=ax)



plt.show()
x=np.arange(0, 8.1, 0.05)

y1 = np.sin(np.pi*x)

y2 = np.cos(np.pi*(x+1/2))



fig, ax = plt.subplots(2,1,figsize=(14,8))

ax[0].plot(y1)

ax[1].plot(y2)

plt.tight_layout()

plt.show()





print(np.corrcoef(y1,y2))
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