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
mean = 0

std = 1

repeticoes=1000



rv_norm = stat.norm(loc=mean, scale=std)

populacao_norm = rv_norm.rvs(size=repeticoes, random_state=random_state)



mean_ = np.mean(populacao_norm)

std_ = np.std(populacao_norm)



print('População normal, média {}, desvio padrão {}'.format(mean_, std_))



populacao_norm += np.linspace(0.0,5*std,repeticoes)



mean_ = np.mean(populacao_norm)

std_ = np.std(populacao_norm)



print('População normal alterada, média {}, desvio padrão {}'.format(mean_, std_))



rv = norm(loc=mean_, scale=std_)



intervalo = np.linspace(mean_-3*std_,mean_+3*std_, num=50)







fig, axs = plt.subplots(1, 2, figsize=(14,6))



axs[0].plot(populacao_norm,'.')

axs[0].grid(True)

axs[0].set_title('populacao_norm')



axs[1].hist(populacao_norm, density=True, facecolor='g', alpha=0.75, bins=50)

axs[1].plot(intervalo, rv.pdf(intervalo), 'k-', label='pdf')

axs[1].grid(True)

axs[1].set_title('populacao_norm')



plt.show()
tamanho_serie = len(populacao_norm) 

mean_ = np.mean(populacao_norm[0:int(tamanho_serie/2)])

print('Média da primeira metade da população {}'.format(mean_))

mean_ = np.mean(populacao_norm[int(tamanho_serie/2):])

print('Média da segunda metade da população {}'.format(mean_))
# Hipótese nula: não é estacionária

# p>0.05 não rejeita hipótese nula



from statsmodels.tsa.stattools import adfuller

result = adfuller(populacao_norm)

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])
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