import numpy as np

import matplotlib.pyplot as plt

import math

import random

import pandas as pd

import scipy.stats as stat

from scipy.stats import norm



import os
from scipy.stats import t





def calcula_numero_desvios_tstudent_para_confianca(confianca, tamanho_amostra):

    mean = 0

    std = 1

    rv = t(df=(tamanho_amostra-1))

    return rv.interval(confianca)[1]
from scipy.stats import norm





def calcula_numero_desvios_normal_para_confianca(confianca):

    mean = 0

    std = 1

    rv = norm(loc=mean, scale=std)

    return rv.interval(confianca)[1]
def recupera_amostra(populacao, tamanho_amostra):

    return populacao[np.random.randint(0, len(populacao), tamanho_amostra)]
# Define população



mean = 160

std = 20

distribuicao_probabilidades = norm(loc=mean, scale=std)



tamanho_populacao = 100000



populacao = distribuicao_probabilidades.rvs(size=tamanho_populacao, random_state=1)



print('Tamanho população {}, média {}, desvio {}'.format(len(populacao), np.mean(populacao), np.std(populacao)))



fig, axs = plt.subplots(1, 1, figsize=(14,6))



axs.hist(populacao, density=True, facecolor='g', alpha=0.75, bins=50)

axs.grid(True)

axs.set_title('Histograma de alturas da População')



plt.show()
tamanho_amostra = 100

amostra = recupera_amostra(populacao, tamanho_amostra)

print(amostra)
fig, axs = plt.subplots(1, 1, figsize=(14,6))



axs.hist(amostra, density=True, facecolor='g', alpha=0.75, bins=50)

axs.grid(True)

axs.set_title('Histograma de alturas da amostra')



plt.show()




#Dada uma amostra de 100 elementos, calcule a média da altura da populacao considerando confiança de 93%



#1-Calcule a média da amostra ex. media_amostra = np.mean(amostra)



#2-Calcule o desvio da amostra ex. desvio_amostra = np.std(amostra)



#3-Calcule quantos desvios precisará para seu grau de confiânça ex. numero_desvios = calcula_numero_desvios_tstudent_para_confianca(confiança, tamanho_amostra)



#4-Calcule o desvio das amostras ex. desvio_amostras = desvio_amostra/np.sqrt(tamanho_amostra)



#5-Calcule a margem de erro ex. margem_erro = numero_desvios*desvio_amostras



#6-Calcule o intervalo ex. inferior = media_amostra-margem_erro, superior = media_amostra+margem_erro
#1-Calcule a média da amostra

media_amostra = np.mean(amostra)

print('Resultado da questao 1 é igual a {}'.format(media_amostra))
#2 Caluclar o desvio da amostra

desvio_amostra = np.std(amostra)

print('Resultado da questao 2 é igual a {}'.format(desvio_amostra))
#3 Calcular a quantidade de desvios necessários para grau de confiança 93%

numero_desvios = calcula_numero_desvios_tstudent_para_confianca (0.93,tamanho_amostra)

print('Resultado da questao 3 é igual a {}'.format(numero_desvios))
#4 Caluclar o desvio das amostras

desvio_amostras = desvio_amostra/np.sqrt(tamanho_amostra)

print('Resultado da questao 4 é igual a {}'.format(desvio_amostras))
#5 Calcular a margem de erro

margem_erro = numero_desvios*desvio_amostras

print('Resultado da questao 5 é igual a {}'.format(margem_erro))
#6 Calcular intervalo

inferior=media_amostra-margem_erro

superior=media_amostra+margem_erro

print('Resultado da questao 6 é igual a: A média da população para a amostra 1 de 100 elementos estará entre {} e {} com confiança {}'.format(inferior,superior,93))
# Obtenha uma amostra de 1000 elementos e calcule a média da altura da populacao considerando confiança de 80%
#1 Obter amostra de 1000 elementos

tamanho_amostra_1000=1000

amostra_1000=recupera_amostra(populacao, tamanho_amostra_1000)

print(amostra)
#2 Fazer o Gráfico de FDP Amostra 1000

fig, axs = plt.subplots (1,1, figsize=(14,6))

axs.hist(amostra_1000, density=True, facecolor='g', alpha=0.75, bins=50)

axs.grid (True)

axs.set_title ('Histograma de alturas da Amostra 1000')

plt.show()
#3 Obter intervalo de 80% de confiança da amostra de 1.000 elementos

media_amostra_1000 = np.mean(amostra_1000)

desvio_amostra_1000 = np.std(amostra_1000)

numero_desvios_1000 = calcula_numero_desvios_tstudent_para_confianca (0.80,tamanho_amostra_1000)

desvio_amostras_1000 = desvio_amostra_1000/np.sqrt(tamanho_amostra_1000)

margem_erro_1000 = numero_desvios_1000*desvio_amostras_1000

inferior_1000=media_amostra_1000-margem_erro_1000

superior_1000=media_amostra_1000+margem_erro_1000

print('Resultado da questao é igual a: A média da população para a amostra de 1.000 elementos estará entre {} e {} com confiança de {}'.format(inferior_1000,superior_1000,80))
# Obtenha uma amostra de 100 elementos e calcule a média da altura da populacao considerando confiança de 99%
#1 Obter amostra de 100 elementos

tamanho_amostra_100=100

amostra_100=recupera_amostra(populacao, tamanho_amostra_100)

print(amostra)
#2 Fazer o Gráfico de FDP Amostra 100

fig, axs = plt.subplots (1,1, figsize=(14,6))

axs.hist(amostra_1000, density=True, facecolor='g', alpha=0.75, bins=50)

axs.grid (True)

axs.set_title ('Histograma de alturas da Amostra 100')

plt.show()
#3 Obter intervalo de 99% de confiança da amostra de 100 elementos

media_amostra_100 = np.mean(amostra_100)

desvio_amostra_100 = np.std(amostra_100)

numero_desvios_100 = calcula_numero_desvios_tstudent_para_confianca (0.99,tamanho_amostra_100)

desvio_amostras_100 = desvio_amostra_100/np.sqrt(tamanho_amostra_100)

margem_erro_100 = numero_desvios_100*desvio_amostras_100

inferior_100=media_amostra_100-margem_erro_100

superior_100=media_amostra_100+margem_erro_100

print('Resultado da questao é igual a: A média da população para a amostra de 1.000 elementos estará entre {} e {} com confiança de {}'.format(inferior_100,superior_100,99))