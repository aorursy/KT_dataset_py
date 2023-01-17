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



tamanho_amostra = 100

confianca = 0.93



#1-Calcule a média da amostra ex. media_amostra = np.mean(amostra)

media_amostra = np.mean(amostra)

print ("Média da amostra {}:". format (media_amostra))



#2-Calcule o desvio da amostra ex. desvio_amostra = np.std(amostra)

desvio_amostra = np.std(amostra)

print ("Desvio da amostra {}:". format (desvio_amostra))



#3-Calcule quantos desvios precisará para seu grau de confiânça ex. numero_desvios = calcula_numero_desvios_tstudent_para_confianca(confiança, tamanho_amostra)

numero_desvios = calcula_numero_desvios_tstudent_para_confianca(0.93, tamanho_amostra)

print ("Numero de desvios {}:". format (numero_desvios))



#4-Calcule o desvio das amostras ex. desvio_amostras = desvio_amostra/np.sqrt(tamanho_amostra)

desvio_amostras = desvio_amostra/np.sqrt(tamanho_amostra)

print ("Desvio das amostras {}:". format (desvio_amostras))



#5-Calcule a margem de erro ex. margem_erro = numero_desvios*desvio_amostras

margem_erro = numero_desvios*desvio_amostras

print ("Margem de erro {}:". format (margem_erro))



#6-Calcule o intervalo ex. inferior = media_amostra-margem_erro, superior = media_amostra+margem_erro

inferior = media_amostra-margem_erro

print ("Limite inferior {}:". format (inferior))

superior = media_amostra+margem_erro

print ("Limite superior {}:". format (superior))



print('A média da população estará entre {} e {} com confiança {}'.format(inferior,superior,93))
# Obtenha uma amostra de 1000 elementos e calcule a média da altura da populacao considerando confiança de 80%

tamanho_amostra_2 = 1000

amostra_2 = recupera_amostra(populacao, tamanho_amostra_2)

print(amostra_2)
fig, axs = plt.subplots(1, 1, figsize=(14,6))



axs.hist(amostra_2, density=True, facecolor='g', alpha=0.75, bins=50)

axs.grid(True)

axs.set_title('Histograma de alturas da amostra_2')



plt.show()
# Obtenha uma amostra de 1000 elementos e calcule a média da altura da populacao considerando confiança de 80%



tamanho_amostra = 1000

amostra = recupera_amostra(populacao, tamanho_amostra)

print(amostra)

confianca = 0.80



media_amostra = np.mean(amostra)

print ("Média da amostra {}:". format (media_amostra))



desvio_amostra = np.std(amostra)

print ("Desvio da amostra {}:". format (desvio_amostra))



numero_desvios = calcula_numero_desvios_tstudent_para_confianca(0.80, tamanho_amostra)

print ("Numero de desvios {}:". format (numero_desvios))



desvio_amostras = desvio_amostra/np.sqrt(tamanho_amostra)

print ("Desvio das amostras {}:". format (desvio_amostras))



margem_erro = numero_desvios*desvio_amostras

print ("Margem de erro {}:". format (margem_erro))



inferior = media_amostra-margem_erro

print ("Limite inferior {}:". format (inferior))

superior = media_amostra+margem_erro

print ("Limite superior {}:". format (superior))



print('A média da população estará entre {} e {} com confiança {}'.format(inferior,superior,80))
# Obtenha uma amostra de 100 elementos e calcule a média da altura da populacao considerando confiança de 99%
tamanho_amostra_3 = 100

amostra_3 = recupera_amostra(populacao, tamanho_amostra_3)

print(amostra_3)
fig, axs = plt.subplots(1, 1, figsize=(14,6))



axs.hist(amostra_3, density=True, facecolor='g', alpha=0.75, bins=50)

axs.grid(True)

axs.set_title('Histograma de alturas da amostra_3')



plt.show()
# Obtenha uma amostra de 100 elementos e calcule a média da altura da populacao considerando confiança de 99%



tamanho_amostra = 100

amostra = recupera_amostra(populacao, tamanho_amostra)

print(amostra)

confianca = 0.99



media_amostra = np.mean(amostra)

print ("Média da amostra {}:". format (media_amostra))



desvio_amostra = np.std(amostra)

print ("Desvio da amostra {}:". format (desvio_amostra))



numero_desvios = calcula_numero_desvios_tstudent_para_confianca(0.99, tamanho_amostra)

print ("Numero de desvios {}:". format (numero_desvios))



desvio_amostras = desvio_amostra/np.sqrt(tamanho_amostra)

print ("Desvio das amostras {}:". format (desvio_amostras))



margem_erro = numero_desvios*desvio_amostras

print ("Margem de erro {}:". format (margem_erro))



inferior = media_amostra-margem_erro

print ("Limite inferior {}:". format (inferior))

superior = media_amostra+margem_erro

print ("Limite superior {}:". format (superior))



print('A média da população estará entre {} e {} com confiança {}'.format(inferior,superior,99))