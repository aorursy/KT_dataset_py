import numpy as np

import matplotlib.pyplot as plt

import math

import random 

import pandas as pd

import scipy.stats as stat

# Por que executou o form importanto a normal nesse quadro?

from scipy.stats import norm



import os
#Desvios T Student



from scipy.stats import t



def calcula_numero_desvios_tstudent_para_confianca (confianca, tamanho_amostra):

    mean = 0

    std = 1

    rv = t(df=(tamanho_amostra-1))

    return rv.interval (confianca)[1]



#A fórmula acima é do T Student?
#Desvios Normal 



from scipy.stats import norm

def calcula_numero_desvios_normal_para_confiança (confianca):

    mean=0

    std=1

    return rv.interval(confiaca)[1]
def recupera_amostra(populacao, tamanho_amostra):

    return populacao [np.random.randint (0, len(populacao), tamanho_amostra)]
#Definir conceitos da população a ser trabalhada 

mean=160

std=20

distribuicao_probabilidades=norm(loc=mean, scale=std)

tamanho_populacao = 100000

populacao=distribuicao_probabilidades.rvs(size=tamanho_populacao, random_state=1)

# não endenti o code população 



print('Tamanho populacao {}, média {}, desvio {}' .format (len(populacao), np.mean(populacao), np.std(populacao)))

# a partir de .format não entendi
#Fazer o Gráfico de FDP Normal - População



fig, axs = plt.subplots (1,1, figsize=(14,6))

    #o que o 1,1 siginifica?

axs.hist(populacao, density=True, facecolor='g', alpha=0.75, bins=50)

    # alpha é a transparência, bins é o recorte angular. Como parametrizá-lo? Valores altos o histograma perde a forma normal perfeita

axs.grid (True)

axs.set_title ('Histograma de alturas da População')



plt.show()
# Seleconar amostra dentro do universo da populaçao parametrizada anteriormente - tamanho 100.000, média 160, desvio 20

tamanho_amostra=100

amostra=recupera_amostra(populacao, tamanho_amostra)

print(amostra)
#Fazer o Gráfico de FDP Amostra



fig, axs = plt.subplots (1,1, figsize=(14,6))

    #o que o 1,1 siginifica?

axs.hist(amostra, density=True, facecolor='g', alpha=0.75, bins=50)

    # alpha é a transparência, bins é o recorte angular. Como parametrizá-lo? Valores altos o histograma perde a forma normal perfeita

axs.grid (True)

axs.set_title ('Histograma de alturas da Amostra 1')



plt.show()
#Exercício - Dada a amostra de 100 elementos, calcule a média da altura da população considerando confianção de 93%



#1 Caluclar a média da amonstra



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

    #np é a biblioteca. sqrt é qual operação?



print('Resultado da questao 4 é igual a {}'.format(desvio_amostras))
#5 Calcular a margem de erro



margem_erro = numero_desvios*desvio_amostras



print('Resultado da questao 5 é igual a {}'.format(margem_erro))
#6 Calcular intervalo



inferior=media_amostra-margem_erro

superior=media_amostra+margem_erro



print('Resultado da questao 6 é igual a: A média da população para a amostra 1 de 100 elementos estará entre {} e {} com confiança {}'.format(inferior,superior,93))
#7 Obtenha uma amostra de 1000 elementos e calcule a média da altura da populacao considerando confiança de 80%





#7.1 Obter amostra de 1000 elementos 



tamanho_amostra_2=1000

amostra_2=recupera_amostra(populacao, tamanho_amostra_2)



#Fazer o Gráfico de FDP Amostra



fig, axs = plt.subplots (1,1, figsize=(14,6))

    #o que o 1,1 siginifica?

axs.hist(amostra_2, density=True, facecolor='g', alpha=0.75, bins=50)

    # alpha é a transparência, bins é o recorte angular. Como parametrizá-lo? Valores altos o histograma perde a forma normal perfeita

axs.grid (True)

axs.set_title ('Histograma de alturas da Amostra 2')



plt.show()
#7.2 Obter intervalo de 80% de confiança da amostra de 1.000 elementos



media_amostra_2 = np.mean(amostra_2)

desvio_amostra_2 = np.std(amostra_2)

numero_desvios_2 = calcula_numero_desvios_tstudent_para_confianca (0.80,tamanho_amostra_2)

desvio_amostras_2 = desvio_amostra_2/np.sqrt(tamanho_amostra_2)

margem_erro_2 = numero_desvios_2*desvio_amostras_2

inferior_2=media_amostra_2-margem_erro_2

superior_2=media_amostra_2+margem_erro_2

print('Resultado da questao 7 é igual a: A média da população para a amostra 2 de 1.000 elementos estará entre {} e {} com confiança de {}'.format(inferior_2,superior_2,80))
#8 Obtenha uma amostra de 100 elementos e calcule a média da altura da populacao considerando confiança de 99%



#8.1 Obter amostra de 100 elementos 



tamanho_amostra_3=100

amostra_3=recupera_amostra(populacao, tamanho_amostra_3)



#Fazer o Gráfico de FDP Amostra 3



fig, axs = plt.subplots (1,1, figsize=(14,6))

    #o que o 1,1 siginifica?

axs.hist(amostra_3, density=True, facecolor='g', alpha=0.75, bins=50)

    # alpha é a transparência, bins é o recorte angular. Como parametrizá-lo? Valores altos o histograma perde a forma normal perfeita

axs.grid (True)

axs.set_title ('Histograma de alturas da Amostra 3')



plt.show()
#8.2 Obter intervalo de 99% de confiança da amostra de 100 elementos



media_amostra_3 = np.mean(amostra_3)

desvio_amostra_3 = np.std(amostra_3)

numero_desvios_3 = calcula_numero_desvios_tstudent_para_confianca (0.99,tamanho_amostra_3)

desvio_amostras_3 = desvio_amostra_3/np.sqrt(tamanho_amostra_3)

margem_erro_3 = numero_desvios_3*desvio_amostras_3

inferior_3=media_amostra_3-margem_erro_3

superior_3=media_amostra_3+margem_erro_3

print('Resultado da questao 8 é igual a: A média da população para a amostra 3 de 100 elementos estará entre {} e {} com confiança de {}'.format(inferior_3,superior_3,99))