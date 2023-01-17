import numpy as np

import matplotlib.pyplot as plt

import math

import random

import pandas as pd

import scipy.stats as stat

from collections import Counter

from scipy.stats import t

from scipy.stats import norm

from sklearn.utils import shuffle



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
# Calcula número de desvios na distribuição t de student para determinada confianca





def calcula_Z_tstudent(confianca, tamanho_amostra):

    mean = 0

    std = 1

    rv = t(df=(tamanho_amostra-1))

    return rv.interval(confianca)[1]
# Calcula número de desvios na distribuição normal para determinada confianca





def calcula_Z_normal(confianca):

    mean = 0

    std = 1

    rv = norm(loc=mean, scale=std)

    return rv.interval(confianca)[1]
# Calcula número de desvios na distribuição normal para determinada confianca e múltiplas proporções



def calcula_Z_Multiplas_Proporcoes(confianca, numero_proporcoes):

    area = math.pow(  confianca, 1/(numero_proporcoes-1) )

    return calcula_Z_normal(area)
# Recupera amostra da população



def recupera_amostra(populacao, tamanho_amostra):

    tamanho_amostra = int(tamanho_amostra)

    if type(populacao) is pd.DataFrame:

        return populacao.sample(tamanho_amostra)

    else:

        #return populacao[np.random.randint(0, len(populacao), tamanho_amostra)]

        return [populacao[i] for i in np.random.randint(0, len(populacao), tamanho_amostra)]

# Função que calcula intervalo de confiança para média com base em uma amostra e a confiança desejada



def calcula_intervalo_media(amostra, confianca, tamanho_populacao):

    tamanho_amostra = len(amostra)

    

    #1-Calcule a média da amostra ex. media_amostra = np.mean(amostra)

    media_amostra = np.mean(amostra)



    #2-Calcule o desvio da amostra ex. desvio_amostra = np.std(amostra)

    desvio_amostra = np.std(amostra)



    #3-Calcule quantos desvios precisará para seu grau de confiânça ex. numero_desvios = calcula_numero_desvios_tstudent_para_confianca(confiança, tamanho_amostra)

    numero_desvios = calcula_Z_tstudent(confianca, tamanho_amostra)



    #4-Calcule o desvio das amostras ex. desvio_amostras = desvio_amostra/np.sqrt(tamanho_amostra)

    desvio_amostras = desvio_amostra/np.sqrt(tamanho_amostra)



    #5-Calcule a margem de erro ex. margem_erro = numero_desvios*desvio_amostras

    margem_erro = numero_desvios*desvio_amostras

    

    if tamanho_amostra>0.05*tamanho_populacao:

        margem_erro = margem_erro * (np.sqrt(tamanho_populacao-tamanho_amostra)/np.sqrt(tamanho_populacao-1))



    #6-Calcule o intervalo ex. inferior = media_amostra-margem_erro, superior = media_amostra+margem_erro

    inferior = media_amostra-margem_erro

    superior = media_amostra+margem_erro



    return inferior,superior
# Função que calcula intervalo de confiança para proporção de determinado valor com base em uma amostra e a confiança desejada

# O parâmetro valor representa a categoria na amostra para a qual se deseja o intervalo de confiança da proporção, por exemplo o nome de um candidato



def calcula_intervalo_proporcao(amostra, confianca, valor, tamanho_populacao):

    tamanho_amostra = len(amostra)

    

    proporcao_valor = Counter(amostra)[valor]/tamanho_amostra

    

    numero_desvios = calcula_Z_tstudent(confianca, tamanho_amostra)



    margem_erro = numero_desvios*np.sqrt(proporcao_valor*(1-proporcao_valor))/np.sqrt(tamanho_amostra)

    

    if tamanho_amostra>0.05*tamanho_populacao:

        margem_erro = margem_erro * (np.sqrt(tamanho_populacao-tamanho_amostra)/np.sqrt(tamanho_populacao-1))



    inferior = proporcao_valor-margem_erro

    superior = proporcao_valor+margem_erro



    return inferior,superior
# Função para cálculo do número de amostras para médias e somas



def Tamanho_Amostra_Valor_Pontual(Tamanho_Populacao, Grau_Confianca, Variancia, Margem_Erro):

    Numero_Desvios = calcula_Z_normal(Grau_Confianca)

    return math.ceil( 

            (Tamanho_Populacao * math.pow(Numero_Desvios,2) * Variancia) / \

            ( (Tamanho_Populacao-1)*math.pow(Margem_Erro,2) + math.pow(Numero_Desvios,2)*Variancia )

            )
# Função para cálculo do número de amostras para proporções simples



def Tamanho_Amostra_Proporcao_Simples(Tamanho_Populacao, Grau_Confianca, Proporcao, Margem_Erro):

    Numero_Desvios = calcula_Z_normal(Grau_Confianca)

    return math.ceil( 

            (Tamanho_Populacao * math.pow(Numero_Desvios,2) * Proporcao * (1-Proporcao)) / \

            ( (Tamanho_Populacao-1)*math.pow(Margem_Erro,2) + math.pow(Numero_Desvios,2)*Proporcao*(1-Proporcao) )

            )
# Função para cálculo do tamanho da amostra para proporções múltiplas



def Tamanho_Amostra_Proporcao_Multipla(Tamanho_Populacao, Confianca, Numero_Proporcoes, Margem_Erro):

    Confianca_ = calcula_Z_Multiplas_Proporcoes(Confianca, Numero_Proporcoes)

    return math.ceil( 

            (Tamanho_Populacao * math.pow(Confianca_,2) * 0.25 ) / \

            ( (Tamanho_Populacao-1)*math.pow(Margem_Erro,2) + math.pow(Confianca_,2)*0.25 )

            )
def grafico(planilha, coluna):

    planilha[coluna].value_counts().plot(kind='bar')

    plt.xlabel(coluna)

    plt.ylabel('Quantidade')



    plt.title(coluna)

    plt.grid(True)

    plt.show()

    

    print(planilha[coluna].value_counts()/len(planilha))
def histograma(planilha, coluna):

    n, bins, patches = plt.hist(planilha[coluna],  facecolor='g', alpha=0.75, bins=50)

    plt.xlabel(coluna)

    plt.ylabel('Quantidade')



    plt.title(coluna)

    plt.grid(True)

    plt.show()

    print('Média {}, desvio padrão {}'.format(planilha[coluna].mean(),planilha[coluna].std()))

    
def aplicaPolitica(PopulacaoControle,PopulacaoTeste ):

    PopulacaoControlePosPolitica = PopulacaoControle.copy()

    PopulacaoControlePosPoliticaMediaSalario = PopulacaoControlePosPolitica.Salario.mean()

    PopulacaoControlePosPoliticaMediaSalario = PopulacaoControlePosPoliticaMediaSalario * 1.1

    PopulacaoControlePosPolitica.Salario = PopulacaoControlePosPolitica.Salario * norm(loc=PopulacaoControlePosPoliticaMediaSalario, scale=0.1*PopulacaoControlePosPoliticaMediaSalario).rvs(size=TamanhoPopulacaoControle)



    PopulacaoTestePosPolitica = PopulacaoTeste.copy()

    PopulacaoTestePosPoliticaMediaSalario = PopulacaoTestePosPolitica.Salario.mean()

    PopulacaoTestePosPoliticaMediaSalario = PopulacaoTestePosPoliticaMediaSalario * 1.3

    PopulacaoTestePosPolitica.Salario = PopulacaoTestePosPolitica.Salario * norm(loc=PopulacaoTestePosPoliticaMediaSalario, scale=0.1*PopulacaoTestePosPoliticaMediaSalario).rvs(size=TamanhoPopulacaoTeste)



    return PopulacaoControlePosPolitica, PopulacaoTestePosPolitica
populacao = pd.read_excel('../input/votacaosimuladaeleicaopresidencialbrasil2018/populacao.xlsx', index_col=0) 
populacao.head()
len(populacao)
histograma(populacao,'Salario')
histograma(populacao,'Idade')
grafico(populacao,'Regiao')
grafico(populacao,'Voto')
grafico(populacao,'Sexo')
Margem_Erro_Salario = 100

Margem_Erro_Proporcao_Sexo = 0.05

Margem_Erro_Proporcao_Votos = 0.05

    

Grau_Confianca = 0.95
Tamanho_Amostra = 30

amostra = recupera_amostra(populacao, Tamanho_Amostra)



Variancia_Salario_Estimada = amostra.Salario.var()

print(Variancia_Salario_Estimada)



Proporcao_Sexo_M = sum (amostra.Sexo=='M')/Tamanho_Amostra

print(Proporcao_Sexo_M)
amostra.head()
Tamanho_Populacao = len(populacao)
#Tamanho_Amostra_Salario



Tamanho_Amostra_Salario = Tamanho_Amostra_Valor_Pontual(Tamanho_Populacao, Grau_Confianca, Variancia_Salario_Estimada, Margem_Erro_Salario)

print (Tamanho_Amostra_Salario)
#Tamanho_Amostra_Sexo



Tamanho_Amostra_Sexo = Tamanho_Amostra_Proporcao_Simples(Tamanho_Populacao, Grau_Confianca, Proporcao_Sexo_M, Margem_Erro_Proporcao_Sexo)



print (Tamanho_Amostra_Sexo)
#Tamanho_Amostra_Votacao  



Tamanho_Amostra_Votacao = Tamanho_Amostra_Proporcao_Multipla(Tamanho_Populacao, Grau_Confianca, 4, Margem_Erro_Proporcao_Votos)

print (Tamanho_Amostra_Votacao)
amostra = pd.read_excel('../input/amostra/amostra.xlsx', index_col=0) 

amostra.head()
inferior,superior = calcula_intervalo_media(amostra.Salario, Grau_Confianca, Tamanho_Populacao)

print('Intervalo da média {} {}'.format(inferior,superior))

print('Margem de erro {}'.format((superior-inferior)/2))
inferior,superior = calcula_intervalo_proporcao(amostra.Sexo, Grau_Confianca, 'M', Tamanho_Populacao)

print('Intervalo de confiança M {} {}'.format(inferior,superior))

print('Margem de erro {}'.format((superior-inferior)/2))
inferior,superior = calcula_intervalo_proporcao(amostra.Voto, Grau_Confianca, 'B', Tamanho_Populacao)

print('Intervalo de confiança B {} {}'.format(inferior,superior))

print('Margem de erro {}'.format((superior-inferior)/2))
inferior,superior = calcula_intervalo_media(amostra.Idade, Grau_Confianca, Tamanho_Populacao)

print('Intervalo da média {} {}'.format(inferior,superior))

print('Margem de erro {}'.format((superior-inferior)/2))
inferior,superior = calcula_intervalo_proporcao(amostra.Regiao, Grau_Confianca, 'Sudeste', Tamanho_Populacao)

print('Intervalo de confiança B {} {}'.format(inferior,superior))

print('Margem de erro {}'.format((superior-inferior)/2))