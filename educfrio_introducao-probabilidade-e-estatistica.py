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
# Para qualquer valor de F podemos determinar precisamente qual será a aceleração do corpo



m = 1

F = np.arange(0.0,10.0,1)

a = F/m



fig, ax = plt.subplots(figsize=(10,6))

plt.plot(F,a,'*')

plt.plot(F,a)



plt.xlabel('Força')

plt.ylabel('Aceleração')

plt.title('Força X Aceleração')

plt.grid(True)



plt.show()
# No lançamento da moeda não podemos prever qualquer resultado específico, 

# mas podemos determinar a probabilidade de cada resultado



# Inicialização de variáveis

escolhas = ['Cara','Coroa']

lancamentos = 100

# Executa 100 lançamentos da moeda com probabilidade 6/10 de Cara e 4/10 de Coroa

resultados = random.choices(population=escolhas, weights=[6/10,4/10], k=lancamentos)



fig, ax = plt.subplots(figsize=(10,6))

ind = range(1,len(escolhas)+1)

# Calcula o percentual de Caras e Coroas

proporcoes = [resultados.count('Cara')/lancamentos*100,resultados.count('Coroa')/lancamentos*100]

plt.bar(ind,proporcoes,align='center')

ax.set_xticks(ind)

ax.set_xticklabels(escolhas)





plt.xlabel('Resultado')

plt.ylabel('Proporção')

plt.title('Resultados de lançamentos de moeda não equilibrada')

plt.grid(True)

plt.show()
# exemplo: 3 variáveis de entrada com relacionamento não linear entre elas



# Cria três variáveis com valores aleatórios uniformemente distribuídos no intervalo [0,0 1,0)

x = np.random.random_sample(size=1000)

y = np.random.random_sample(size=1000)

z = np.random.random_sample(size=1000)



# Cria uma variável formada pela combinação não linear das três anteriores

w = x**2-y**2+z**3

# Escalona w para que fique no intervalo [-1 1]

w = w/np.max(np.abs(w))





plt.subplots(figsize=(14,6))

plt.plot(x,w,'.')

plt.xlabel('x')

plt.ylabel('w')

plt.title('w = x**2-y**2+z**3')

plt.grid(True)

plt.show()



plt.subplots(figsize=(14,6))

plt.xlabel('w')

plt.ylabel('Probabilidade')

plt.title('Distribuição de probabilidade de W')

plt.grid(True)

n, bins, patches = plt.hist(w, density=True, facecolor='g', alpha=0.75, bins=50)

plt.show()





print('Matriz de correlações entre as variáveis')

print(np.corrcoef([x,y,z,w]))



print("\nAnálise estatística/probabilística")

print("Média de w {}, desvio padrão de w {}".format(np.mean(w),np.std(w)))
from sklearn.neural_network import MLPRegressor



# Cria uma matriz de 3 colunas com as variáveis de entrada x, y e z

x_ = np.concatenate((np.reshape(x,(-1,1)),np.reshape(y,(-1,1)),np.reshape(z,(-1,1))), axis=1)

# Cria uma matriz de 1 coluna com a variável w

y_ = np.reshape(w,(-1,1))



# Treina o modelo com 900 valores e testa com 100 valores

x_train = x_[0:900,:]

y_train = y_[0:900,:]

x_test = x_[900:1000,:]

y_test = y_[900:1000,:]



estimator = MLPRegressor(

                              learning_rate = 'adaptive',

                              random_state = random_state,

                              verbose=True,

                                max_iter = 200,

                            hidden_layer_sizes = [100,50,40,30,20,10],   

                    solver = 'adam',

                    alpha = 0.0001,

                    activation = 'relu'

                            )



estimator.fit(x_train,y_train)

pd.DataFrame(estimator.loss_curve_).plot(figsize=(14,6))
# Testa o modelo com os 100 exemplos reservados para teste (não usados no treinamento), 

# de forma a verificar a capacidade de generalização



plt.subplots(figsize=(14,6))

plt.plot(y_test,'r.')

plt.plot(estimator.predict(x_test),'b*')

plt.xlabel('Amostra de teste')

plt.ylabel('w')

plt.title('w = x**2-y**2+z**3  -  Vermelho = valor real, Azul = previsão da rede neural')

plt.grid(True)

plt.show()
# Cálculo analítico baseado em contagem 

# Númerador = número possível de 15 caras em 30 lançamentos, denominador = total de resultados possíveis em 30 lançamentos da moeda

combinacao(30,15)/possibilidades(30,2)
# Cálculo analítico baseado na probabilidade básica

probabilidades = np.zeros((31,1))

for i in range(0,31,1):

    probabilidades[i]=combinacao(30,i)*((1/2)**(i))*((1/2)**(30-i))



plt.bar(range(0,31,1),probabilidades[:,0], facecolor='black', alpha=0.75)



plt.xlabel('# Caras')

plt.ylabel('Probabilidade')

plt.title('Histogram Moeda')

plt.grid(True)

plt.show()
print('Soma das probabilidades {}'.format(sum(probabilidades)))
# Cálculo por simulação - usando probabilidade básica

Cara = 1

Coroa = 0

Moeda = [Cara,Coroa]

Equilibrio = [1/2,1/2]

lancamentos = 30

repeticoes = 100

np.random.seed(1)

resultado = np.random.choice(a=Moeda, p=Equilibrio, replace=True, size=(repeticoes,lancamentos))

resultado=np.sum(resultado, axis=1)

probabilidades,_ = np.histogram(a=resultado, density=True, bins=range(0,31,1))



#n, bins, patches = plt.hist(resultado, density=True, facecolor='g', alpha=0.75, bins=range(0,31,1))

plt.bar(range(0,30,1),probabilidades, facecolor='black', alpha=0.75)



plt.xlabel('# Caras')

plt.ylabel('Probabilidade')

plt.title('Histogram Moeda')

plt.grid(True)

plt.show()



print (np.sum(probabilidades))
# Cálculo por simulação - usando probabilidade básica

Cara = 1

Coroa = 0

Moeda = [Cara,Coroa]

Equilibrio = [1/2,1/2]

lancamentos = 30

repeticoes = 1000

np.random.seed(1)

resultado = np.random.choice(a=Moeda, p=Equilibrio, replace=True, size=(repeticoes,lancamentos))

resultado=np.sum(resultado, axis=1)

probabilidades,_ = np.histogram(a=resultado, density=True, bins=range(0,31,1))



#n, bins, patches = plt.hist(resultado, density=True, facecolor='g', alpha=0.75, bins=range(0,31,1))

plt.bar(range(0,30,1),probabilidades, facecolor='g', alpha=0.75)



plt.xlabel('# Caras')

plt.ylabel('Probability')

plt.title('Histogram Moeda')

plt.grid(True)

plt.show()
# Cálculo por simulação - usando probabilidade básica

Cara = 1

Coroa = 0

Moeda = [Cara,Coroa]

Equilibrio = [1/2,1/2]

lancamentos = 30

repeticoes = 10000

np.random.seed(1)

resultado = np.random.choice(a=Moeda, p=Equilibrio, replace=True, size=(repeticoes,lancamentos))

resultado=np.sum(resultado, axis=1)

probabilidades,_ = np.histogram(a=resultado, density=True, bins=range(0,31,1))



#n, bins, patches = plt.hist(resultado, density=True, facecolor='g', alpha=0.75, bins=range(0,31,1))

plt.bar(range(0,30,1),probabilidades, facecolor='black', alpha=0.75)



plt.xlabel('# Caras')

plt.ylabel('Probabilidade')

plt.title('Histogram Moeda')

plt.grid(True)

plt.show()
# Cálculo por simulação - usando probabilidade básica

Cara = 1

Coroa = 0

Moeda = [Cara,Coroa]

Equilibrio = [1/2,1/2]

lancamentos = 30

repeticoes = 100000

np.random.seed(1)

resultado = np.random.choice(a=Moeda, p=Equilibrio, replace=True, size=(repeticoes,lancamentos))

resultado=np.sum(resultado, axis=1)

probabilidades,_ = np.histogram(a=resultado, density=True, bins=range(0,lancamentos+1,1))



#n, bins, patches = plt.hist(resultado, density=True, facecolor='g', alpha=0.75, bins=range(0,31,1))

plt.bar(range(0,lancamentos,1),probabilidades, facecolor='g', alpha=0.75)



plt.xlabel('# Caras')

plt.ylabel('Probabilidade')

plt.title('Histogram Moeda')

plt.grid(True)

plt.show()
probabilidades
Dado = [1,2,3,4,5,6]

Equilibrio = [1/6,1/6,1/6,1/6,1/6,1/6]

lancamentos = 2

repeticoes = 100000

np.random.seed(1)

#Simula 100000 lançamentos de dois dados e soma, em cada um dos 100000 lançamentos, os resultados obtidos pelos 2 dados

resultado = np.random.choice(a=Dado, p=Equilibrio, replace=True, size=(repeticoes,lancamentos))

resultado=np.sum(resultado, axis=1)

probabilidades,_ = np.histogram(a=resultado, density=True, bins=range(1,14,1))



#n, bins, patches = plt.hist(resultado, density=True, facecolor='g', alpha=0.75, bins=range(0,31,1))

plt.bar(range(1,13,1),probabilidades, facecolor='g', alpha=0.75)



plt.xlabel('Soma dois dados')

plt.ylabel('Probabilidade')

plt.title('Histogram Dado')

plt.grid(True)

plt.show()
# Usando scipy



from scipy.stats import binom

tentativas = 30

rv = binom(tentativas, 1/2)



# calcula probabilide de obter 0,1,2,...30 caras em 30 lançamentos

resultado = rv.pmf(range(0,31,1))



plt.bar(range(0,31,1),resultado)



plt.xlabel('# Cara')

plt.ylabel('Probability')

plt.title('Histogram Moeda')

plt.grid(True)

plt.show()



print('Valor da PMF em 15: {}, correspondente à probabilidade de 15 caras em 30 lançamentos'.format(rv.pmf(15)))

print('Valor da PMF em 0: {}, correspondente à probabilidade de 0 caras em 30 lançamentos'.format(rv.pmf(0)))

print('Soma das probabilidades {}'.format(sum(resultado)))
# Usando scipy



from scipy.stats import binom

tentativas = 30

rv = binom(tentativas, 1/2)



# calcula probabilide de obter 0,1,2,...30 ou menos caras em 30 lançamentos

resultado = rv.cdf(range(0,31,1))



plt.bar(range(0,31,1),resultado)



plt.xlabel('# Caras')

plt.ylabel('Probabilidade')

plt.title('Histogram Moeda')

plt.grid(True)

plt.show()



print('Probabilidade de conseguirmos 15 ou menos caras {}'.format(rv.cdf(15)))

# Usando scipy



from scipy.stats import binom

tentativas = 30

rv = binom(tentativas, 1/2)



resultado = rv.cdf(range(0,31,1))



# Calcula o interval, em número de caras, para o qual intervalo temos 95% de certeza de acertar o resultado

intervalo = rv.interval(0.95)



print('Com 95% de chance teremos entre {} e {} caras em 30 lançamentos'.format(intervalo[0],intervalo[1]))

from scipy.stats import binom

tentativas = 30

rv = binom(tentativas, 1/2)



media = rv.mean()



print('Média {}'.format(media))



variancia = rv.var()



print('Variância {}'.format(variancia))



desvio_padrao = rv.std()



print('Desvio padrão {}'.format(desvio_padrao))



prob_media = rv.pmf(media)



print('Probabilidade da média {} é {}'.format(media,prob_media))



desv = (rv.cdf(media+desvio_padrao)-rv.cdf(media-desvio_padrao))



print('Probabilidade do resultado estar afastado até 1 desvio padrão da média é {}'.format(desv))
from scipy.stats import chi2

graus = 3

rv = chi2(graus)

x = np.linspace(0,15,1000)

# calcula probabilide de obter 0,1,2,...30 caras em 30 lançamentos

resultado = rv.pdf(x)

plt.subplots(figsize=(14,6))

plt.plot(x,resultado, color='black')



plt.xlabel('x')

plt.ylabel('Probabilidade')

plt.title('Distribuição Qui-Quadrado com 3 graus de liberdade')

plt.grid(True)

plt.show()
from scipy.stats import cauchy

tentativas = 30

rv = cauchy()

x = np.linspace(-10,10,1000)

# calcula probabilide de obter 0,1,2,...30 caras em 30 lançamentos

resultado = rv.pdf(x)

plt.subplots(figsize=(14,6))

plt.plot(x,resultado, color='black')



plt.xlabel('x')

plt.ylabel('Probabilidade')

plt.title('Distribuição Cauchy [0 1]')

plt.grid(True)

plt.show()
from scipy.stats import uniform

tentativas = 30

rv = uniform()

x = np.linspace(0,2,1000)

# calcula probabilide de obter 0,1,2,...30 caras em 30 lançamentos

resultado = rv.pdf(x)

plt.subplots(figsize=(14,6))

plt.plot(x,resultado, color='black')



plt.xlabel('x')

plt.ylabel('Probabilidade')

plt.title('Distribuição Uniforme [0 1]')

plt.grid(True)

plt.show()
from scipy.stats import binom

tentativas = 30

rv = binom(tentativas, 1/2)



# calcula probabilide de obter 0,1,2,...30 caras em 30 lançamentos

resultado = rv.pmf(range(0,31,1))

plt.subplots(figsize=(14,6))

plt.plot(range(0,31,1),resultado, color='black')



plt.xlabel('Número de caras')

plt.ylabel('Probabilidade')

plt.title('Distribuição Binomial')

plt.grid(True)

plt.show()
from scipy.stats import binom

tentativas = 30

rv = binom(tentativas, 1/2)



# calcula probabilide de obter 0,1,2,...30 caras em 30 lançamentos

resultado = rv.pmf(range(0,31,1))

plt.subplots(figsize=(14,6))

plt.bar(range(0,31,1),resultado, color='black')



plt.xlabel('x')

plt.ylabel('Probabilidade')

plt.title('Distribuição Binomial (30, 0,5)')

plt.grid(True)

plt.show()
# Poisson



from scipy.stats import bernoulli



rv = bernoulli(0,4)



# Poisson



from scipy.stats import gengamma

graus = 30

a, c = 4.42, -3.12

rv = gengamma(a, c)

variacao = np.linspace(0,2,1000)



# calcula probabilide de obter 0,1,2,...30 caras em 30 lançamentos

resultado = rv.pdf(variacao)

plt.subplots(figsize=(14,6))

plt.plot(variacao,resultado, color='black')



plt.xlabel('x')

plt.ylabel('Probabilidade')

plt.title('Distribuição Gama (alfa 4,4 - beta -3,1)')

plt.grid(True)

plt.show()
# t-Student



from scipy.stats import t

graus = 30

rv = t(graus)

variacao = np.linspace(-10,10,1000)



# calcula probabilide de obter 0,1,2,...30 caras em 30 lançamentos

resultado = rv.pdf(variacao)

plt.subplots(figsize=(14,6))

plt.plot(variacao,resultado, color='black')



plt.xlabel('x')

plt.ylabel('Probabilidade')

plt.title('Distribuição t-Student (Graus de liberdade 30)')

plt.grid(True)

plt.show()
# Poisson



from scipy.stats import poisson

taxa = 10

rv = poisson(taxa)

variacao = range(0,50,1)



# calcula probabilide de obter 0,1,2,...30 caras em 30 lançamentos

resultado = rv.pmf(variacao)

plt.subplots(figsize=(14,6))

plt.bar(variacao,resultado, color='black')



plt.xlabel('x')

plt.ylabel('Probabilidade')

plt.title('Distribuição Poisson (taxa 10)')

plt.grid(True)

plt.show()
# Exponencial



from scipy.stats import expon

lambda_ = 1 # taxa

rv = expon( scale=lambda_)





variacao = np.linspace(0,5,1000)



# calcula probabilide de obter 0,1,2,...30 caras em 30 lançamentos

resultado = rv.pdf(variacao)

plt.subplots(figsize=(14,6))

plt.plot(variacao,resultado, color='black')



plt.xlabel('x')

plt.ylabel('Probabilidade')

plt.title('Distribuição Exponencial(lambda 1)')

plt.grid(True)

plt.show()
# Poisson



from scipy.stats import norm

mean = 0

std = 1

rv = norm(loc=mean, scale=std)





variacao = np.linspace(-5,5,1000)



# calcula probabilide de obter 0,1,2,...30 caras em 30 lançamentos

resultado = rv.pdf(variacao)

plt.subplots(figsize=(14,6))

plt.plot(variacao,resultado, color='black')



plt.xlabel('x')

plt.ylabel('Probabilidade')

plt.title('Distribuição Normal(0,1)')

plt.grid(True)

plt.show()
# Normal

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html#scipy.stats.norm





from scipy.stats import norm

from scipy.stats import kstest



mean = 0

std = 1

rv = norm(loc=mean, scale=std)

np.random.seed(1)

resultado = rv.rvs(size=1000)



n, bins, patches = plt.hist(resultado, density=True, facecolor='black', alpha=0.75, bins=50)

intervalo = np.linspace(mean-5*std,mean+5*std, num=50)



plt.plot(intervalo, rv.pdf(intervalo), 'k-', label='pdf')

plt.xlabel('Valores')

plt.ylabel('Probabilidade')



plt.title('Distribuição normal')

plt.grid(True)

plt.show()
print('Probabilidade do valor 0: {}'.format(rv.pdf(0)))

print('Probabilidade de valor menor ou igual a 0: {}'.format(rv.cdf(0)))

print('Média: {}'.format(rv.mean()))

print('Mediana: {}'.format(rv.median()))

print('Variância: {}'.format(rv.var()))

print('Desvio padrão: {}'.format(rv.std()))

fig1, ax1 = plt.subplots()

ax1.set_title('Box Plot')

ax1.boxplot(resultado)


tamanho_amostra = (len(resultado))



tamanho_amostra_entre_1_desvios = sum( (resultado>(mean-1*std)) & (resultado<(mean+1*std)) )

tamanho_amostra_entre_2_desvios = sum( (resultado>(mean-2*std)) & (resultado<(mean+2*std)) )

tamanho_amostra_entre_3_desvios = sum( (resultado>(mean-3*std)) & (resultado<(mean+3*std)) )



print('Percentual dos dados entre {} desvios: {}'.format(1,tamanho_amostra_entre_1_desvios*100/tamanho_amostra))

print('Percentual dos dados entre {} desvios: {}'.format(2,tamanho_amostra_entre_2_desvios*100/tamanho_amostra))

print('Percentual dos dados entre {} desvios: {}'.format(3,tamanho_amostra_entre_3_desvios*100/tamanho_amostra))
# Determinação de parâmetros baseado nos dados



media, desvio = norm.fit(resultado)
#https://plot.ly/python/normality-test/



# Teste de normalidade

    

kstest(resultado, 'norm')



s = 0.3

repeticoes = 100000



rv = stat.lognorm(s=s)



populacao = rv.rvs(size=repeticoes, random_state=random_state)



print('Média: {}'.format(rv.mean()))

print('Mediana: {}'.format(rv.median()))

print('Variância: {}'.format(rv.var()))

print('Desvio padrão: {}'.format(rv.std()))
n, bins, patches = plt.hist(populacao, density=True, facecolor='black', alpha=0.75, bins=50)





plt.title('Distribuição logonormal')

plt.xlabel('Valores')

plt.ylabel('Probabilidade')

plt.grid(True)

plt.show()
fig1, ax1 = plt.subplots()

ax1.set_title('Box Plot')

ax1.boxplot(populacao)
n, bins, patches = plt.hist(np.log(populacao), density=True, facecolor='g', alpha=0.75, bins=50)





plt.title('O log da var aleatória logonormal tem distribuição normal')

plt.xlabel('Valores')

plt.ylabel('Probabilidade')

plt.grid(True)

plt.show()
plt.plot(populacao,'.')



plt.xlabel('Amostra')

plt.ylabel('Valor')

plt.title('Distribuição logonormal')

plt.grid(True)

plt.show()
# Geração da população, esta parte é desconhecida para o estatístico



lancamentos = 30

repeticoes = 100000

np.random.seed(1)

populacao = np.random.binomial(30, 1/4, size=repeticoes)



# Estatística descritiva da população



probabilidades,_ = np.histogram(a=populacao, density=True, bins=range(0,31,1))



#n, bins, patches = plt.hist(resultado, density=True, facecolor='g', alpha=0.75, bins=range(0,31,1))

plt.bar(range(0,30,1),probabilidades, facecolor='g', alpha=0.75)



plt.xlabel('# Caras')

plt.ylabel('Probabilidade')

plt.title('Histogram Moeda')

plt.grid(True)

plt.show()



print('Média: {}'.format(np.mean(populacao)))

print('Probabilidade Cara: {}'.format(np.mean(populacao)/lancamentos))
# Amostra de 1% da população

amostra = populacao[np.random.randint(0, len(populacao),int(0.01*repeticoes))]
# Estatística descritiva da amostra



probabilidades,_ = np.histogram(a=amostra, density=True, bins=range(0,31,1))



#n, bins, patches = plt.hist(resultado, density=True, facecolor='g', alpha=0.75, bins=range(0,31,1))

plt.bar(range(0,30,1),probabilidades, facecolor='g', alpha=0.75)



plt.xlabel('# Caras')

plt.ylabel('Probability')

plt.title('Histogram Moeda')

plt.grid(True)

plt.show()



print('Média: {}'.format(np.mean(amostra)))

print('Probabilidade Cara: {}'.format(np.mean(amostra)/lancamentos))


repeticoes = 100000

mean = 5

np.random.seed(1)



# População lognormal

s = 0.8

rv_lognorm = stat.lognorm(s=s,loc=mean-1.3)

populacao_lognorm = rv_lognorm.rvs(size=repeticoes, random_state=random_state)



# População normal

std = 1.3

rv_norm = stat.norm(loc=mean, scale=std)

populacao_norm = rv_norm.rvs(size=repeticoes, random_state=random_state)











fig, axs = plt.subplots(1, 2, figsize=(14,6))



axs[0].plot(populacao_lognorm,'.')

axs[0].grid(True)

axs[0].set_title('populacao_lognorm')



axs[1].plot(populacao_norm,'.')

axs[1].grid(True)

axs[1].set_title('populacao_norm')



plt.show()
print('\nPopulação lognormal: \nmédia {}, \ndesvio padrão {}, \nmoda {}, \nmediana {}, \nCurtose {}, \nSimetria {}'.format(

    np.mean(populacao_lognorm), 

    np.std(populacao_lognorm),

    stat.mode(populacao_lognorm),

    np.median(populacao_lognorm),

    stat.kurtosis(populacao_lognorm),

    stat.skew(populacao_lognorm)

))
print('\nPopulação normal: \nmédia {}, \ndesvio padrão {}, \nmoda {}, \nmediana {}, \nCurtose {}, \nSimetria {}'.format(

    np.mean(populacao_norm), 

    np.std(populacao_norm),

    stat.mode(populacao_norm),

    np.median(populacao_norm),

    stat.kurtosis(populacao_norm),

    stat.skew(populacao_norm)

))
fig, axs = plt.subplots(1, 2, figsize=(14,6))



axs[0].hist(populacao_lognorm, density=True, facecolor='black', alpha=0.75, bins=50)

axs[0].grid(True)

axs[0].set_title('populacao_lognorm')

axs[0].set_xlabel('Idade')

axs[0].set_ylabel('Proporção')



axs[1].hist(populacao_norm, density=True, facecolor='black', alpha=0.75, bins=50)

axs[1].grid(True)

axs[1].set_title('populacao_norm')

axs[1].set_xlabel('Idade')

axs[1].set_ylabel('Proporção')



plt.show()
dados = np.concatenate((populacao_lognorm, populacao_norm), 0)

dados = np.reshape(dados,(2,repeticoes))

dados = dados.T

fig, axs = plt.subplots(1, 1, figsize=(10,6))

_ = plt.boxplot(dados,vert =False, labels =['lognorm','norm'], meanline =True)

plt.title('Boxplot')
from statsmodels.graphics.gofplots import qqplot



fig, axs = plt.subplots(1, 2, figsize=(14,6))



qqplot(populacao_lognorm, line='s', ax=axs[0])

axs[0].set_title('populacao_lognorm')



qqplot(populacao_norm, line='s', ax=axs[1])

axs[1].set_title('populacao_norm')



plt.show()



repeticoes = 100000

mean = 5



# População lognormal

s = 0.8

rv_lognorm = stat.lognorm(s=s,loc=mean-1.3)

populacao_lognorm = rv_lognorm.rvs(size=repeticoes, random_state=random_state)



amostra_1000 = populacao_lognorm[np.random.randint(0, len(populacao_lognorm),1000)]

amostra_100 = populacao_lognorm[np.random.randint(0, len(populacao_lognorm),100)]

amostra_10 = populacao_lognorm[np.random.randint(0, len(populacao_lognorm),10)]



print('Desvios real {}\n'.format(rv_lognorm.std()))



print('\nDesvios amostra 1000 sem correção {}'.format(np.std(amostra_1000)))

print('Desvios amostra 1000 com correção {}'.format(np.std(amostra_1000, ddof =1)))



print('\nDesvios amostra 100 sem correção {}'.format(np.std(amostra_100)))

print('Desvios amostra 100 com correção {}'.format(np.std(amostra_100, ddof =1)))



print('\nDesvios amostra 10 sem correção {}'.format(np.std(amostra_10)))

print('Desvios amostra 10 com correção {}'.format(np.std(amostra_10, ddof =1)))
# Vamos criar uma população distribuída de forma lognormal





repeticoes = 100000

mean = 50



# População lognormal

s = 0.9

np.random.seed(1)

rv_lognorm = stat.lognorm(s=s,loc=mean)

populacao_lognorm = rv_lognorm.rvs(size=repeticoes, random_state=random_state)



escolhas = [0,1]

sexo_doadores = random.choices(population=escolhas, weights=[6/10,4/10], k=repeticoes)

sexo_doadores = np.asarray(sexo_doadores, dtype=np.int)











print('Mínimo {}'.format(np.min(populacao_lognorm)))

print('Máximo {}'.format(np.max(populacao_lognorm)))

print('Média {}'.format(np.mean(populacao_lognorm)))

print('Desvio {}'.format(np.std(populacao_lognorm)))

print('Var {}'.format(np.var(populacao_lognorm)))



fig, axs = plt.subplots(1, 1, figsize=(14,6))



axs.hist(populacao_lognorm, density=False, facecolor='black', alpha=0.75, bins=100)

axs.set_xlabel('Valor doação')

axs.set_ylabel('Quantidade')

axs.grid(True)

axs.set_title('Doações')





plt.show()


p = np.sum(sexo_doadores)/repeticoes

p
p*(1-p)
np.var(sexo_doadores)
tamanho_amostra = 1000

selecionados = np.random.randint(0, len(populacao_lognorm),tamanho_amostra)



amostra_sexo_doadores = sexo_doadores[selecionados]

np.sum(amostra_sexo_doadores)/tamanho_amostra


amostra = populacao_lognorm[selecionados]

print('Mínimo {}'.format(np.min(amostra)))

print('Máximo {}'.format(np.max(amostra)))

print('Média {}'.format(np.mean(amostra)))

print('Desvio {}'.format(np.std(amostra)))

print('Desvio {}'.format(np.std(amostra, ddof =1)))

print('Var {}'.format(np.var(amostra, ddof =1)))



print('Raiz tamanho {}'.format(np.sqrt(tamanho_amostra)))

print('Desvio / Raiz tamanho {}'.format(np.std(amostra, ddof =1)/np.sqrt(tamanho_amostra)))
# Vamos extrair 1000 amostras e calcular suas médias

amostras = 1000

tamanho_amostra = 300

np.random.seed(1)

medias = np.zeros((amostras,1))

variancias= np.zeros((amostras,1))

qtd_sexo_masculino= np.zeros((amostras,1))

percentuais = np.zeros((amostras,1))

for i in range(0,amostras,1):

    medias[i]=np.mean(populacao_lognorm[np.random.randint(0, len(populacao_lognorm),tamanho_amostra)])

    percentuais[i] = np.sum(sexo_doadores[np.random.randint(0, len(populacao_lognorm),tamanho_amostra)])/tamanho_amostra

    qtd_sexo_masculino[i] = np.sum(sexo_doadores[np.random.randint(0, len(populacao_lognorm),tamanho_amostra)])

    variancias[i]=np.var(populacao_lognorm[np.random.randint(0, len(populacao_lognorm),tamanho_amostra)])

# A distribuição das médias aproxima-se de uma Normal, independente da distribuição original que gerou as amostras

from scipy.stats import norm

n, bins, patches = plt.hist(medias, density=True, facecolor='black', alpha=0.75, bins=50)



mean_ = np.mean(medias)

std_ = np.std(medias)

print('Média das médias das amostras {}'.format(mean_))

print('Desvio das médias das amostras {}'.format(std_))



rv = norm(loc=mean_, scale=std_)



intervalo = np.linspace(mean_-3*std_,mean_+3*std_, num=50)

plt.plot(intervalo, rv.pdf(intervalo), 'k-', label='pdf')



plt.xlabel('Média da amostra')

plt.ylabel('Probabilidade')

plt.title('Histogram de médias')

plt.grid(True)

plt.show()
def calcula_Z_normal(confianca):

    mean = 0

    std = 1

    rv = norm(loc=mean, scale=std)

    return rv.interval(confianca)[1]



def calcula_Z_tstudent(confianca, tamanho_amostra):

    mean = 0

    std = 1

    rv = t(df=(tamanho_amostra-1))

    return rv.interval(confianca)[1]



calcula_Z_normal(0.99)
from scipy.stats import norm

n, bins, patches = plt.hist(variancias, density=True, facecolor='black', alpha=0.75, bins=50)



mean_ = np.mean(variancias)

std_ = np.std(variancias)

print('Média das variancias {}'.format(mean_))

print('Desvio das variancias {}'.format(std_))



rv = norm(loc=mean_, scale=std_)



intervalo = np.linspace(mean_-3*std_,mean_+3*std_, num=50)

plt.plot(intervalo, rv.pdf(intervalo), 'k-', label='pdf')



plt.xlabel('Variancias')

plt.ylabel('Probabilidade')

plt.title('Histogram de médias')

plt.grid(True)

plt.show()
# A distribuição das médias aproxima-se de uma Normal, independente da distribuição original que gerou as amostras

from scipy.stats import norm

n, bins, patches = plt.hist(percentuais, density=True, facecolor='black', alpha=0.75, bins=50)



mean_ = np.mean(percentuais)

std_ = np.std(percentuais)

print('Média das proporções das amostras {}'.format(mean_))

print('Desvio das proporções das amostras {}'.format(std_))



rv = norm(loc=mean_, scale=std_)



intervalo = np.linspace(mean_-3*std_,mean_+3*std_, num=50)

plt.plot(intervalo, rv.pdf(intervalo), 'k-', label='pdf')



plt.xlabel('Proporções nas amostras')

plt.ylabel('Probabilidade')

plt.title('Histogram de proporções')

plt.grid(True)

plt.show()
from scipy.stats import norm

n, bins, patches = plt.hist(qtd_sexo_masculino, density=True, facecolor='black', alpha=0.75, bins=50)



mean_ = np.mean(qtd_sexo_masculino)

std_ = np.std(qtd_sexo_masculino)

print('Média das qtd_sexo_masculino {}'.format(mean_))

print('Desvio das qtd_sexo_masculino {}'.format(std_))



rv = norm(loc=mean_, scale=std_)



intervalo = np.linspace(mean_-3*std_,mean_+3*std_, num=50)

plt.plot(intervalo, rv.pdf(intervalo), 'k-', label='pdf')



plt.xlabel('qtd_sexo_masculino')

plt.ylabel('Probabilidade')

plt.title('Histogram de proporções')

plt.grid(True)

plt.show()
mean_ = np.mean(medias)

std_ = np.std(medias)

print('Média das médias das amostras {}'.format(mean_))

print('Desvio das médias das amostras {}'.format(std_))

print('Dois desvios {}'.format(2*std_))



print('Intervalo de 95,45% de confiança {} - {}'.format(mean_-2*std_,mean_+2*std_))
np.random.seed(1)

for i in range(0,10,1):

    print('Média da amostra {}: {}'.format(i,

                                np.mean(populacao_lognorm[np.random.randint(0, len(populacao_lognorm),tamanho_amostra)])))
np.random.seed(1)

tamanho_amostra = 1000

std_ = np.std(populacao_lognorm)/np.sqrt(tamanho_amostra)

print('Tamanho amostra {}, margem de erro considerando dois desvios {}, média real da população {}'.format(tamanho_amostra,2*std_, np.mean(populacao_lognorm)))

# Vamos capturar 10 amostras da população

for i in range(0,10,1):

    # Para cada amostra i calculamos a média

    media_i = np.mean(populacao_lognorm[np.random.randint(0, len(populacao_lognorm),tamanho_amostra)])

    print('Amostra {}, média {}, com margem de erro de dois desvios, a média da população estará entre {} e {} com 95% de chance'.format(i,

                                                                    media_i,

                                                                   media_i-2*std_,

                                                                   media_i+2*std_))
np.random.seed(1)

tamanho_amostra = 100

std_ = np.std(populacao_lognorm)/np.sqrt(tamanho_amostra)

print('Tamanho amostra {}, margem de erro considerando dois desvios {}, média real da população {}'.format(tamanho_amostra,2*std_, np.mean(populacao_lognorm)))

# Vamos capturar 10 amostras da população

for i in range(0,10,1):

    # Para cada amostra i calculamos a média

    media_i = np.mean(populacao_lognorm[np.random.randint(0, len(populacao_lognorm),tamanho_amostra)])

    print('Amostra {}, média {}, com margem de erro de dois desvios, a média da população estará entre {} e {} com 95% de chance'.format(i,

                                                                    media_i,

                                                                   media_i-2*std_,

                                                                   media_i+2*std_))
np.random.seed(1)

tamanho_amostra = 100

amostra_100_elementos = populacao_lognorm[np.random.randint(0, len(populacao_lognorm),tamanho_amostra)]

print('Média amostra (estima a média da população) {}'.format(np.mean(amostra_100_elementos)))

print('Desvio amostra (estima o desvio da população) {}'.format(np.std(amostra_100_elementos)))

print('Desvio estimado da média das amostras (desvio da população/raiz(tamanho amostra)) {}'.format(np.std(amostra_100_elementos)/np.sqrt(tamanho_amostra)))

print('Margem erro (considerando dois desvios para a confiança de 95%) {}'.format(2*np.std(amostra_100_elementos)/np.sqrt(tamanho_amostra)))





# o desvio padrão das médias das amostras é desvio da população / sqrt(samples)

from scipy.stats import norm





def calcula_numero_desvios_normal_para_confianca(confianca):

    mean = 0

    std = 1

    rv = norm(loc=mean, scale=std)

    return rv.interval(confianca)[1]
calcula_numero_desvios_normal_para_confianca(0.9545)
calcula_numero_desvios_normal_para_confianca(0.99)
from scipy.stats import t





def calcula_numero_desvios_tstudent_para_confianca(confianca, tamanho_amostra):

    mean = 0

    std = 1

    rv = t(df=(tamanho_amostra-1))

    return rv.interval(confianca)[1]
calcula_numero_desvios_tstudent_para_confianca(0.9545, 100)
calcula_numero_desvios_tstudent_para_confianca(0.99, 100)
from scipy.stats import norm





mean = 16

std = 4

rvNordeste = norm(loc=mean, scale=std)



mean = 31

std = 4

rvSudeste = norm(loc=mean, scale=std)





proporcao_Nordeste = 28/(28+42)

populacaoNordeste = rvNordeste.rvs(size=int(proporcao_Nordeste*100000), random_state=random_state)

print('Média {}'.format(np.mean(populacaoNordeste)))

print('Desvio {}'.format(np.std(populacaoNordeste)))

print('Tamanho {}'.format(len(populacaoNordeste)))





proporcao_Sudeste = 42/(28+42)

populacaoSudeste = rvSudeste.rvs(size=int(proporcao_Sudeste*100000), random_state=random_state)

print('Média {}'.format(np.mean(populacaoSudeste)))

print('Desvio {}'.format(np.std(populacaoSudeste)))

print('Tamanho {}'.format(len(populacaoSudeste)))



populacaoBrasil = np.concatenate((populacaoNordeste,populacaoSudeste))

np.random.shuffle(populacaoBrasil)



fig, axs = plt.subplots(1, 1, figsize=(14,6))



axs.hist(populacaoBrasil, density=True, facecolor='g', alpha=0.75, bins=100)

axs.grid(True)

axs.set_title('Distribuição Brasil')





plt.show()



print('Média {}'.format(np.mean(populacaoBrasil)))

print('Desvio {}'.format(np.std(populacaoBrasil)))

print('Tamanho {}'.format(len(populacaoBrasil)))
amostras = 1000

tamanho_amostra = 100

np.random.seed(1)

medias = np.zeros((amostras,1))

for i in range(0,amostras,1):

    medias[i]=np.mean(populacaoBrasil[np.random.randint(0, len(populacaoBrasil),tamanho_amostra)])
# A distribuição das médias aproxima-se de uma Normal, independente da distribuição original que gerou as amostras



n, bins, patches = plt.hist(medias, density=True, facecolor='g', alpha=0.75, bins=50)



mean_ = np.mean(medias)

std_ = np.std(medias)

print('Média {}'.format(mean_))

print('Desvio {}'.format(std_))



rv = norm(loc=mean_, scale=std_)



intervalo = np.linspace(mean_-3*std_,mean_+3*std_, num=50)

plt.plot(intervalo, rv.pdf(intervalo), 'k-', label='pdf')



plt.xlabel('Média')

plt.ylabel('Probabilidade')

plt.title('Histogram de médias')

plt.grid(True)

plt.show()
np.random.seed(1)

amostra_1 = populacaoBrasil[np.random.randint(0, len(populacaoBrasil),tamanho_amostra)]

media_1 =np.mean(amostra_1)

std_1 =np.std(amostra_1)



print('Média {}'.format(media_1))

print('Desvio {}'.format(std_1))

print('Desvio corrigido {}'.format(std_1/np.sqrt(tamanho_amostra)))

print('O valor real estará entre {} e {}'.format(media_1-2*std_1/np.sqrt(tamanho_amostra),

                                                media_1+2*std_1/np.sqrt(tamanho_amostra)))
np.random.seed(2)

amostra_Nordeste = populacaoNordeste[np.random.randint(0, len(populacaoNordeste),40)]

media_Nordeste =np.mean(amostra_Nordeste)

std_Nordeste =np.std(amostra_Nordeste)



print('Média {}'.format(media_Nordeste))

print('Desvio {}'.format(std_Nordeste))

print('Desvio corrigido {}'.format(std_Nordeste/np.sqrt(40)))

min_Nordeste = media_Nordeste-2*std_Nordeste/np.sqrt(40)

max_Nordeste = media_Nordeste+2*std_Nordeste/np.sqrt(40)

print('O valor real estará entre {} e {}'.format(min_Nordeste,max_Nordeste

                                                ))
np.random.seed(2)

amostra_Sudeste = populacaoSudeste[np.random.randint(0, len(populacaoSudeste),60)]

media_Sudeste =np.mean(amostra_Sudeste)

std_Sudeste =np.std(amostra_Sudeste)



print('Média {}'.format(media_Sudeste))

print('Desvio {}'.format(std_Sudeste))

print('Desvio corrigido {}'.format(std_Sudeste/np.sqrt(60)))

min_Sudeste = media_Sudeste-2*std_Sudeste/np.sqrt(60)

max_Sudeste = media_Sudeste+2*std_Sudeste/np.sqrt(60)

print('O valor real estará entre {} e {}'.format(min_Sudeste,max_Sudeste

                                                ))
print(min_Nordeste*proporcao_Nordeste + min_Sudeste*proporcao_Sudeste)

print(max_Nordeste*proporcao_Nordeste + max_Sudeste*proporcao_Sudeste)
repeticoes = 1000



# População normal

mean = 5

std = 1.3

rv_norm = stat.norm(loc=mean, scale=std)

populacao_norm = rv_norm.rvs(size=repeticoes, random_state=random_state)



fig, axs = plt.subplots(1, 1, figsize=(10,6))

_ = plt.boxplot(populacao_norm,vert =False, meanline =True)
media = np.mean(populacao_norm)

std = np.std(populacao_norm)



print('Média {}, STD {}'.format(media,std))



outliers1 = np.where(populacao_norm > (media+3*std))

outliers2 = np.where(populacao_norm < (media-3*std))



outliers = np.concatenate( (outliers1,outliers2), axis=1)



populacao_norm[outliers]
q25, q75 = np.percentile(populacao_norm, 25), np.percentile(populacao_norm, 75)



iqr= q75 - q25



outliers1 = np.where(populacao_norm > (q75+1.5*iqr))

outliers2 = np.where(populacao_norm < (q25-1.5*iqr))



outliers = np.concatenate( (outliers1,outliers2), axis=1)



populacao_norm[outliers]
if IN_KAGGLE:

    df_original = pd.read_csv("../input/2017.csv")

else:

    df_original = pd.read_csv("2017.csv")

    



df_original.head(2)
df = df_original.loc[:,[  'Happiness.Score',  'Economy..GDP.per.Capita.', 'Family',

       'Health..Life.Expectancy.', 'Freedom', 'Generosity',

       'Trust..Government.Corruption.','Dystopia.Residual']]
fig, axs = plt.subplots(1, 1, figsize=(10,6))

_ = plt.boxplot(df.T,vert =False, meanline =True)
populacao_norm = df.Family.values



q25, q75 = np.percentile(populacao_norm, 25), np.percentile(populacao_norm, 75)



iqr= q75 - q25



outliers1 = np.where(populacao_norm > (q75+1.5*iqr))

outliers2 = np.where(populacao_norm < (q25-1.5*iqr))



outliers = np.concatenate( (outliers1,outliers2), axis=1)



df_original.Country[outliers[0]]
media = np.mean(populacao_norm)

std = np.std(populacao_norm)



print('Média {}, STD {}'.format(media,std))



outliers1 = np.where(populacao_norm > (media+3*std))

outliers2 = np.where(populacao_norm < (media-3*std))



outliers = np.concatenate( (outliers1,outliers2), axis=1)



df_original.Country[outliers[0]]
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