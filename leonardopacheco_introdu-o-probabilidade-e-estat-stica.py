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

random.seed(random_state)

escolhas = ['Cara','Coroa']
lancamentos = 100
resultados = random.choices(population=escolhas, weights=[6/10,4/10], k=lancamentos)

fig, ax = plt.subplots(figsize=(10,6))
ind = range(1,len(escolhas)+1)
proporcoes = [resultados.count('Cara')/lancamentos,resultados.count('Coroa')/lancamentos]
plt.bar(ind,proporcoes,align='center')
ax.set_xticks(ind)
ax.set_xticklabels(escolhas)


plt.xlabel('Resultado')
plt.ylabel('Proporção')
plt.title('Resultados de lançamentos de moeda não equilibrada')
plt.grid(True)
plt.show()
# exemplo: 3 variáveis de entrada com relacionamento não linear entre elas

np.random.seed(1)
x = np.random.random_sample(size=1000)
y = np.random.random_sample(size=1000)
z = np.random.random_sample(size=1000)
w = x**2-y**2+z**3
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
from sklearn.neural_network import MLPRegressor


x_ = np.concatenate((np.reshape(x,(-1,1)),np.reshape(y,(-1,1)),np.reshape(z,(-1,1))), axis=1)
y_ = np.reshape(w,(-1,1))

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


plt.subplots(figsize=(14,6))
plt.plot(y_test,'r.')
plt.plot(estimator.predict(x_test),'b*')

plt.ylabel('w')
plt.title('w = x**2-y**2+z**3')
plt.grid(True)
plt.show()
# Cálculo analítico baseado em contagem 
combinacao(30,15)/possibilidades(30,2)
# Cálculo analítico baseado na probabilidade básica
probabilidades = np.zeros((31,1))
for i in range(0,31,1):
    probabilidades[i]=combinacao(30,i)*((1/2)**(i))*((1/2)**(30-i))

plt.bar(range(0,31,1),probabilidades[:,0], facecolor='g', alpha=0.75)

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
repeticoes = 100
np.random.seed(1)
resultado = np.random.choice(a=Moeda, p=Equilibrio, replace=True, size=(repeticoes,lancamentos))
resultado=np.sum(resultado, axis=1)
probabilidades,_ = np.histogram(a=resultado, density=True, bins=range(0,31,1))

#n, bins, patches = plt.hist(resultado, density=True, facecolor='g', alpha=0.75, bins=range(0,31,1))
plt.bar(range(0,30,1),probabilidades, facecolor='g', alpha=0.75)

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
plt.bar(range(0,30,1),probabilidades, facecolor='g', alpha=0.75)

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
probabilidades,_ = np.histogram(a=resultado, density=True, bins=range(0,31,1))

#n, bins, patches = plt.hist(resultado, density=True, facecolor='g', alpha=0.75, bins=range(0,31,1))
plt.bar(range(0,30,1),probabilidades, facecolor='g', alpha=0.75)

plt.xlabel('# Caras')
plt.ylabel('Probabilidade')
plt.title('Histogram Moeda')
plt.grid(True)
plt.show()
Dado = [1,2,3,4,5,6]
Equilibrio = [1/6,1/6,1/6,1/6,1/6,1/6]
repeticoes = 100
np.random.seed(1)
resultado = np.random.choice(a=Dado, p=Equilibrio, replace=True, size=(2, repeticoes))
resultado=np.sum(resultado, axis=0)

probabilidades,_ = np.histogram(a=resultado, density=True, bins=range(2,14,1))
probabilidades

n, bins, patches = plt.hist(resultado, density=True, facecolor='g', alpha=0.75, bins=range(2,14,1))
plt.bar(range(2,13,1), probabilidades*100, facecolor='g', alpha=0.75)

plt.xlabel('Soma dos dados')
plt.ylabel('Probabilidade%')
plt.title('Histograma Dados')
plt.grid(True)
plt.show()
# Usando scipy

from scipy.stats import binom
tentativas = 30
rv = binom(tentativas, 1/2)

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
# Normal
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html#scipy.stats.norm
# Equivalente a np.random.normal(loc=0.0, scale=1.0, size=10)

from scipy.stats import norm
from scipy.stats import kstest

mean = 0
std = 1
rv = norm(loc=mean, scale=std)
np.random.seed(1)
resultado = rv.rvs(size=1000)

n, bins, patches = plt.hist(resultado, density=True, facecolor='g', alpha=0.75, bins=50)
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
print('Variância: {}'.format(rv.var()))
print('Desvio padrão: {}'.format(rv.std()))

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





s = 0.8
repeticoes = 100000

rv = stat.lognorm(s=s)

populacao = rv.rvs(size=repeticoes, random_state=random_state)


n, bins, patches = plt.hist(populacao, density=True, facecolor='g', alpha=0.75, bins=50)


plt.title('Distribuição logonormal')
plt.xlabel('Valores')
plt.ylabel('Probabilidade')
plt.grid(True)
plt.show()
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
np.random.seed(1)
amostra = populacao[np.random.randint(0, len(populacao),int(0.01*repeticoes))]
# Estatística descritiva da população

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

axs[0].plot(populacao_lognorm,'.')
axs[0].grid(True)
axs[0].set_title('populacao_lognorm')

axs[1].plot(populacao_norm,'.')
axs[1].grid(True)
axs[1].set_title('populacao_norm')

plt.show()
fig, axs = plt.subplots(1, 2, figsize=(14,6))

axs[0].hist(populacao_lognorm, density=True, facecolor='g', alpha=0.75, bins=50)
axs[0].grid(True)
axs[0].set_title('populacao_lognorm')

axs[1].hist(populacao_norm, density=True, facecolor='g', alpha=0.75, bins=50)
axs[1].grid(True)
axs[1].set_title('populacao_norm')

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
s = 1
np.random.seed(1)
rv_lognorm = stat.lognorm(s=s,loc=mean-1.3)
populacao_lognorm = rv_lognorm.rvs(size=repeticoes, random_state=random_state)


print('Mínimo {}'.format(np.min(populacao_lognorm)))
print('Máximo {}'.format(np.max(populacao_lognorm)))
print('Média {}'.format(np.mean(populacao_lognorm)))
print('Desvio {}'.format(np.std(populacao_lognorm)))

fig, axs = plt.subplots(1, 1, figsize=(14,6))

axs.hist(populacao_lognorm, density=True, facecolor='g', alpha=0.75, bins=100)
axs.grid(True)
axs.set_title('populacao_lognorm')


plt.show()
# Vamos extrair 1000 amostras e calcular suas médias
amostras = 10000
np.random.seed(1)
medias = np.zeros((amostras,1))
for i in range(0,amostras,1):
    medias[i]=np.mean(populacao_lognorm[np.random.randint(0, len(populacao_lognorm),amostras)])


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
mean_ = np.mean(medias)
std_ = np.std(medias)
print('Média amostras {}'.format(mean_))
print('Desvio amostras {}'.format(std_))

print('Intervalo de 95,45% de confiança {} - {}'.format(mean_-2*std_,mean_+2*std_))
np.random.seed(1)
for i in range(0,10,1):
    print('Média da amostra {}: {}'.format(i,
                                np.mean(populacao_lognorm[np.random.randint(0, len(populacao_lognorm),amostras)])))
np.random.seed(1)
for i in range(0,10,1):
    media_i = np.mean(populacao_lognorm[np.random.randint(0, len(populacao_lognorm),amostras)])
    print('Amostra {}, média {}, com margem de erro {} - {}'.format(i,
                                                                    media_i,
                                                                   media_i-2*std_,
                                                                   media_i+2*std_))
amostras = 1000
np.random.seed(1)
medias = np.zeros((amostras,1))
for i in range(0,amostras,1):
    medias[i]=np.mean(populacao_lognorm[np.random.randint(0, len(populacao_lognorm),100)])

mean_ = np.mean(medias)
std_ = np.std(medias)
print('Média amostras {}'.format(mean_))
print('Desvio amostras {}'.format(std_))

print('Intervalo de 95,45% de confiança {} - {}'.format(mean_-2*std_,mean_+2*std_))

print('Margem de erro: {}'.format(100*2*std_/mean_))
np.random.seed(1)
amostra_100_elementos = populacao_lognorm[np.random.randint(0, len(populacao_lognorm),100)]
print('Média amostra {}'.format(np.mean(amostra_100_elementos)))
print('Desvio amostra {}'.format(np.std(amostra_100_elementos)))
print('Desvio estimado amostras {}'.format(np.std(amostra_100_elementos)/np.sqrt(len(amostra_100_elementos))))
print('Margem erro {}'.format(100*2*np.std(amostra_100_elementos)/np.sqrt(len(amostra_100_elementos))/np.mean(amostra_100_elementos)))


# o desvio padrão das médias das amostras é desvio da população / sqrt(samples)
#print(np.std(populacao_lognorm)/np.sqrt(100) )
from scipy.stats import norm
from scipy.stats import kstest

mean = 0
std = 1
rv = norm(loc=mean, scale=std)

print(rv.std() )
print(rv.interval(0.9545))

print(rv.interval(0.99))
from scipy.stats import t
from scipy.stats import kstest


rv = t(df=(100-1))

print(rv.std() )
print(rv.interval(0.9545))

print(rv.interval(0.99))
if IN_KAGGLE:
    df = pd.read_csv("../input/2017.csv")
else:
    df = pd.read_csv("2017.csv")
    

df.head(2)
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
ax1.plot( y)
plt.grid()
plt.tight_layout()
plt.show()

print(len(x))

print(np.corrcoef(x,y))
# https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

fig, ax = plt.subplots(2,1,figsize=(14,8))
plot_acf(y,ax=ax[0])
plot_pacf(y,ax=ax[1])

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
from scipy.stats import binom
tentativas = 30
rv_honesta = binom(tentativas, 1/2)
populacao_honesta = rv_honesta.rvs(size=1000000)
print(rv_honesta.mean())
print(rv_honesta.std())

rv = binom(tentativas, 1/2.5)
resultado = rv.rvs(size=100)
print(np.mean(resultado))
amostras = 500
medias = np.zeros((amostras,1))
for i in range(0,amostras,1):
    medias[i]=np.mean(populacao_honesta[np.random.randint(0, len(populacao_honesta),100)])

print(medias.mean())
print(medias.std())    

fig, axs = plt.subplots(1, 1, figsize=(14,6))


axs.hist(medias, density=True, facecolor='g', alpha=0.75, bins=50)
axs.grid(True)
axs.set_title('Distribuição das médias')

plt.show()
rv_honesta.cdf(12.23)
if IN_KAGGLE:
    df = pd.read_csv("../input/2016.csv")
else:
    df = pd.read_csv("2016.csv")
df.head(2)
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