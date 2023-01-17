from scipy import stats

p = 0.2

Ber=stats.bernoulli(p)
Ber.pmf(0)
Ber.pmf(1)
Ber.cdf(0)
Ber.cdf(1)
Ber.cdf(0.5)

Ber.cdf(0.2)
Ber.cdf(-1)
Ber.cdf(2)
Ber.rvs(10)
Ber.mean()
Ber.var()
Ber.std()
m, v,  s,  k = stats.bernoulli.stats(0.8,moments='mvsk')

k
Ber.ppf(0.8)
#from scipy.stats import binom

n=10

p=0.3

X=stats.binom(n,p) 

print(X.pmf(9))

print( X.pmf(10) )

X.pmf(9) + X.pmf(10)
# otra forma 

1-X.cdf(8)



X.cdf(5) # P(X<=5)

X.ppf(0.9526510126)

X.ppf(X.cdf(5) ) 
X=stats.binom(4,0.5)

X.ppf(0.6875)
X.ppf(X.cdf(8))
import numpy as np

import matplotlib.pyplot as plt

bd1 = stats.binom(10, 0.3)

bd2 = stats.binom(10, 0.5)

bd3 = stats.binom(10, 0.75)

bd4 = stats.binom(10, 0.95)

k = np.arange(11)

#print(bd1.pmf(k))

plt.plot(k, bd1.pmf(k), 'o-b')

plt.plot(k, bd2.pmf(k), 'd-r')

plt.plot(k, bd3.pmf(k), 's-g')

plt.plot(k, bd4.pmf(k), '*-k')

plt.title('Distribución Binomial')

plt.legend(['p=0.3', 'p=0.5', 'p=0.7','p=0.95'])

plt.xlabel('X')

plt.ylabel('P(X)')

plt.show()
from scipy.stats import binom

p=0.5

n=10

mean, var, skew, kurt = binom.stats(n,p,moments='mvsk')

print(mean)

print(var)

print(skew)

print(kurt)
p=1/2.5

#print(p)

Z=stats.geom(p)

Z.pmf(1)
print(Z.pmf(4))

Z.pmf(5)
### P(X <= 4)$

Z.cdf(4)
### P(X > 3)=1-P(X<=3)$

1-Z.cdf(3)
px=0.1

py=0.9

X=stats.geom(px)

Y=stats.geom(py)

### valores del eje x comun para graficar 

x = np.arange(1,20)

fig, ax = plt.subplots(1, 1)

ax.plot(x, Y.pmf(x), 'bo', ms=2, label='geom pmf')

ax.plot(x, X.pmf(x), 'gs', ms=2, label='geom pmf')

ax.vlines(x, 0, Y.pmf(x), colors='b', lw=2, alpha=0.5)

ax.vlines(x, 0, X.pmf(x), colors='g', lw=2, alpha=0.5)

plt.legend(['p=0.1', 'p=0.9'])

plt.show()
n=4

p=0.1

X=stats.nbinom(n,p) 

print(X.pmf(20))

print(X.pmf(19))

print(X.pmf(21))

print(X.cdf(10)) 
print(X.mean())

print(n*(1-p)/p)

print(X.var())

n*(1-p)/(p**2)
li=round( X.mean()-np.sqrt(X.var()) )

ls=round( X.mean()+np.sqrt(X.var()) )

x=np.arange(li,ls)

#print(x)

fig, ax = plt.subplots(1, 1)

ax.plot(x, X.pmf(x), 'bo', ms=2, label='BN pmf')

ax.vlines(x, 0, X.pmf(x), colors='b', lw=2, alpha=0.5)

#ax.vlines(x, 0, X.pmf(x), colors='g', lw=2, alpha=0.5)

plt.show()
### cual es el valor de mayor probabilidad 

probs=X.pmf(x)

print(np.max(probs))

x[probs==np.max(probs)]

X.pmf(27)
M=300

n=100

N=4

X=stats.hypergeom(M,n,N)

## ¿Cuál es la probabilidad que todas sean del proveedor local?

X.pmf(4)
## ¿Cuál es la probabilidad que dos o mas piezas sean del proveedor local?

print(1-X.cdf(1))

## otra forma 

X.pmf(2)+X.pmf(3)+X.pmf(4)
## ¿Cuál es la probabilidad que al menos una piezas sea del proveedor local?

1-X.pmf(0)
#%%script /usr/bin/python3.8 

from math import comb 

print( (comb(100,4)*comb(200,0) )/comb(300,4) )
### distribución de Poisson
from scipy.stats import poisson

mu=10

poisson.pmf(0, mu)
mean, var, skew, kurt = poisson.stats(mu, moments='mvsk')



print(mean, var, skew, kurt )
### se usa la propiedad del complemento 

mu=1

1-poisson.pmf(0,mu)
mu=2

poisson.cdf(4,mu)