# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from scipy import stats

## sf= 1- cdf 

stats.uniform.sf(35,loc=30,scale=10) 
# otra forma 

1-stats.uniform.cdf(35,loc=30,scale=10)
stats.uniform.ppf(0.1,loc=30,scale=10)
X=stats.uniform(loc=30,scale=10)

print(X.mean())

X.var()

X.pdf(31)

X.pdf(25)
np.random.seed(12345)

stats.uniform.rvs(30,scale=10,size=5) 

X.rvs(5) 
#funciones para cláculos con la distribución normal 

#la normal

from scipy.stats import norm

import matplotlib.pyplot as plt

#fig, ax = plt.subplots(1, 1)



# función de densidad 

# f(x) para una normal con media 10 y sd 3

# evaluada en 9, es decir f(9)

print(norm.pdf(9,loc=10,scale=3) )

# otra forma

#X=norm(loc=10,scale=3)

#print(X.pdf(9))



# Hallar P(Z\leq z), donde Z es una normal estándar 

print( norm.cdf(-1.959964,loc=0,scale=1) )



# Hallar P(X\leq x), donde X es una normal con media 10 y desviación estándar

# 3. Hallar   P(X\leq 4.120108)

norm.cdf(4.120108,loc=10,scale=3)

# Hallar P(X\qeq x), donde X es una normal con media 10 y desviación estándar

# 3. Hallar   P(X\geq 4.120108)

print(1 - norm.cdf(4.120108,loc=10,scale=3))

print(norm.sf(4.120108,10,3))



# dada la probabilidad, encontrar el valor de la variable 

# hallar z tal que P(Z\leq Z )=\Phi(z)=0.025 donde F es la función de distribución 

# de una normal estándar 

print(norm.ppf(0.025))

# redondeado a dos cifras

round(norm.ppf(0.025),ndigits=2)

# si X es una V. A. con distribución normal de media 10 y desviación estándar 

# 3, hallar x tal que P(X\leq x )= 0.025

norm.ppf(0.025,loc=10,scale=3) 
print(norm.rvs(loc=10,scale=3,size=1))

# 200 valores de una normal de media 10 y sigma=3

np.random.seed(12345)

x=norm.rvs(loc=10,scale=3,size=200)

#print(x)
# valores de la variable 

z = np.linspace(norm.ppf(0.001,loc=10,scale=3),norm.ppf(0.999,loc=10,scale=3), 101)

# valores de f(z)

y=norm.pdf(z,loc=10,scale=3) 

fig, ax = plt.subplots(1, 1)

ax.hist(x, density=True, histtype='stepfilled', alpha=0.2)

ax.plot(z, y, 'r-', lw=3, alpha=0.6, label='norm pdf') 

ax.legend(['Densidad norm'])

ax.plot(x, np.zeros(x.shape), 'b+', ms=20)  # rug plot

#sns.distplot(x, rug=True,hist=False,kde=False,ax=ax,rug_kws={"color":"b","alpha":0.2})

plt.show()
f, (ax1, ax2) = plt.subplots(1, 2,sharex=True, sharey=True)

z = np.linspace(norm.ppf(0.001,loc=10,scale=3),norm.ppf(0.999,loc=10,scale=3), 101)

y1=norm.pdf(z,loc=10,scale=1)

y2=norm.pdf(z,loc=10,scale=2)

y3=norm.pdf(z,loc=10,scale=3) 



ax1.plot(z,y1,'r-',lw=3,alpha=0.6,label=r'$\sigma=1$')

ax1.plot(z,y2,'b-',lw=3,alpha=0.6,label=r'$\sigma=2$')

ax1.plot(z,y3,'g-',lw=3,alpha=0.6,label=r'$\sigma=3$')

ax1.set_title(r'La misma media $(\mu=10)$')

ax1.legend(loc='upper left', shadow=True, fontsize='x-large')

# graficar densidad normal con media 8 y 10 y 12 sigma=2  

z2 = np.arange(norm.ppf(0.001,loc=8,scale=2),norm.ppf(0.999,loc=12,scale=2),0.01)

y21=norm.pdf(z2,loc=8,scale=2)

y22=norm.pdf(z2,loc=10,scale=2)

y23=norm.pdf(z2,loc=12,scale=2) 



ax2.plot(z2,y21,'r-',lw=3,alpha=0.6,label=r'$\mu=8$')

ax2.plot(z2,y22,'b-',lw=3,alpha=0.6,label=r'$\mu=10$')

ax2.plot(z2,y23,'g-',lw=3,alpha=0.6,label=r'$\mu=12$')

ax2.set_title(r'La misma desviación $\sigma=2$')

ax2.legend(loc='upper left', shadow=True, fontsize='x-large')



#legend = axs[0].legend(loc='upper center', shadow=True, fontsize='x-large')

#legend = axs[1].legend(loc='upper center', shadow=True, fontsize='x-large')

plt.show()
X=norm(loc=165,scale=8/np.sqrt(36))

print(X.sf(167))

1-X.cdf(167)
from scipy.stats import expon  

#fig, ax = plt.subplots(1, 1)

print(25*np.exp(-25*0.04))

expon.pdf(0.04,scale=1/25) 
expon.cdf(0.1,scale=1/25)
# dada la probabilidad, encontrar el valor de la variable 

# hallar x tal que P(X\leq x )=0.91791 donde F es la función de distribución 

# de una normal estándar 

expon.ppf(0.91791,scale=1/25)

# redondeado a dos cifras

round( expon.ppf(0.91791,scale=1/25), 3)
print(expon.rvs(scale=1/25,size=1))

# 200 valores de una normal de media 10 y sigma=3

np.random.seed(12345)

x=expon.rvs(scale=1/25,size=200)
# graficar histograma y densidad exponencial con media = 1/25 

z = np.linspace(expon.ppf(0.001,scale=1/25),expon.ppf(0.999,scale=1/25), 101)

y=expon.pdf(z,scale=1/25) 
plt.hist(x, density=True, histtype='stepfilled', alpha=0.2)

#sns.distplot(x, rug=True,hist=False,kde=False,rug_kws={"color":"b","alpha":0.2})

plt.plot(z, y, 'r-', lw=3, alpha=0.6, label='norm pdf') 

plt.legend(['Densidad expon'], frameon=False )

plt.show()
fig, ax = plt.subplots(1, 1)

z = np.linspace(expon.ppf(0.001,scale=1/0.5),expon.ppf(0.999,scale=1/25), 101)

y1=expon.pdf(z,scale=1/0.5) 

y2=expon.pdf(z,scale=1/2)

y3=expon.pdf(z,scale=1/10)  



ax.plot(z, y1, 'r-', lw=3, alpha=0.6, label=r'$\lambda=0.5$') 

ax.plot(z, y2, 'b-', lw=3, alpha=0.6, label=r'$\lambda=2$') 

ax.plot(z, y3, 'g-', lw=3, alpha=0.6, label=r'$\lambda=10$') 

#legend = ax.legend(loc='upper center', frameon=False,shadow=True, fontsize='x-large')

legend = ax.legend(loc='upper center',shadow=True, fontsize='x-large')

# Put a nicer background color on the legend.

#legend.get_frame().set_facecolor('C0') 



#ax.legend(['Densidad expon'], frameon=False )

plt.show()
### Momentos 
# media

print(expon(scale=1/25).mean()) 

# desviación estándar 

expon(scale=1/25).std()

# desviación varianza 

expon(scale=1/25).var()

1/25
### calculo simbólico 
from sympy import stats 

from sympy.abc import x

X=stats.Normal('x',0,10)

stats.density(X)(x)
# P(X<0)

stats.cdf(X)(0)
# P(X<0)

stats.P(X<0)
stats.E(X).evalf()
stats.E(X**2).evalf()
stats.E(abs(X)**(1/2)).evalf()
import sympy as S

from sympy import stats 

L=S.Symbol("L", real = True, positive = True) 

y=S.Symbol("y")

t=S.Symbol("t" )

Y=stats.Exponential('y',L)

stats.density(Y)(y)
stats.E(Y).evalf()

stats.E(Y**2).evalf()
stats.E(S.exp(t*Y)).evalf()
from scipy import stats

# a = parámetro de forma 

X1=stats.gamma(a=1,scale=1)

X2=stats.gamma(a=8.3,scale=1/2)

X3=stats.gamma(a=7.5,scale=1/3.75)

x = np.linspace(X1.ppf(0.0001),X1.ppf(0.9999), 100)

y1 = X1.pdf(x)

y2 = X2.pdf(x)

y3 = X3.pdf(x)

fig, ax = plt.subplots(1, 1)

ax.plot(x, y1 ,'r-', lw=3, alpha=0.6, label=r'$\alpha=1 , \lambda=1$')

ax.plot(x, y2 ,'b-', lw=3, alpha=0.6, label=r'$\alpha=8.3 , \lambda=1/2$')

ax.plot(x, y3 ,'k--', lw=3, alpha=0.6, label=r'$\alpha=7.5 , \lambda=1/3.75$')

legend = ax.legend(loc='upper right',shadow=True, fontsize='x-large')

plt.show()
X=stats.gamma(a=5,scale=1/0.5)

print(X.cdf(10))

X.cdf(5)
X=stats.weibull_min(c=1/2,scale=5000)

1-X.cdf(6000)

#np.exp(-np.sqrt(6000/5000)) 
#### ojo, no da la respuesta de mongomery 

import sympy as S

from sympy import stats 

y=S.Symbol("y")

a=S.Symbol("a")

l=S.Symbol("l")

Y=stats.Weibull('y',1/2,5000)

stats.E(Y).evalf() 

#stats.density(Y)(y) 
from scipy import stats

# a = parámetro de forma 

X1=stats.weibull_min(c=1,scale=1)

X2=stats.weibull_min(c=2,scale=3.4)

X3=stats.weibull_min(c=6.2,scale=4.5)

x = np.linspace(X2.ppf(0.0001),X2.ppf(0.9999), 100)

y1 = X1.pdf(x)

y2 = X2.pdf(x)

y3 = X3.pdf(x)

fig, ax = plt.subplots(1, 1)

ax.plot(x, y1 ,'r-', lw=3, alpha=0.6, label=r'$c=1 , \lambda=1$')

ax.plot(x, y2 ,'b-', lw=3, alpha=0.6, label=r'$c=2 , \lambda=3.4$')

ax.plot(x, y3 ,'k--', lw=3, alpha=0.6, label=r'$c=6.2 , \lambda=4.5$')

legend = ax.legend(loc='upper right',shadow=True, fontsize='x-large')

plt.show()
1-stats.t.cdf(3.2,df=15)
### otra forma 

stats.t.sf(3.2,df=15)
stats.chi2.sf(38.4,24)