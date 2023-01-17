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
# punto 2
import scipy.stats as ss

X = ss.binom(8,0.73)
pr = X.pmf(0)  #a
print(pr)
import scipy.stats as stats
X=stats.binom(8,0.73) #b
X.cdf(1)
#punto 3
from scipy import stats
n=3
p=0.43
X=stats.nbinom(n,p) 
print(X.pmf(7))
# punto 4
from scipy.stats import poisson
mu=1.1*6
poisson.pmf(8,mu)
# Punto 5
M=10+6
n=5
N=6
X=stats.hypergeom(M,n,N)
1-X.pmf(0)
# Punto 6
from scipy.stats import expon  
expon.cdf(12,loc=0,scale=15)
# Punto 7
1-expon.cdf(18,loc=0,scale=10)
# punto 8 
Y=stats.uniform(loc=18,scale=22)
print(Y.mean())
# C
stats.uniform.cdf(16.49,loc=7,scale=22) 
# D
1-stats.uniform.cdf(22.06,loc=7,scale=22) 
# E
stats.uniform.ppf(0.93,loc=7,scale=22)
# Punto 9 
# C
stats.uniform.cdf(11.74,loc=5,scale=35) 
# D
1-stats.uniform.cdf(5.38,loc=5,scale=35) 
# E
stats.uniform.ppf(0.9,loc=5,scale=35)
# Punto 10
from scipy.stats import norm
import matplotlib.pyplot as plt
print(6*norm.cdf(0.9,loc=78,scale=6)+78)
# Punto 11
from scipy.stats import poisson
mu=12.5
poisson.pmf(9,mu)
# 12 
# a
import numpy as np
import scipy.special
alfa=100
beta=1/2
EX=alfa*scipy.special.gamma(1+1/beta)
EX
# b
X=stats.weibull_min(c=2,scale=10000)
X.cdf(8000)
# c
n=10
p=0.4727
X=stats.binom(n,p)
print(X.pmf(6))
# 2

#a
n=18
p=0.29
X=stats.binom(n,p)
print(X.pmf(16))
# b
n=18
p=0.29
X=stats.binom(n,p)
1-X.cdf(3)
# 3
p=1/2
Z=stats.geom(p)
Z.pmf(4)
# 4
from scipy.stats import poisson
mu=10*2
poisson.pmf(22,mu)
# 5
M=15
n=3
N=5
X=stats.hypergeom(M,n,N)
X.pmf(2)
# 6
from scipy.stats import expon  
1-expon.cdf(60,scale=10) 
# 7
from scipy.stats import expon  
expon.ppf(0.95,scale=10)
# 8
# a
# (a+b)/2 = 22
# (b-c)= 30
# punto a
# a= 5 minimo 
# Punto b
# b=25 maximo
A = np.matrix([[1/2, 1/2],[1, -1]])
b = np.matrix([[20],[30]])
x = (A**-1)*b
x
# C
stats.uniform.cdf(11.74,loc=5,scale=35) 
# D
1-stats.uniform.cdf(5.38,loc=5,scale=35) 
# E
stats.uniform.ppf(0.9,loc=5,scale=35)
#9
#a
#(a+b)/2=35
#(b-a)=3.4
#entonces el valor minimo es 
#a
A = np.matrix([[1/2, 1/2],[1, -1]])
b = np.matrix([[35],[3.4]])
x = (A**-1)*b
x
# 10 
from scipy.stats import norm
print(norm.ppf(0.8643))
norm.cdf(1.0998442239755872,loc=0,scale=1)
# 11
from scipy.stats import poisson
mu=5*3
1-poisson.pmf(12,mu)
# 12
X=stats.weibull_min(c=1/2,scale=100)
1-X.cdf(5000)
X.cdf(400)
# 2

#a
n=16
p=0.37
X=stats.binom(n,p)
print(X.pmf(16))
# b
n=16
p=0.37
X=stats.binom(n,p)
1-X.cdf(2)
# 3
n=4
p=0.3
X=stats.nbinom(n,p) 
X.pmf(3)
# 4
from scipy.stats import poisson
poisson.pmf(37,5*7)
# 5
import scipy.stats as stats
n=15
p=1-(4800/11500)
X=stats.binom(n,p)
X.cdf(5)
# 6
from scipy.stats import expon  
1-expon.cdf(25,scale=15) 
# 7
1-(expon.cdf(8,scale=6)-expon.cdf(6,scale=6))
# 8
import numpy as np
#a
a0 = np.matrix([[1/2, 1/2],[1, -1]])
a1 = np.matrix([[35],[3.4]])
(np.linalg.inv(a0)*a1).T
# 9
#a,b
a0 = np.matrix([[1/2, 1/2],[1, -1]])
a1 = np.matrix([[18],[22]])
(np.linalg.inv(a0)*a1).T
# C
stats.uniform.cdf(16.49,loc=7,scale=29) 

# D
1-stats.uniform.cdf(22.06,loc=7,scale=29) 
# E
stats.uniform.ppf(0.93,loc=7,scale=29)
#10
from scipy.stats import norm
import math as mat
sig=mat.sqrt(36)
mu=78
norm.ppf(1-0.1)*sig+mu
# 11
1-expon.cdf(3,scale=2.4) 
X=stats.gamma(a=3,scale=2.4)
X.cdf(3)
# 12
# a
import numpy as np
import scipy.special
alfa=5000
beta=1/2
a=alfa*scipy.special.gamma(1+1/beta)
a
# b
X=stats.weibull_min(c=beta,scale=alfa)
1-X.cdf(6000)
