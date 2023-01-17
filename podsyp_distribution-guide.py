%matplotlib inline



import matplotlib.pyplot as plt

from IPython.display import Math, Latex

from IPython.core.display import Image

import seaborn as sns

sns.set(color_codes=True)

sns.set(rc={'figure.figsize':(5,5)})
from scipy.stats import uniform
n = 10000

start = 10

width = 20

data_uniform = uniform.rvs(size=n, loc = start, scale=width)
ax = sns.distplot(data_uniform,

                  bins=100,

                  kde=True,

                  color='skyblue',

                  hist_kws={"linewidth": 15,'alpha':1})

ax.set(xlabel='Uniform Distribution ', ylabel='Frequency')
from scipy.stats import norm
data_normal = norm.rvs(size=10000,loc=0,scale=1)
ax = sns.distplot(data_normal,

                  bins=100,

                  kde=True,

                  color='skyblue',

                  hist_kws={"linewidth": 15,'alpha':1})

ax.set(xlabel='Normal Distribution', ylabel='Frequency')
from scipy import stats
stats.kstest(data_normal, 'norm')
stats.anderson(data_normal, dist='norm')
from scipy.stats import gamma
data_gamma = gamma.rvs(a=5, size=10000)
ax = sns.distplot(data_gamma,

                  kde=True,

                  bins=100,

                  color='skyblue',

                  hist_kws={"linewidth": 15,'alpha':1})

ax.set(xlabel='Gamma Distribution', ylabel='Frequency')
from scipy.stats import expon
data_expon = expon.rvs(scale=1,loc=0,size=1000)
ax = sns.distplot(data_expon,

                  kde=True,

                  bins=100,

                  color='skyblue',

                  hist_kws={"linewidth": 15,'alpha':1})

ax.set(xlabel='Exponential Distribution', ylabel='Frequency')
from scipy.stats import poisson
data_poisson = poisson.rvs(mu=3, size=10000)
ax = sns.distplot(data_poisson,

                  bins=30,

                  kde=True,

                  color='skyblue',

                  hist_kws={"linewidth": 15,'alpha':1})

ax.set(xlabel='Poisson Distribution', ylabel='Frequency')
from scipy.stats import binom
data_binom = binom.rvs(n=10,p=0.8,size=10000)
ax = sns.distplot(data_binom,

                  kde=False,

                  color='skyblue',

                  hist_kws={"linewidth": 15,'alpha':1})

ax.set(xlabel='Binomial Distribution', ylabel='Frequency')
from scipy.stats import bernoulli
data_bern = bernoulli.rvs(size=10000,p=0.6)
ax= sns.distplot(data_bern,

                 kde=False,

                 color="skyblue",

                 hist_kws={"linewidth": 15,'alpha':1})

ax.set(xlabel='Bernoulli Distribution', ylabel='Frequency')