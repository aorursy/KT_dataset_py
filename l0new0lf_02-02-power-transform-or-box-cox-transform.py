import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



np.random.seed(123)
# gen. right-skew-norm (log norm) data

mu, sigma = 3, 1

s = np.random.lognormal(mu, sigma, 1000)



sns.distplot(s)

plt.show()
counts, bin_edges = np.histogram(s, 

                                 bins=5,

                                 density = True)

pdf = counts/(sum(counts))

ys = pdf

xs = bin_edges[1:]

# regular (x,y) plot where x is `bin_edges` and y is `pdf`

# plt.plot(xs, ys)



# try to avoid runtime error for logarithm

# make sure pdf has no `0` cz, log of x<=0 is not defined



# log-log plot

# (x,y) -> (log(x), log(y))

log_ys = np.log(ys)

log_xs = np.log(xs)



plt.plot(log_xs, log_ys)



plt.title('Log-Log Plot')

plt.show()
# log transformation: convert log-norm to norm distribution

log_transformed = np.log(s)



# plot

sns.distplot(log_transformed)

plt.show()
# exp transformation: convert norm distribution to log-norm 

# (reverse - get original data back)

exp_transformed = np.exp(log_transformed)



# plot

sns.distplot(exp_transformed)

plt.show()
import scipy.stats as stats
# gen

pareto_data = np.random.pareto(a=2, size=1000)



sns.distplot(pareto_data, kde=False)

plt.show()
# convert to guass/norm using power transform

to_guass, _ = stats.boxcox(pareto_data)



# plot

sns.distplot(to_guass, kde=True)

plt.show()
# sanity check w/ probplot (q-q plot)

fig = plt.figure()

ax1 = fig.add_subplot(111)



prob = stats.probplot(to_guass, dist=stats.norm, plot=ax1)

ax1.set_title('Probplot after Box-Cox transformation')



plt.show()