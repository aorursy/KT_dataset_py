import numpy as np

import scipy.stats as stats

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# A standard normal distribution, which means mean = 0, and stddev = 1

rvs = stats.norm(loc=0, scale=1)
# Generate or take out a sample of points from Normal Disb.

normal_sample = rvs.rvs(size=100000)
# We can plot also, to see whether the values are indeed taken from normal distribution

sns.set()

sns.distplot(normal_sample)
normal_sample = rvs.rvs(size=1000)

stats.probplot(normal_sample, dist="norm", plot=plt)

plt.show()
# This time it's a million

normal_sample = rvs.rvs(size=1000000)

stats.probplot(normal_sample, dist="norm", plot=plt)

plt.show()
new_rvs = stats.norm(loc=10, scale=2)

new_normal_sample = new_rvs.rvs(size=10000)

stats.probplot(normal_sample, dist="norm", plot=plt)

plt.show()
# We can plot exponential distribution also

sns.set(style="whitegrid")

plt.figure(figsize=(8, 4))

sns.distplot(stats.expon().rvs(size=10000))
# Let's compare them

expon_rvs = stats.expon().rvs(size=100000)

normal_rvs = stats.norm().rvs(size=100000)

stats.probplot(x=expon_rvs, dist=stats.norm(), plot=plt)
# I will tell you later what Pareto distribution is.

pareto_rvs = stats.pareto(b=2.62).rvs(size=1000000)

stats.probplot(x=pareto_rvs, dist=stats.expon(), plot=plt)

plt.show()