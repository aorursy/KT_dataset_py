import numpy as np

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns



np.random.seed(123)
mu, sigma = 10, 4

rvs_norm = np.random.normal(mu, sigma, 100)

# we already know it is normal distribution.

# can k-s test find it?
# important: standardize ~ N(0,1)

rvs_std_norm = (rvs_norm - mu) / sigma
# plot

sns.kdeplot(rvs_std_norm)

plt.title("rvs_std_norm")

plt.show()
# by default,

# H_o: normal distribution

# H_a: not a normal distribution

# level_of_significance = 5% = 0.05



stats.kstest(rvs_std_norm, "norm")
rvs_uniform = np.random.uniform(0,1, 100)

# we already know it is uniform distribution.

# can k-s test find it?
# plot

sns.distplot(rvs_uniform)

plt.title("rvs_uniform")

plt.show()
# by default,

# H_o: uniform distribution

# H_a: not a uniform distribution

# level_of_significance = 5% = 0.05



stats.kstest(rvs_uniform, "uniform")