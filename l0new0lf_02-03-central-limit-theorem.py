import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# for selecting w/o replacement

from numpy.random import default_rng

rng = default_rng()



np.random.seed(123)
# gen some disribution w/ finite mu, sigma

distribution = np.array([])

for _part_mu in range(0, 100, 10):

    _part_sigma = np.random.choice(30)

    _part_dist = np.random.normal(_part_mu, _part_sigma, 1000)

    distribution = np.append(distribution, _part_dist)



old_mu = np.mean(distribution)

old_var = pow(np.std(distribution), 2)



# plot

sns.distplot(distribution)

plt.title(f"Distribution\nmu: {old_mu:.2f}\n var: {old_var:.2f}")

plt.show()
def CLT(rvs, k=10000, n=50, replacement=True):

    m = len(rvs)

    if n>m: raise("n should be <= m")

    

    # 1. gen k random samples of size n

    random_samples = []

    if replacement is True:

        for _ in range(0,k):

            sample_idxs = np.random.choice(m, size=n) # with replacement

            sample = rvs[sample_idxs]

            random_samples.append(sample)

    elif replacement is False:

        for _ in range(0,k):

            sample_idxs = rng.choice(m, size=n, replace=False) # without replacement

            sample = rvs[sample_idxs]

            random_samples.append(sample)

    

    # 2. gen k number of means --  sampling distribution of sample means

    normal_dist = []

    for sample in random_samples:

        normal_dist.append(np.mean(sample))

        

    new_mu    = np.mean(normal_dist) # almost same as old mu (depending on k and n)

    new_var   = pow(np.std(normal_dist), 2) # same as old_var/n

    

    # Note: we are checking only errors due to sampling here (large k and n values => errors converge to 0)

    print("DELTAS")

    print(f"mu: \t before: {old_mu} \t after:{new_mu} \t error: {np.abs(old_mu-new_mu)}")

    print(f"var: \t before: {old_var/n} \t after:{new_var} \t error: {np.abs((old_var/n)-new_var)}")

    

    return normal_dist
to_guass = CLT(distribution)

mu  = np.mean(to_guass)

var = pow(np.std(to_guass),2)





# plot

sns.distplot( to_guass )



plt.title(f'Same distibution after CLT\nmu: {mu:.2f}\nvar: {var:.2f}')

plt.show()