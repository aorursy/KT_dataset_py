import numpy as np

import seaborn as sns

from scipy import stats

import matplotlib.pyplot as plt



# Generate 1000 points from the normal distrbution

rvs = stats.norm.rvs(size=10000)

sns.set_style('whitegrid')

# sns.kdeplot(rvs)

sns.distplot(rvs)

plt.show()
# Now compare it with normal distribution using K-S test

stats.kstest(rvs=rvs, cdf='norm')
# Lets compare the cdf of two samples taken from the same distribution



def generate_cdf_plots():

    rvs1 = sorted(stats.norm(scale=1, loc=0).rvs(size=10000))

    rvs2 = sorted(stats.norm(scale=2, loc=0).rvs(size=10000))

    normal_sample_cdf = np.array(list(range(0, 10000)))

    normal_sample_cdf = normal_sample_cdf/normal_sample_cdf.max()

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)

    plt.plot(rvs1, normal_sample_cdf)

    plt.plot(rvs2, normal_sample_cdf)

    plt.show()



generate_cdf_plots()
rvs = stats.norm().rvs(size=10000)

stats.kstest(rvs, "expon")
# Comparing normal and exponential samples

rvs_norm = stats.norm().rvs(100)

rvs_expon = stats.expon().rvs(100)

stats.ks_2samp(rvs_norm, rvs_expon)
# Comparing two exponential samples

rvs_expon1 = stats.expon().rvs(10000)

rvs_expon2 = stats.expon().rvs(10000)

stats.ks_2samp(rvs_expon1, rvs_expon2)