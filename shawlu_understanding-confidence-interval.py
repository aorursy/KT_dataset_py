import numpy as np

from matplotlib import pyplot as plt

from sklearn.neighbors import KernelDensity
def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):

    """Kernel Density Estimation with Scikit-learn"""

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)

    kde_skl.fit(x[:, np.newaxis])

    # score_samples() returns the log-likelihood of the samples

    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])

    return np.exp(log_pdf)
engagement = np.loadtxt('../input/engagement.csv')



plt.hist(engagement, bins=100, alpha=0.8, density=True)



x_grid = np.linspace(-1, 1, 100)

pdf_engagement = kde_sklearn(engagement, x_grid, bandwidth=0.007)



plt.plot(x_grid, pdf_engagement, alpha=0.8, lw=2, color='r')



plt.xlabel("engagement")

plt.ylabel("density")



plt.xlim(0, 1)



plt.show()
mean = np.mean(engagement)

std = np.std(engagement)



print("""

Population mean: %.5f

Population std: %.5f

Population size: %i

"""%(mean, std, len(engagement)))
sample_size = 300

n_trials = 1000000



# draw one million samples, each of size 300

samples = np.array([np.random.choice(engagement, sample_size) 

                    for _ in range(n_trials)])
# calculate sample mean for each sample

means = samples.mean(axis=1)



# mean of sampling distribution

sample_mean = np.mean(means)



# empirical standard error

sample_std = np.std(means)



analytical_std = std / np.sqrt(sample_size)



print("""

sampling distribution mean: %.5f

sampling distribution std: %.5f

analytical std: %.5f

"""%(sample_mean, sample_std, analytical_std))
from scipy.stats import t



# sampling distribution

z = 1.96

plt.hist(means, bins=50, alpha=0.9, density=True)

plt.axvline(sample_mean - 1.96 * sample_std, color='r')

plt.axvline(sample_mean + 1.96 * sample_std, color='r')



plt.xlabel('sample_mean')

plt.ylabel('density')



plt.show()
print("lower tail: %.2f%%"%(100 * sum(means < sample_mean - 1.96 * sample_std) / len(means)))

print("upper tail: %.2f%%"%(100 * sum(means > sample_mean + 1.96 * sample_std) / len(means)))
import pylab 

import scipy.stats as stats
stats.probplot(means, dist="norm", plot=pylab)

pylab.show()
# make 95% confidence interval

z = 1.96



se = samples.std(axis=1) / np.sqrt(sample_size)

ups = means + z * se

los = means - z * se

success = np.mean((mean >= los) & (mean <= ups))

fpr = np.mean((mean < los) | (mean > ups))

print("False positive rate: %.3f"%fpr)
n_points = 8000



# plt.figure(figsize=(14, 6))

plt.scatter(list(range(len(ups[:n_points]))), ups[:n_points], alpha=0.3)

plt.scatter(list(range(len(los[:n_points]))), los[:n_points], alpha=0.3)

plt.axhline(y=0.07727)



plt.xlabel("sample")

plt.ylabel("sample_mean")