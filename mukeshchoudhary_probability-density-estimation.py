import matplotlib.pyplot as plt

import numpy as np

from numpy.random import normal

# generate a sample

sample= normal(size= 1000)

#plot a histogram

plt.hist(sample, bins= 10)

plt.show()
plt.hist(sample, bins= 5)

plt.show()
plt.hist(sample, bins= 3)

plt.show()
sample= normal(loc= 50, scale= 5, size= 1000)
sample_mean= np.mean(sample)

sample_std= np.std(sample)
print('Mean=%.3f, Standard Deviation=%.3f' % (sample_mean, sample_std))
from scipy.stats import norm

dist = norm(sample_mean, sample_std)
values = [value for value in range(30, 70)]

probabilities = [dist.pdf(value) for value in values]
# plot the histogram and pdf

plt.hist(sample, bins=10, density=True)

plt.plot(values, probabilities)
from numpy import hstack

# generate a sample

sample1 = normal(loc=20, scale=5, size=300)

sample2 = normal(loc=40, scale=5, size=700)

sample = hstack((sample1, sample2))

# plot the histogram

plt.hist(sample, bins=50)

plt.show()
from sklearn.neighbors import KernelDensity

# fit density

model = KernelDensity(bandwidth=3, kernel='gaussian')

sample = sample.reshape((len(sample), 1))

model.fit(sample)
# sample probabilities for a range of outcomes

values = np.asarray([value for value in range(1, 60)])

values = values.reshape((len(values), 1))

probabilities = model.score_samples(values)

probabilities = np.exp(probabilities)
plt.hist(sample, bins=50, density=True)

plt.plot(values[:], probabilities)

plt.show()