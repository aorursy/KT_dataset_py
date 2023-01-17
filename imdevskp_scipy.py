import matplotlib.pyplot as plt

from scipy.stats import bernoulli, binom
# Random variates of bernoulli distribution

# 1 or 0 of probability = 0.5

bernoulli.rvs(p=0.5)
# 2 samples of 1 or 0 of probability = 0.5

bernoulli.rvs(p=0.5, size=2)
# 10 samples of 1 or 0 of probability = 0.3

bernoulli.rvs(p=0.3, size=10)
sum(bernoulli.rvs(p=0.5, size=20))
# No. of success if conducted 6 times with success probability of 0.5

binom.rvs(n=6, p=0.5)
# Random Variates of binomial distribution

binom.rvs(n=5, p=0.5, size=10)
binom.rvs(n=4, p=0.8, size=7)
dist = binom.rvs(n=10, p=0.5, size=1000)

plt.hist(dist, bins=10, range=(0,11))

plt.show()
# Probability mass function at k(quantile) of the given RV
# Probability of getting 2 heads after 10 throws with a fair coin (ie. p=0.5)

binom.pmf(k=2, n=10, p=0.5)
# Probability of 5 heads after 10 throws with a fair coin

binom.pmf(k=5, n=10, p=0.5)
x = []

y = []



for i in range(0,11):

    x.append(i)

    y.append(binom.pmf(k=i, n=10, p=0.5))



print(x)

print(y)



plt.bar(x, y)

plt.show()
# Cumulative distribution function of the given RV at k (quantile)
binom.cdf(k=1, n=10, p=0.5)
binom.cdf(k=5, n=10, p=0.5)
binom.cdf(k=10, n=10, p=0.5)
# Probability of 5 heads or less after 10 throws with a fair coin

binom.cdf(k=5, n=10, p=0.5)
# Probability of 50 heads or less after 100 throws with p=0.3

binom.cdf(k=50, n=100, p=0.3)
# Probability of more than 59 heads after 100 throws with p=0.7

1-binom.cdf(k=59, n=100, p=0.7)
# Probability of more than 59 heads after 100 throws with p=0.7

binom.sf(k=59, n=100, p=0.7)
z = []



for i in range(0,11):

    z.append(binom.cdf(k=i, n=10, p=0.5))



print(x)

print(z)



plt.plot(x, z, c='red', lw=2)

plt.show()
plt.bar(x, y, label='PMF')

plt.plot(x, z, c='red', lw=2, label='CDF')

plt.legend()

plt.show()