# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

import scipy.stats as stats

import matplotlib.pyplot as plt

import numpy as np



%matplotlib inline



# Any results you write to the current directory are saved as output.
n = 100   # number of samples



tdist = stats.t(df=n-1)



t = np.linspace(-5, 5, 501)

pdf = tdist.pdf(t)

plt.plot(t, pdf)
mu = 100

n = 50

xbar = 99.1

s = 4.5



t_stat = (xbar - mu)/(s/np.sqrt(n))

print('t_stat = ', t_stat)



tdist = stats.t(df=n-1)

prob_lower = tdist.cdf(t_stat)

prob_higher = 1 - tdist.cdf(t_stat)

prob_extreme = tdist.cdf(abs(t_stat)) - tdist.cdf(-abs(t_stat))

print('prob_lower = ', prob_lower)

print('prob_higher = ', prob_higher)

print('prob_extreme = ', prob_extreme)



t = np.linspace(-5, 5, 501)

pdf = tdist.pdf(t)

plt.plot(t, pdf)

plt.axvline(t_stat, c='r')

plt.axvline(-t_stat, c='r', linestyle='--')


tdist = stats.t(df=n-1)

alpha = 0.05

t95 = tdist.ppf(1 - alpha/2)



# Test it out

prob = tdist.cdf(t95) - tdist.cdf(-t95)

print('t95 = ', t95)

print('prob = ', prob)
tdist = stats.t(df=n-1)

alpha = 0.05

t95 = tdist.ppf(1 - alpha/2)



CI_lower = xbar - t95*s/np.sqrt(n)

CI_upper = xbar + t95*s/np.sqrt(n)



print('The confidence interval for xbar is {} to {}'.format(CI_lower, CI_upper))
samples = [26, 31, 23, 22, 11, 22, 14, 31]



n = len(samples)

xbar = np.mean(samples)

s = np.std(samples, ddof=1)



tdist = stats.t(df=n-1)

t95 = tdist.ppf(1 - 0.05/2)



CI_lower = xbar - t95*s/np.sqrt(n)

CI_upper = xbar + t95*s/np.sqrt(n)



print('The 95% confidence interval for vitamin C is {} to {}'.format(CI_lower, CI_upper))


t_stat = (xbar - 40)/(s/np.sqrt(n))



t = np.linspace(-9, 9, 501)

pdf = tdist.pdf(t)

plt.plot(t, pdf)

plt.axvline(t_stat, c='r')

plt.axvline(-t_stat, c='r')



p_value_upper_tail = 1 - tdist.cdf(abs(t_stat))

p_value_lower_tail = tdist.cdf(-abs(t_stat))

p_value = p_value_upper_tail + p_value_lower_tail

p_value
n = 5

mu = 32

xbar = 33.5

s = 2



t_stat = (xbar - mu)/(s/np.sqrt(n))



tdist = stats.t(df=n-1)



t = np.linspace(-9, 9, 501)

pdf = tdist.pdf(t)

plt.plot(t, pdf)

plt.axvline(t_stat, c='r')

plt.axvline(-t_stat, c='r')



p_value_upper_tail = 1 - tdist.cdf(abs(t_stat))

p_value_lower_tail = tdist.cdf(-abs(t_stat))

p_value = p_value_upper_tail + p_value_lower_tail

print('p_value = ', p_value)



CI_lower = xbar - t95*s/np.sqrt(n)

CI_upper = xbar + t95*s/np.sqrt(n)

print('The 95% confidence interval = {} to {}'.format(CI_lower, CI_upper))

# H0: xbar1 == xbar2 (t == 0)



n1 = 450

xbar1 = 4.6

s1 = 1.2



n2 = 523

xbar2 = 5.0

s2 = 1.4



t = (xbar1 - xbar2)/np.sqrt(s1**2/n1 + s2**2/n2)



tdist = stats.t(df=n1 + n2 - 2)

p_lower = tdist.cdf(-abs(t))

p_upper = 1 - tdist.cdf(abs(t))

p_value = p_upper + p_lower

(t, p_value)
# H0: xbar1 >= xbar2 (t >= 0)



n1 = 450

xbar1 = 4.6

s1 = 1.2



n2 = 523

xbar2 = 5.0

s2 = 1.4



t = (xbar1 - xbar2)/np.sqrt(s1**2/n1 + s2**2/n2)



tdist = stats.t(df=n1 + n2 - 2)

p_lower = tdist.cdf(-abs(t))

p_value = p_lower

(t, p_value)
k = 32

n = 59



alpha_prior = 301 + 1

beta_prior = 523 - 301 + 1



alpha_posterior = alpha_prior + k

beta_posterior = beta_prior + n - k



beta_dist = stats.beta(a=alpha_posterior, b=beta_posterior)

p = np.linspace(0, 1, 501)

pdf = beta_dist.pdf(p)

plt.plot(p, pdf)



mle = k/n

map = (alpha_posterior - 1)/(alpha_posterior + beta_posterior - 2)

mle, map


k = 310

n = 600



alpha_prior = 301 + 1

beta_prior = 523 - 301 + 1



alpha_posterior = alpha_prior + k

beta_posterior = beta_prior + n - k



beta_dist = stats.beta(a=alpha_posterior, b=beta_posterior)

p = np.linspace(0, 1, 501)

pdf = beta_dist.pdf(p)

plt.plot(p, pdf)



mle = k/n

map = (alpha_posterior - 1)/(alpha_posterior + beta_posterior - 2)

mle, map
alpha_prior = 1

beta_prior = 1



k = 11

n = 15



alpha_posterior = alpha_prior + k

beta_posterior = beta_prior + n - k



beta_dist = stats.beta(a=alpha_posterior, b=beta_posterior)

p = np.linspace(0, 1, 501)

pdf = beta_dist.pdf(p)

plt.plot(p, pdf)

plt.axvline(0.5, c='r')



p_less_than_50p = beta_dist.cdf(0.5)

p_more_than_50p = 1 - p_less_than_50p

p_less_than_50p, p_more_than_50p
mle = k/n

map = (alpha_posterior - 1)/(alpha_posterior + beta_posterior - 2)

mle, map