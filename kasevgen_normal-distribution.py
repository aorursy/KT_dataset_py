import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



from scipy.stats import norm

from scipy import stats



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



!pip install scikit-gof

# дополнительно

from statsmodels.stats import stattools

from scipy.stats import norm, uniform

from skgof import ks_test, cvm_test, ad_test

from statsmodels.graphics.gofplots import qqplot
mu = 0

sigma = 10

N = 100000

gamma = 0.99

form = r'$f(x\mid \mu ,\sigma ^{2}) = {\frac {1}{\sqrt{2\pi\sigma^2} }} e^{-{\frac {(x - \mu)^2}{2\sigma^2}}}\n$'



normal = np.random.normal(mu, sigma, size=N)



num_bins = 50

plt.subplots(figsize=(11, 8))

n, bins, _ = plt.hist(normal, num_bins, range=(mu - 5 * sigma, mu + 5 * sigma), 

                      density=1, edgecolor='k', alpha=.6, 

                      label=r"$\hat \mu=%.4f$" % normal.mean() + "\n" + r"$\hat \sigma=%.4f$" % normal.std())



a, b = norm.interval(alpha=gamma, loc=mu, scale=sigma)

px = np.arange(a, b, 0.01)

plt.fill_between(px, norm.pdf(px, mu, sigma), color='gold')



x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 100)



plt.plot(x, norm.pdf(x, mu, sigma), 'darkblue', label=r"$\mu=%.2f$" % mu + "\n" + r"$\sigma=%.2f$" % sigma)





plt.xlabel('Smarts', fontsize=14)

plt.ylabel('Probability', fontsize=14)

plt.title('Histogram normal distribution, count of bins = %d' % num_bins, fontsize=14)



plt.legend(loc='best', fontsize=16)



plt.grid(color='k', linewidth=1, linestyle='--')
alpha = 0.05
stat, pvalue = stats.shapiro(normal)

stat, pvalue, f"Гипотеза принадлежности нормальному распределению {'не' if pvalue > 0.05 else ''} отвергается"
params = (normal.mean(), normal.std())



stat, pvalue = stats.kstest(normal, 'norm', args=params)

stat, pvalue, f"Гипотеза принадлежности нормальному распределению {'не' if pvalue > 0.05 else ''} отвергается"
normal2 = normal - normal.mean()

normal2 /= normal2.std()



stats.kstest(normal2, 'norm')
ks_test(normal, norm(normal.mean(), normal.std()))
stat, crit_values, percents = stats.anderson(normal, 'norm')



print(f"Статистика: {stat}")

for i, percent in enumerate(percents):

    print(f"{percent}%: Значение статистики {'>' if stat > crit_values[i] else '<'} критического значения = {crit_values[i]}")
i = list(percents).index(alpha * 100)

f"Т.к. {stat} {'>' if stat > crit_values[i] else '<'} {crit_values[i]}: H_0 может быть {'отклонена' if stat > crit_values[i] else 'принята'}"
ad_test(normal, norm(normal.mean(), normal.std()))
qqplot(normal, line='s')

plt.show()
stat, pvalue = stats.normaltest(normal)

stat, pvalue, f"Гипотеза принадлежности нормальному распределению {'не' if pvalue > 0.05 else ''} отвергается"
stattools.omni_normtest(normal)
stat, pvalue, skew, kurtosis = stattools.jarque_bera(normal)

stat, pvalue, f"Гипотеза принадлежности нормальному распределению {'не' if pvalue > 0.05 else ''} отвергается"
cvm_test(normal, norm(normal.mean(), normal.std()))