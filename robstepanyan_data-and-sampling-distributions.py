import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from scipy.stats import sem, t

import seaborn as sns

import scipy.stats as stats

import statsmodels.api as sm



import os

print(os.listdir("../input"))
loans_income = np.array(pd.read_csv("../input/loans-income/loans_income.csv"))

loans_income[:5]
# Making a flat list from list of lists

loans_income = np.array([item for sublist in loans_income for item in sublist])
def bootstrap(l,R):

    n = len(loans_income)

    # Number of Bootstrap Samples

    means_of_boot_samples = []

    for reps in range(R):

        #Steps 1,2

        boot_sample = np.random.choice(loans_income, size = n)

        #Step 3

        means_of_boot_samples.append(round(np.mean(boot_sample), 3))

    return means_of_boot_samples



bootstrap(loans_income, 5)
np.std(bootstrap(loans_income, 100))
plt.figure(dpi = 200)



plt.subplot(221)

plt.title("R = 10.000")

plt.hist(bootstrap(loans_income, 10000), edgecolor = 'k')



plt.subplot(222)

plt.title("R = 1000")

plt.hist(bootstrap(loans_income, 1000), edgecolor = 'k')



plt.subplot(223)

plt.title("R = 100")

plt.hist(bootstrap(loans_income, 100), edgecolor = 'k')



plt.subplot(224)

plt.title("R = 10")

plt.hist(bootstrap(loans_income, 10), edgecolor = 'k')



plt.tight_layout()
data = bootstrap(loans_income, 1000)

lower_lim, upper_lim = np.percentile(data, 2.5), np.percentile(data, 95)

print("Lower Limit: ", lower_lim)

print("Upper Limit: ", upper_lim)
plt.figure(dpi = 200)

plt.title("95% Confidence interval of loan aplicants based on a sample of 1000 means")



sns.distplot(bootstrap(loans_income, 1000), hist=True, kde=True, 

             color = 'darkblue', bins = 50,

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 2})



plt.axvline(x=lower_lim,color='red')

plt.axvline(x=upper_lim,color='red')
#  Genreating normal distribution (0 is the mean, 0.1 is the std, 100 is the quanity of values)

norm = np.random.normal(0,0.1,100)



# Plotting

fig, ax = plt.subplots(dpi = 300)

probplot = sm.ProbPlot(norm)

# We use semicolon ";" here to avoid plotting the same thing twice.

probplot.qqplot(line = 's', xlabel = "", ylabel = "", ax=ax);
# Importing data

nflx = np.array(pd.read_csv("../input/stock-price/sp500_data.csv")["NFLX"])



# Plotting

fig, ax = plt.subplots(dpi = 300)

probplot = sm.ProbPlot(nflx)

# We use semicolon ";" here to avoid plotting the same thing twice.

probplot.qqplot(line = 's', xlabel = "", ylabel = "", ax=ax);
# n = 200, p = 0.02, x = ?

print("x(number of sucesses): ", np.random.binomial(200, 0.02))
np.random.poisson(2, 100)
np.random.exponential(.2, 100)
np.random.weibull(1.5, 100)