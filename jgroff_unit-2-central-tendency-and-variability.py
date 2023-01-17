# Python uses packages to simplify data manipulation and analysis. Three of the most important packages

# are called Numpy, Pandas, and Matplotlib

import numpy as np # linear algebra and data processing

import pandas as pd # data processing and file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for ploting data

import seaborn as sns # makes even nicer plots (might use, don't know)
wine_data = pd.read_csv("../input/winemag-data-130k-v2.csv")
import scipy.stats as stats # some useful stuff



x = np.array([2, 5, 3, 8, 6, 3, 4, 1, 9, 4, 3])

# find the mean

sum_x = x.sum()

length_x = len(x)

print("sum: ", sum_x)

print("count: ", length_x)

print("mean: ", sum_x/length_x)

print("mean: ", x.mean())



idx_middle = length_x//2

x_sorted = np.sort(x)

print("sorted x: ", x_sorted)

print("index of middle: ", idx_middle)

print("median: ", x_sorted[idx_middle])

print("median: ", np.median(x).astype(int))



from collections import Counter

x_counts = Counter(x)

print("mode: ", x_counts.most_common(1)[0][0])

print("mode: ", stats.mode(x).mode[0])
prices = wine_data.loc[wine_data.index[wine_data['price']>0], "price"] # doing some filtering to remove NaN

scores = wine_data["points"]



# these are pandas data series so it is easy to get mean, median, and mode using built in methods

print("price data stats:")

print("mean: ", prices.mean())

print("median: ", prices.median())

print("mode: ", prices.mode())

print("point score data stats:")

print("mean: ", scores.mean())

print("median: ", scores.median())

print("mode: ", scores.mode())
print("variance of point scores: ", scores.var(ddof=0)) # normalizing by N-0 (just N) since this is population parameter

print("standard deviation of point scores: ", scores.std(ddof=0)) # by default N-1 would be used

print("variance of prices: ", prices.var(ddof=0)) 

print("standard deviation of prices: ", prices.std(ddof=0))
z = (21-18)/1.5 

print(z)
import matplotlib.mlab as mlab

mu = 0

sigma = 1

x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)

plt.plot(x,mlab.normpdf(x, mu, sigma))

plt.xlabel('z')

plt.ylabel('probability density')

plt.ylim([0, 0.4])

plt.plot([2,2],[0,0.4],color='r')

plt.show()
x = np.array([21,19,17,20,18,19])

x_bar = x.mean()

N = len(x)

SE = 1.5/np.sqrt(N)

print("SE = ", SE)

print("x_bar = ", x_bar, " +/- ", SE)
x_bar_low = x_bar - 1.96*SE

x_bar_high = x_bar + 1.96*SE

print("x_bar = ", x_bar, ", 95% CI [", x_bar_low, ", ", x_bar_high, "]")