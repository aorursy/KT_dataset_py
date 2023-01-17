# Python uses packages to simplify data manipulation and analysis. Three of the most important packages

# are called Numpy, Pandas, and Matplotlib

import numpy as np # linear algebra and data processing

import pandas as pd # data processing and file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for ploting data

import seaborn as sns # makes even nicer plots (might use, don't know)
wine_data = pd.read_csv("../input/winemag-data-130k-v2.csv")

print(wine_data.head())
import collections # so I can use a handly method called Counter()

variety_counts = collections.Counter(wine_data["variety"].astype('str'))

variety_counts = dict(variety_counts.most_common(7)) # only look at the seven most common varieties

varieties = variety_counts.keys()

counts = variety_counts.values()

plt.bar(varieties, counts)

plt.xticks(rotation=90)

plt.ylabel('Frequency')

plt.show()
# how many wines are scored and how many uniquely different scores are possible in this data set?

scores = wine_data["points"]

print("A total of {} wines are scored in the data set.".format(len(scores)))

print("The possible scores are:")

print(np.unique(scores))
# since there are 21 possible point scores lets make a histogram with 21 bins

plt.hist(scores,bins=21,color='b',alpha=1) # could be more specific with bin edges but this is good enough

plt.xlabel('Point Score')

plt.ylabel('Frequency')

plt.show()
# boxplot shows 1st (25%), 2nd (50%, median), and 3rd (75%) quartiles

# shows min/max and/or 1.5xIQR 

# may also show outliers

plt.boxplot(scores, vert=False)

plt.xlabel('Point Score')

plt.show()
scores_sample = scores.sample(5000)

plt.hist(scores_sample,bins=20,color='b',alpha=1) # could be more specific with bin edges but this is good enough

plt.xlabel('Point Score')

plt.ylabel('Frequency')

plt.show()
gaussian_data = np.random.normal(loc=0, scale=1, size=10000)

plt.hist(gaussian_data,bins=50,color='b',alpha=1)

plt.ylabel('Frequency')

plt.show()
prices_idx = wine_data.index[wine_data["price"] >= 0] # get rid of wines without a specified price

prices = wine_data.loc[prices_idx,"price"]



fig = plt.figure(0, [10,4])

(ax1, ax2) = fig.subplots(1,2)

ax1.hist(prices,100,color="b")

ax1.set_ylabel('Frequency')

ax1.set_xlabel('Price ($)')

ax2.hist(prices,100,log=True,color="b")

ax2.set_ylabel('Frequency')

ax2.set_xlabel('Price ($)')

plt.show()
def get_sample_means(data,n_samples,sample_size):

    sample_means = []

    for ii in range(n_samples):

        sample = data.sample(sample_size,replace=True)

        sample_means.append(sample.mean())

    return sample_means



fig = plt.figure(0, [15,4])

(ax1, ax2, ax3) = fig.subplots(1,3)

ax1.hist(get_sample_means(prices,1000,10),50,color="b")

ax1.set_ylabel('Frequency')

ax1.set_xlabel('Average Price ($)')

ax2.hist(get_sample_means(prices,1000,100),50,color="b")

ax2.set_ylabel('Frequency')

ax2.set_xlabel('Average Price ($)')

ax3.hist(get_sample_means(prices,1000,1000),50,color="b")

ax3.set_ylabel('Frequency')

ax3.set_xlabel('Average Price ($)')

plt.show()