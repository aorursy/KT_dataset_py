import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.mlab as mlab

import matplotlib.pyplot as plt

import pylab as pl
opioids_df = pd.read_csv("../input/opioids.csv")
p_df = pd.read_csv("../input/prescriber-info.csv")
df = pd.read_csv("../input/overdoses.csv", sep='"', thousands = ',')

df = df[['Population', 'Deaths']]

# Calculate the death rate per 100,000

df['rate'] = (df['Deaths'] / df['Population'])*100000

rate_data = df['rate']

print(df['rate'].describe())

data = df.sort_values(['Population'])
bins = np.linspace(0, 40, 10)

bins2 = np.linspace(0, 40, 100)
# Calculate the statistics from the data

mu = np.mean(rate_data)

var = np.var(rate_data)

sigma = np.sqrt(var)

# Plot the histogram and normal distribution

pl.hist(rate_data,bins,normed = 'true',color = 'blue')

pl.plot(bins2,pl.normpdf(bins2,mu,sigma), color = 'red')

pl.title('Distribution of Death Rates')

pl.xlabel('Value')

pl.ylabel('Probability')

pl.grid()
x = list(data['Population'])

y = list(data['Deaths'])

width = 7

plt.scatter(x, y, width, color="b")

plt.grid()

plt.title("Death Rate by State")

plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))