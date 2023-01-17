from collections import Counter



import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



data = pd.read_csv('../input/chopstick-effectiveness.csv')
data.info()
data.describe()
data['Individual'].unique()
data['Chopstick.Length'].unique()
# visualize everything in a single scatterplot with a trend line

ax = sns.regplot(x='Chopstick.Length', y='Food.Pinching.Efficiency', data=data)

ax.set(xlabel='Chopstick Length (mm)', 

       ylabel='Food Pinching Efficiency',

       title='Food Pinching Efficiency vs. Chopstick Length')

plt.show()
# visualize everything while binning the data to keep the 

ax = sns.regplot(x='Chopstick.Length', y='Food.Pinching.Efficiency', data=data, x_bins=6)

ax.set(xlabel='Chopstick Length (mm)', 

       ylabel='Food Pinching Efficiency',

       title='Food Pinching Efficiency vs. Chopstick Length')

plt.show()
# obtain the optimal chopstick length for each individual

maxFPE = []

optimalLength = []



for i in range(31):

    maxFPE.append(0)

    optimalLength.append(0)



for index, row in data.iterrows():

    individual = int(row['Individual']) - 1

    if (row['Food.Pinching.Efficiency'] > maxFPE[individual]):

        maxFPE[individual] = row['Food.Pinching.Efficiency']

        optimalLength[individual] = row['Chopstick.Length']



# print(optimalLength)

optimalLengthCounts = dict(Counter(optimalLength))

# print(optimalLengthCounts)

lengths = list(optimalLengthCounts.keys())

counts = list(optimalLengthCounts.values())

ax = sns.barplot(x=lengths, y=counts)

ax.set(xlabel='Chopstick Length (mm)', 

       ylabel='Number of Best Performances',

      title='Number of Best Performances for Each Chopstick Length')

plt.show()
# similarly, obtain the worst chopstick length for each individual

minFPE = []

worstLength = []



for i in range(31):

    minFPE.append(1000)

    worstLength.append(0)



for index, row in data.iterrows():

    individual = int(row['Individual']) - 1

    if (row['Food.Pinching.Efficiency'] < minFPE[individual]):

        minFPE[individual] = row['Food.Pinching.Efficiency']

        worstLength[individual] = row['Chopstick.Length']



# print(worstLength)

worstLengthCounts = dict(Counter(worstLength))

# print(worstLengthCounts)

lengths = list(worstLengthCounts.keys())

counts = list(worstLengthCounts.values())

ax = sns.barplot(x=lengths, y=counts)

ax.set(xlabel='Chopstick Length (mm)', 

       ylabel='Number of Worst Performances',

       title='Number of Worst Performances for Each Chopstick Length')

plt.show()