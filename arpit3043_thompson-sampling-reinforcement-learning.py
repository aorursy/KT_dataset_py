import os

import random

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Importing the dataset

dataset = pd.read_csv('/kaggle/input/ten-ads-optimization/Ads_CTR_Optimisation.csv')
dataset.head()
dataset.tail()
dataset.describe()
dataset.corr()
sns.scatterplot(x="Ad 1", y="Ad 2",data=dataset)
sns.jointplot(x="Ad 3", y="Ad 4",data=dataset)
sns.relplot(x="Ad 5", y="Ad 6",data=dataset)
sns.jointplot(x="Ad 7", y="Ad 8",data=dataset)
sns.jointplot(x="Ad 9", y="Ad 10",data=dataset)
N = 10000

d = 10

ads_selected = []

numbers_of_rewards_1 = [0] * d

numbers_of_rewards_0 = [0] * d

total_reward = 0

for n in range(0, N):

    ad = 0

    max_random = 0

    for i in range(0, d):

        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)

        if random_beta > max_random:

            max_random = random_beta

            ad = i

    ads_selected.append(ad)

    reward = dataset.values[n, ad]

    if reward == 1:

        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1

    else:

        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1

    total_reward = total_reward + reward
plt.hist(ads_selected)

plt.title('Histogram of ads selections')

plt.xlabel('Ads')

plt.ylabel('Number of times each ad was selected')

plt.show()