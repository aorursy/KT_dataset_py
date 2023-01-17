import pandas as pd

import numpy as np

from scipy.stats import ks_2samp, mannwhitneyu, ttest_ind

import seaborn as sns

import matplotlib.pyplot as plt



import math

import random
data = pd.read_csv('/kaggle/input/u_testl.csv', sep=';', usecols=['values', 'groups'])
#Исправил разделители для float

data['values'] = data['values'].apply(lambda x: x.replace(',', '.')).astype(float)
x1 = data[data['groups'] == 0]['values'].values

x2 = data[data['groups'] == 1]['values'].values
plt.figure(figsize=(15, 8))

sns.distplot(x1)

sns.distplot(x2)
ks_2samp(x1, x2)
plt.figure(figsize=(15, 8))

backet_size = 50

#Перемешал выборки

random.shuffle(x1)

random.shuffle(x2)

#Разбил на бакеты

backets_x1 = [np.mean(x1[i:i + backet_size]) for i in range(0, len(x1), backet_size)]

backets_x2 = [np.mean(x2[i:i + backet_size]) for i in range(0, len(x2), backet_size)]

sns.distplot(backets_x1)

sns.distplot(backets_x2)
ttest_ind(backets_x1, backets_x2)
def bootstrap(data, nboot):

    metrics = []

    for i in range(nboot):

        idx = np.random.randint(len(data), size=len(data))

        sample = data[idx]

        metrics.append(sample.mean())

    return metrics
bootx1 = bootstrap(x1, 10000)

bootx2 = bootstrap(x2, 10000)



ci1 = np.percentile(bootx1, q=[1, 99])

ci2 = np.percentile(bootx2, q=[1, 99])
plt.vlines(ci1, color='red', ymin=0, ymax=0.35, linestyle='--')

plt.vlines(ci2, color='blue', ymin=0, ymax=0.35, linestyle='--')

sns.distplot(bootx1, color='red')

sns.distplot(bootx2, color='blue')