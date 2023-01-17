import os, sys

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/canudos_data.csv')

dataset.head()
dataset_1 = dataset['1']

dataset_2 = dataset['2']

dataset_3 = dataset['3']



dataset_1.head()
plt.plot(dataset_1)

plt.plot(dataset_2)

plt.plot(dataset_3)

plt.show()
a = []

for i in list(dataset_1):

    a.append(float(i))



b = []

for i in list(dataset_2):

    b.append(float(i))



c = []

for i in list(dataset_3):

    c.append(float(i))



sns.set(style="darkgrid")



sns.jointplot(range(40), a, kind="reg", color="m")

sns.jointplot(range(40), b, kind="reg", color="m")

sns.jointplot(range(40), c, kind="reg", color="m")
bp_a = sns.boxplot(data=dataset_1)
bp_b = sns.boxplot(data=dataset_2)
bp_c = sns.boxplot(data=dataset_3)