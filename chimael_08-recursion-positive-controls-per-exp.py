import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))
train_controls = pd.read_csv('../input/train_controls.csv')

test_controls = pd.read_csv('../input/test_controls.csv')
d = {}

for i,exp in enumerate(train_controls.experiment.unique()):

    d[exp] = len(train_controls[train_controls.experiment == exp].sirna.unique())

d
d = {}

for i,exp in enumerate(test_controls.experiment.unique()):

    d[exp] = len(test_controls[test_controls.experiment == exp].sirna.unique())

d
siRNA = train_controls.sirna.unique()

siRNA
test_siRNA = test_controls.sirna.unique()

sorted(test_siRNA) == sorted(siRNA)
train_exp = train_controls.experiment.unique()

d = {}

for exp in train_exp:

    siRNA_per_exp = []

    siRNA_per_exp.extend([len(train_controls[(train_controls.experiment == exp) & (train_controls.sirna == i)]) for i in list(siRNA)])

    d[exp] = siRNA_per_exp
df = pd.DataFrame(d, index=siRNA)
sns.heatmap(df)

plt.title('Number of siRNA replicates per experiment [train]')

plt.show()
test_exp = test_controls.experiment.unique()

d = {}

for exp in test_exp:

    siRNA_per_exp = []

    siRNA_per_exp.extend([len(test_controls[(test_controls.experiment == exp) & (test_controls.sirna == i)]) for i in list(siRNA)])

    d[exp] = siRNA_per_exp
df = pd.DataFrame(d, index=siRNA)
sns.heatmap(df)

plt.title('Number of siRNA replicates per experiment [test]')

plt.show()