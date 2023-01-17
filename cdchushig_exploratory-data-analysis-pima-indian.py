import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from subprocess import check_output

from sklearn import preprocessing



print(check_output(["ls", "../input"]).decode("utf8"))
dataset = pd.read_csv("../input/diabetes.csv")

print(dataset.shape)

dataset.head()
dataset.describe()
dataset.hist(figsize=(10, 8))

plt.show()
# Correlation

corr = dataset[dataset.columns].corr()

sns.heatmap(corr, annot = True)

plt.show()
features = dataset.columns[:8]

for feature in features:

    print(feature, np.count_nonzero(dataset[feature]==0))