import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('whitegrid')

%matplotlib inline
df = pd.read_csv('/kaggle/input/spacex-missions/database.csv')
df.head()
df.shape
df.info()
df.describe()
plt.figure(figsize=(18,6))

sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)

plt.title('Missing Values')
def main():

    for each in df.columns:

        plt.figure(figsize=(18,6))

        chart = sns.countplot(df[each])

        chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='center',fontweight='light')

        plt.title(each)

        yield



col = main()
next(col)
next(col)
next(col)
next(col)
next(col)
next(col)
next(col)
next(col)
next(col)
next(col)
next(col)
next(col)
next(col)
next(col)
next(col)
next(col)