import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



sns.set(style='darkgrid', palette='Set1')
df = pd.read_csv('../input/archive.csv')
df.head()
df.dropna(inplace=True)
sns.distplot(df['February Average Temperature'], kde=False, bins=20)

plt.title("February Average Temperature Distribution")

plt.show()
sns.distplot(df['March Average Temperature'], kde=False, bins=20)

plt.title("March Average Temperature Distribution")

plt.show()