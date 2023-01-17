import numpy as np

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/drug-classification/drug200.csv')



df.head()
sns.countplot('Drug', data=df)

plt.show()
plt.figure(figsize=(10, 10))

sns.scatterplot(x='Age', y='Na_to_K', hue='Drug', data=df, s=100)

plt.plot(range(10, 80), [15 for i in range(10, 80)], 'b--')

plt.show()
df = df[df.Na_to_K <= 15]

df.head()
sns.countplot(x='Sex', hue='Drug', data=df)

plt.show()
sns.countplot(x='BP', hue='Drug', data=df)

plt.show()
df_low = df[df.BP == 'LOW']

df_high = df[df.BP == 'HIGH']
for i, drug in enumerate(set(df_high.Drug.values)):

    sns.kdeplot(df_high[df_high.Drug == drug].Age, shade=True, legend=False)

plt.legend(list(set(df_high.Drug.values)))

plt.show()
plt.figure(figsize=(10, 10))

sns.scatterplot(x='Age', y='Na_to_K', hue='Drug', data=df_high, s=100)

plt.plot([50 for i in range(5, 20)], range(5, 20), 'r--')

plt.show()
sns.countplot(x='Cholesterol', hue='Drug', data=df_low)

plt.show()
im = plt.imread('../input/decision-tree/Blank Diagram.png')

plt.figure(figsize=(20, 20))

plt.imshow(im, cmap='gray')

plt.show()