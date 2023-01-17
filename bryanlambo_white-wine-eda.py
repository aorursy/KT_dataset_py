import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('../input/cs178-wine-quality/winequality-white.csv',sep=';')

df.head()
df.shape
df.columns
df.info()
df.quality.unique()
df.quality.value_counts()
sns.heatmap(df.isnull(),yticklabels=False, cbar=False)

plt.title('')
plt.figure(figsize=(10,6))

sns.heatmap(df.sort_values(by='quality',ascending=True).corr(),cmap='viridis',annot=True)
cols=df.columns.values

l=len(cols)

plt.figure(figsize=(24,6))

print(cols)

for i in range(0,l):

    plt.subplot(1,l,i+1)

    sns.boxplot(df[cols[i]],orient='v',color='green')

    plt.tight_layout()
plt.figure(figsize=(24,6))

for i in range(0,l):

    plt.subplot(1,l,i+1)

    sns.distplot(df[cols[i]])