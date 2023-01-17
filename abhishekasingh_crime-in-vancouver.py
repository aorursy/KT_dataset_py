import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline

sns.set_style('whitegrid')
df = pd.read_csv('../input/crime.csv')

df['counter'] = 1
df.head()
plt.figure(figsize=(12,6))

plt.title('Types of Crime',fontdict={'fontsize':'30'},pad=20)

ax = sns.countplot(x='TYPE',data=df,palette='Blues_d', order = df['TYPE'].value_counts().index)

ax.set(xlabel='Types of Crime')

ax.set(ylabel='Counts')

plt.setp(ax.get_xticklabels(), rotation=25, horizontalalignment='right')

plt.tight_layout()
plt.figure(figsize=(12,6))

plt.title('Years Trend',fontdict={'fontsize':'30'},pad=20)

ax = sns.countplot(x='YEAR',data=df,palette='Blues_d')

ax.set(xlabel='Years', ylabel='Counts')

plt.setp(ax.get_xticklabels(), rotation=25, horizontalalignment='right')

plt.tight_layout()
df.groupby(['NEIGHBOURHOOD','TYPE']).count()['counter'].sort_values(ascending=True).head(5)
danger_region = df.groupby(['NEIGHBOURHOOD','TYPE']).count()['counter'].sort_values(ascending=False).head(10)

danger_region
df.groupby(['MONTH']).count()['counter'].sort_values(ascending=False).head(5)
df.groupby(['DAY']).count()['counter'].sort_values(ascending=False).head(5)
df.groupby(['HOUR']).count()['counter'].sort_values(ascending=False).head(5)