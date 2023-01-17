import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
df = pd.read_csv('../input/swing-states/2008_swing_states.csv')

df.head()
df[['state', 'county', 'dem_share']]
sns.set()

plot = plt.hist(df['dem_share'])

plot = plt.xlabel('percentage of votes for obama')

plot = plt.ylabel('number of counties')

plt.show()
plot = sns.swarmplot(x='state', y='dem_share', data=df)

plot = plt.xlabel('state')

plot = plt.ylabel('percentage of votes for obama')

plt.show()
x = np.sort(df['dem_share'])

y = np.arange(1, len(x)+1) / len(x)

plot = plt.plot(x, y, marker='.', linestyle='none')

plot = plt.ylabel('ECDF')

plot = plt.xlabel('percentage of votes for obama')

plt.margins(0.02) #keeps data off plot edges

plt.show()





#annotate
df.describe()
df.mean()

#np.mean(df['dem_share'])
df.median()
np.percentile(df['dem_share'], [25,50,75])
sns.boxplot(x='state', y='dem_share', data=df)

plt.xlabel('state')

plt.ylabel('percentage of votes for obama')

plt.show()

#annotate percentiles and outliers

#calculate if outliers
np.var(df['dem_share'])
#scatter plot

plt.plot(df['total_votes']/1000, df['dem_share'], marker='.', linestyle='none')

plt.xlabel('votes (thousands)')

plt.ylabel('percentage of votes for obama')