%matplotlib inline
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



data_path = '../input/Video_Games_Sales_as_at_22_Dec_2016.csv'
# Read in the table with headers

df = pd.read_csv(data_path, header=0)

# First, get rid of games missing ratings

dfsub = df[df.Rating.notnull()]

dfsub.head()
dfsub.groupby(df['Rating']).count()
dfsub[dfsub['Rating'].isin(['AO','EC','K-A','RP'])]
# Use boolean mask, but use logical_not to invert it

mask = np.logical_not(dfsub['Rating'].isin(['AO','EC','K-A', 'RP']))

# filter out those unwanted games

dfsub = dfsub[mask]

# sanity check

dfsub['Rating'].unique()
# group by rating and look at the total Sales numbers

df_group = dfsub[['NA_Sales','EU_Sales','JP_Sales','Global_Sales','Other_Sales']].groupby(dfsub['Rating']).sum()

df_group
f, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))

sns.countplot(x="Rating", data=dfsub, palette="Greens_d", ax=ax1)

ax1.set_ylabel("Number of Games")

sns.barplot(x="Rating", y="Global_Sales",data=dfsub, estimator=sum, ax=ax2, ci=None)

ax2.set_ylabel("Total Global Sales (millions)")
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col')

sns.barplot(x="Rating", y="Global_Sales", data=dfsub, ax=ax1)

ax1.set_ylabel("Global Sales")

sns.barplot(x="Rating", y="NA_Sales", data=dfsub, ax=ax2)

ax2.set_ylabel("NA Sales")

sns.barplot(x="Rating", y="EU_Sales", data=dfsub, ax=ax3)

ax3.set_ylabel("EU Sales")

sns.barplot(x="Rating", y="JP_Sales", data=dfsub, ax=ax4)

ax4.set_ylabel("JP Sales")
fig, axs = plt.subplots(1,4)

ratings = ['E', 'E10+', 'T', 'M']



for rating, plot in zip(ratings, range(4)):

    scores = dfsub['Global_Sales'][dfsub['Rating'] == rating]

    ax = scores.plot.hist(bins=2000, ax=axs[plot], figsize=(10,2.5), title="%s Sales" % rating)

    ax.set_xscale('log')
E_sales = dfsub['Global_Sales'][dfsub['Rating'] == 'E']

E10_sales = dfsub.Global_Sales[dfsub['Rating'] == 'E10+']

M_sales = dfsub.Global_Sales[dfsub['Rating'] == 'M']

T_sales = dfsub.Global_Sales[dfsub['Rating'] == 'T']
## Write these values to a file for external analysis

#E_sales.to_csv(path='data/E.csv')

#E10_sales.to_csv(path='data/E10.csv')

#M_sales.to_csv(path='data/M.csv')

#T_sales.to_csv(path='data/T.csv')
sns.boxplot(data=dfsub, x="Rating", y="Critic_Score")
sns.boxplot(data=dfsub, x="Platform", y="Critic_Score")
dfsub.query("Critic_Score < 20").Name
dfsub = df[df.Year_of_Release.notnull()]

sns.barplot(x="Year_of_Release", y="Critic_Score", data=dfsub)

plt.xticks(rotation=75)