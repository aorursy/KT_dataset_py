# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt 

from  matplotlib.ticker import PercentFormatter

import numpy as np



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/honeyproduction.csv')
df.head(10)
df['per_10k_col']=(df['numcol']/10000)
df.describe()
a = df.groupby('year')

yr2012 = a.get_group(2012)

total = a['totalprod'].sum()

print (total)
df['percent_prod'] = (df['totalprod']/140907000)*100

yr2012['percent_prod'] = (yr2012['totalprod']/140907000)*100

yr2012['percent_prod'].sum() #check to make sure sum is actually 100%
plot = sns.barplot(x='state', y = 'per_10k_col', data=yr2012)

plt.xticks(rotation=90)

plt.xlabel('State')

plt.ylabel('Number of colonies (per 10k colonies)')

plt.title('2012: Number of colonies per state')
plot = sns.barplot(x='state', y = 'percent_prod', data=yr2012)

plt.xticks(rotation=90)

plt.xlabel('State')

plt.ylabel('Percent of domestic production')

plt.title('2012: Domestic production of honey')
plot = sns.barplot(x='state', y = 'yieldpercol', data=yr2012)

plt.xticks(rotation=90)

plt.xlabel('State')

plt.ylabel('Yield per colony')

plt.title('2012: Yield per colony')
sns.set_style("whitegrid")

g=sns.catplot(x="state",y="yieldpercol", hue="year", kind="swarm", data=df).set_xticklabels(rotation=90)

(g.fig.suptitle('Colony yield: USA'))
sns.set_style("whitegrid")

g=sns.catplot(x="state",y="priceperlb", hue='year', kind='point', data=df).set_xticklabels(rotation=90)

(g.fig.suptitle('Price per lb: 1998-2012'))
years = [1998, 1999, 2000, 2001, 2002]

first5 = df[df.year.isin(years)]

first5.head()

first5.tail()
plot = sns.catplot(x='state', y='percent_prod', col='year', kind='swarm', data=first5)

plot.set_xticklabels(rotation=90)
sns.set_style("whitegrid")

sns.set_palette("summer")

g=sns.catplot(x="state",y="yieldpercol", hue="year",dodge=1, kind="swarm", data=first5).set_xticklabels(rotation=90)

(g.fig.suptitle('Colony yield: USA'))
sns.set_style("whitegrid")

g=sns.catplot(x="state",y="priceperlb", hue='year', kind='point', data=first5).set_xticklabels(rotation=90)

(g.fig.suptitle('Price per lb: 1998-2002'))
nextyears = [2003, 2004, 2005, 2006, 2007]

second5 = df[df.year.isin(nextyears)]

second5.tail()
sns.set_style("whitegrid")

sns.set_palette("winter")

g=sns.catplot(x="state",y="priceperlb", hue='year', kind='point', data=second5).set_xticklabels(rotation=90)

(g.fig.suptitle('Price per lb: 2003-2007'))
lastyears = [2008, 2009, 2010, 2011, 2012]

last5 = df[df.year.isin(lastyears)]

last5.head()
sns.set_style("whitegrid")

g=sns.catplot(x="state",y="priceperlb", hue='year', kind='point', data=last5).set_xticklabels(rotation=90)

(g.fig.suptitle('Price per lb: 2008-2012'))
newEngRegion = ['CT', 'ME', 'MA', 'NH', 'RI', 'VT']

newEng = df[df.state.isin(newEngRegion)]

newEng.tail()
sns.set_style("whitegrid")

sns.set_palette('Blues')

g=sns.catplot(x="year",y="priceperlb", hue='state', kind='point', data=newEng).set_xticklabels(rotation=90)

(g.fig.suptitle('New England Honey production'))
sns.set_style("whitegrid")

g=sns.catplot(x="year",y="numcol", hue='state', kind='point', data=newEng).set_xticklabels(rotation=90)

(g.fig.suptitle('New England: Number of honey colonies'))
sns.set_style("whitegrid")

g=sns.catplot(x="year",y="yieldpercol", hue='state', kind='point', data=newEng).set_xticklabels(rotation=90)

(g.fig.suptitle('New England: Coloney yield, 1998-2012'))
sns.set_style("whitegrid")

g=sns.catplot(x="year",y="percent_prod", hue='state', kind='point', data=newEng).set_xticklabels(rotation=90)

(g.fig.suptitle('New England: percent of total US production, 1998-2012'))
midEastRegion = ['DE', 'DC', 'MD', 'NJ', 'NY', 'PA']

midEast = df[df.state.isin(midEastRegion)]

midEast.tail()
sns.set_style("whitegrid")

sns.set_palette("hls")

g=sns.catplot(x="year",y="priceperlb", hue='state', kind='point', data=midEast).set_xticklabels(rotation=90)

(g.fig.suptitle('Mid East Region Honey production'))
sns.set_style("whitegrid")

g=sns.catplot(x="year",y="numcol", hue='state', kind='point', data=midEast).set_xticklabels(rotation=90)

(g.fig.suptitle('Mid East US: Number of honey colonies'))
sns.set_style("whitegrid")

g=sns.catplot(x="year",y="percent_prod", hue='state', kind='point', data=midEast).set_xticklabels(rotation=90)

(g.fig.suptitle('Mid East US: percent of total US production, 1998-2012'))
greatLakesRegion = ['IL', 'IN', 'MI', 'OH', 'WI']

greatLakes = df[df.state.isin(greatLakesRegion)]

greatLakes.tail()
sns.set_style("white")

sns.set_palette("PRGn")

g=sns.catplot(x="year",y="priceperlb", hue='state', kind='point', data=greatLakes).set_xticklabels(rotation=90)

(g.fig.suptitle('Great Lakes Region Honey production'))
g=sns.catplot(x="year",y="yieldpercol", hue='state', kind='point', data=greatLakes).set_xticklabels(rotation=90)

(g.fig.suptitle('Great Lakes Region: Coloney yield, 1998-2012'))
g=sns.catplot(x="year",y="priceperlb", hue='state', kind='point', data=greatLakes).set_xticklabels(rotation=90)

(g.fig.suptitle('Great Lakes Region Honey production'))
g=sns.catplot(x="year",y="numcol", hue='state', kind='point', data=midEast).set_xticklabels(rotation=90)

(g.fig.suptitle('Great Lakes US: Number of honey colonies'))
g=sns.catplot(x="year",y="yieldpercol", hue='state', kind='point', data=greatLakes).set_xticklabels(rotation=90)

(g.fig.suptitle('Great Lakes US: Coloney yield, 1998-2012'))
g=sns.catplot(x="year",y="percent_prod", hue='state', kind='point', data=greatLakes).set_xticklabels(rotation=90)

(g.fig.suptitle('Great Lakes US: percent of total US production, 1998-2012'))
plainsRegion = ['IA', 'KS', 'MN', 'MS', 'NE', 'ND', 'SD']

plains = df[df.state.isin(plainsRegion)]

plains.tail()
plains.head()
sns.set_style("white")

sns.set_palette("BrBG")

g=sns.catplot(x="year",y="priceperlb", hue='state', kind='point', data=plains).set_xticklabels(rotation=90)

(g.fig.suptitle('Plains Region Honey production'))
g=sns.catplot(x="year",y="per_10k_col", hue='state', kind='point', data=plains).set_xticklabels(rotation=90)

(g.fig.suptitle('Plains Region, USA: Number of honey colonies'))
g=sns.catplot(x="year",y="percent_prod", hue='state', kind='point', data=plains).set_xticklabels(rotation=90)

(g.fig.suptitle('Plains Region, USA: percent of total US production, 1998-2012'))
southEastRegion = ['AL', 'AK', 'FL', 'GA', 'KY', 'LA', 'MS', 'NC', 'TN', 'VA', 'WV']

southEast = df[df.state.isin(southEastRegion)]

southEast.tail()
sns.set_style("white")

sns.set_palette("Spectral")

g=sns.catplot(x="year",y="priceperlb", hue='state', kind='point', data=southEast).set_xticklabels(rotation=90)

(g.fig.suptitle('Southeast Region Honey production'))
g=sns.catplot(x="year",y="per_10k_col", hue='state', kind='point', data=southEast).set_xticklabels(rotation=90)

(g.fig.suptitle('Southeast Region, USA: Number of honey colonies'))
g=sns.catplot(x="year",y="percent_prod", hue='state', kind='point', data=southEast).set_xticklabels(rotation=90)

(g.fig.suptitle('Southeast Region, USA: percent of total US production, 1998-2012'))
rockyRegion = ['CO', 'ID', 'MT', 'UT', 'WY']

rocky = df[df.state.isin(rockyRegion)]

rocky.tail()
sns.set_style("white")

sns.set_palette("dark")

g=sns.catplot(x="year",y="priceperlb", hue='state', kind='point', data=rocky).set_xticklabels(rotation=90)

(g.fig.suptitle('Rocky Mountain Region, USA Honey production'))
g=sns.catplot(x="year",y="per_10k_col", hue='state', kind='point', data=rocky).set_xticklabels(rotation=90)

(g.fig.suptitle('Rocky Mountain Region, USA: Number of honey colonies'))
g=sns.catplot(x="year",y="percent_prod", hue='state', kind='point', data=rocky).set_xticklabels(rotation=90)

(g.fig.suptitle('Rocky Mountain Region, USA: percent of total US production, 1998-2012'))
SWRegion = ['AZ', 'NM', 'TX', 'OK']

SW = df[df.state.isin(SWRegion)]

SW.tail()
sns.set_style("white")

sns.set_palette("PuOr")

g=sns.catplot(x="year",y="priceperlb", hue='state', kind='point', data=SW).set_xticklabels(rotation=90)

(g.fig.suptitle('Southwest Region, USA Honey production'))
g=sns.catplot(x="year",y="per_10k_col", hue='state', kind='point', data=SW).set_xticklabels(rotation=90)

(g.fig.suptitle('Southwest Region, USA: Number of honey colonies'))
g=sns.catplot(x="year",y="percent_prod", hue='state', kind='point', data=SW).set_xticklabels(rotation=90)

(g.fig.suptitle('Southwest Region, USA: percent of total US production, 1998-2012'))
farWRegion = ['CA', 'AK', 'HI', 'NV', 'OR', 'WA']

farW = df[df.state.isin(farWRegion)]

farW.tail()
sns.set_style("white")

sns.set_palette("PuRd")

g=sns.catplot(x="year",y="priceperlb", hue='state', kind='point', data=farW).set_xticklabels(rotation=90)

(g.fig.suptitle('Far West region, USA Honey production'))
g=sns.catplot(x="year",y="per_10k_col", hue='state', kind='point', data=farW).set_xticklabels(rotation=90)

(g.fig.suptitle('Far West Region, USA: Number of honey colonies'))
g=sns.catplot(x="year",y="percent_prod", hue='state', kind='point', data=farW).set_xticklabels(rotation=90)

(g.fig.suptitle('Far West Region, USA: percent of total US production, 1998-2012'))