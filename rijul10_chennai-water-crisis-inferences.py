import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
import os

print(os.listdir("../input"))
cdata = pd.read_csv("../input/chennai_reservoir_rainfall.csv")

cdata.reset_index(drop=True, inplace=True)
cdata.tail()
cdata.describe()
cdata.count()
cdata.count() - cdata[cdata == 0].count()
cdata['Year']=[d.split('-')[2] for d in cdata.Date]

cdata['Month']=[d.split('-')[1] for d in cdata.Date]

cdata['Day']=[d.split('-')[0] for d in cdata.Date]



cdata.head()
bdata = cdata.groupby('Year').sum()
bdata
from matplotlib import pyplot

sns.lineplot(sns.set(rc={'figure.figsize':(15,8)}), data = bdata)

plt.title('Total Rain fall in the reservoir regions over the years', size = 20)

plt.ylabel('MM', size = 15)

plt.xlabel('Year',size = 15)
rdata = pd.read_csv("../input/chennai_reservoir_levels.csv")
rdata.tail()
rdata.describe()
rdata['Year']=[d.split('-')[2] for d in rdata.Date]

rdata['Month']=[d.split('-')[1] for d in rdata.Date]

rdata['Day']=[d.split('-')[0] for d in rdata.Date]



rdata.head()
rbdata = rdata.groupby('Year').mean()
rbdata
sns.lineplot(data = rbdata)

plt.title('Mean reservoir water level throughout the years', size = 20)

plt.xlabel('Year', size = 15)

plt.ylabel('Mean mcft', size = 15)
sns.lineplot(data = rbdata['REDHILLS'], label = 'Reservoir Water level')

sns.lineplot(data = bdata['REDHILLS'], label = 'Rainwater Contribution')

plt.title('Mean reservoir water level vs Rainwater contribution in REDHILLS', size = 20)

plt.xlabel('Year', size = 15)

plt.ylabel('Mean mcft & MM', size = 15)
sns.lineplot(data = rbdata['POONDI'], label = 'Reservoir Water level')

sns.lineplot(data = bdata['POONDI'], label = 'Rainwater Contribution')

plt.title('Mean reservoir water level vs Rainwater contribution in POONDI', size = 20)

plt.xlabel('Year', size = 15)

plt.ylabel('Mean mcft & MM', size = 15)
sns.lineplot(data = rbdata['CHOLAVARAM'], label = 'Reservoir Water level')

sns.lineplot(data = bdata['CHOLAVARAM'], label = 'Rainwater Contribution')

plt.title('Mean reservoir water level vs Rainwater contribution in CHOLAVARAM', size = 20)

plt.xlabel('Year', size = 15)

plt.ylabel('Mean mcft & MM', size = 15)
sns.lineplot(data = rbdata['CHEMBARAMBAKKAM'], label = 'Reservoir Water level')

sns.lineplot(data = bdata['CHEMBARAMBAKKAM'], label = 'Rainwater Contribution')

plt.title('Mean reservoir water level vs Rainwater contribution in CHEMBARAMBAKKAM', size = 20)

plt.xlabel('Year', size = 15)

plt.ylabel('Mean mcft & MM', size = 15)
rbdata.plot.bar(stacked=True)

plt.ylabel('Mean reservoir water level in MCft', size = 15)

plt.xlabel('Years', size = 15)