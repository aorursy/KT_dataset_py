import pandas as pd

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from fbprophet import Prophet





avocados = pd.read_csv(

    '../input/avocado.csv',

    index_col=0,

)
avocados['Date'] = pd.to_datetime(avocados['Date'])

avocados.head()
listdate  = avocados["Date"]

print(listdate)
avocados.Date.value_counts()
avocados.region.value_counts()
avocados["AveragePrice"].max()


avocados["AveragePrice"].min()
avocados.groupby('year')['Total Volume'].max()
avocados.groupby('year')['AveragePrice'].max()
avocados.groupby('region')['AveragePrice'].max()
avocados.groupby('region')['AveragePrice'].min()
# set the size of the figure

plt.figure(figsize=(16,8))

# set the title

plt.title("Distribution of the Average Price")

# plot the distribution

ax = sns.distplot(avocados["AveragePrice"])
# set the size of the figure

plt.figure(figsize=(16,8))

# set the title

plt.title("BoxPlot of AveragePrice")

# plot the boxplot

ax = sns.boxplot(avocados["AveragePrice"])
# set the size of the figure

plt.figure(figsize=(16,8))

# set the title

plt.title("Type v.s. AveragePrice")

# plot Type v.s. AveragePrice

ax = sns.boxplot(y="type", x="AveragePrice", data=avocados, palette = 'pink')
sns.set_style('white')
mask = avocados['type']=='organic'

g = sns.factorplot('AveragePrice','region',data=avocados[mask],

                   hue='year',

                   size=8,

                   aspect=0.6,

                   palette='Blues',

                   join=False,

              )
mask = avocados['type']=='conventional'

g = sns.factorplot('AveragePrice','region',data=avocados[mask],

                   hue='year',

                   size=8,

                   aspect=0.6,

                   palette='Blues',

                   join=False,

              )
order = (

    avocados[mask & (avocados['year']==2018)]

    .groupby('region')['AveragePrice']

    .mean()

    .sort_values()

    .index

)
g = sns.factorplot('AveragePrice','region',data=avocados[mask],

                   hue='year',

                   size=8,

                   aspect=0.6,

                   palette='Blues',

                   order=order,

                   join=False,

              )
regions = ['PhoenixTucson', 'Chicago']
mask = (

    avocados['region'].isin(regions)

    & (avocados['type']=='conventional')

)
avocados['Month'] = avocados['Date'].dt.month

avocados[mask].head()
g = sns.factorplot('Month','AveragePrice',data=avocados[mask],

               hue='year',

               row='region',

               aspect=2,

               palette='Blues',

              )
yr2015 = avocados.loc[avocados['year'].isin(['2015'])]

yr2015.head()
g = sns.factorplot(data=yr2015, kind='swarm', palette='magma', x='type', y='AveragePrice', hue='region')
yr2016 = avocados.loc[avocados['year'].isin(['2016'])]

yr2016.head()
g = sns.factorplot(data=yr2016, kind='swarm', palette='magma', x='type', y='AveragePrice', hue='region')
yr2017 = avocados.loc[avocados['year'].isin(['2017'])]

yr2017.head()
g = sns.factorplot(data=yr2017, kind='swarm', palette='magma', x='type', y='AveragePrice', hue='region')
yr2018 = avocados.loc[avocados['year'].isin(['2018'])]

yr2018.head()
g = sns.factorplot(data=yr2018, kind='swarm', palette='magma', x='type', y='AveragePrice', hue='region')
# plt date vs. AveragePrice

# set the size of the figure

plt.figure(figsize=(16,8))

# set the title

plt.title("Date v.s. AveragePrice")



ax = sns.tsplot(data=avocados, time="Date", unit="region",condition="type", value="AveragePrice")
totalUS = avocados.loc[avocados['region'].isin(['TotalUS'])]

totalUS.head()
g = sns.factorplot(data=totalUS, kind='box', x='year', y='AveragePrice', palette='winter')

g.fig.suptitle("Total US: Avocado prices")
g = sns.factorplot(data=totalUS, kind='swarm', x='year', y='AveragePrice', hue='type', palette='winter')

g.fig.suptitle("Total US: Average avocado price")
g = sns.factorplot(data=totalUS, kind='swarm', palette='hls', x='type', y='AveragePrice', hue='year')

g.fig.suptitle("Total US: average avocado price")
avocados.groupby('type').groups
#ทำแยกแล้ว error จึงใช้การพิมเพื่อเปลี่ยนชนิดของอโวคาโดแทน (organic,coventional)

PREDICTION_TYPE = 'organic'

avocados = avocados[avocados.type == PREDICTION_TYPE]
avocados['Date'] = pd.to_datetime(avocados['Date'])
regions = avocados.groupby(avocados.region)

print("Total regions :", len(regions))

print("-------------")

for name, group in regions:

    print(name, " : ", len(group))
PREDICTING_FOR = "TotalUS"
date_price = regions.get_group(PREDICTING_FOR)[['Date', 'AveragePrice']].reset_index(drop=True)
date_price.plot(x='Date', y='AveragePrice', kind="line")