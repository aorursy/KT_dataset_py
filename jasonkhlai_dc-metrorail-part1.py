#load other supporting libraries/methods

import pandas as pd

import matplotlib.pyplot as plt

from collections import defaultdict

def getFootTraffic(df):

    """

    this method compiles the foot traffic

    of each station

    Takes in a pandas df

    Returns a pandas df with station names as index

    followed by respective foot count

    """

    footCount = defaultdict(int)

    for i in df.index:

        if df.enter[i] == df.exit[i]:

                footCount[df.enter[i]] += df.avgrider[i]

        else:

            footCount[df.enter[i]] += df.avgrider[i]

            footCount[df.exit[i]] += df.avgrider[i]

    footCount = pd.DataFrame.from_dict(footCount, orient = 'index', columns = ['rider'])

    return footCount
#clean up and load datasets

df = pd.read_csv("../input/metrorail2012/May-2012-Metrorail-OD-Table-by-Time-of-Day-and-Day-of-Week .csv")

df.head()
#clean up the df

#rename the columns

df.rename(columns = {'Ent Date Holiday:No':'enter', 'Ent Date Day Type:Weekday':'exit', 'Year Month:201205':'period',

       'Unnamed: 3':'avgrider'}, inplace = True)

df = df.iloc[2:]

df = df.astype({'avgrider':'float'})

df.head()
df.dtypes
#foot traffic per station

#if entry station = exit station, don't double count

footTraffic = getFootTraffic(df)

footTraffic.head()
footTraffic.shape
#distribution of riders

footTraffic.plot.hist(bins = 15, figsize = (10,5), title = "Figure 1. Distribution of foot traffic")
footTraffic.sort_values(by = 'rider', ascending = False).plot.bar(figsize = (20,5), title = "Figure 2. Distribution of Foot traffic per station")
#there are different time periods

df.period.unique()
fig, axes = plt.subplots(nrows = 4, ncols = 1, sharex=True)

n = 3

for i in ['AM Peak', 'Midday', 'PM Peak', 'Evening']:

    ft = getFootTraffic(df[df.period == i].reset_index())

    if n == 3:

        ft.sort_values(by = 'rider', ascending = False).plot.bar(figsize = (20,5),ax = axes[n])

        stationOrder = ft.sort_values(by = 'rider', ascending = False).index

    else:

        ft.reindex(stationOrder).plot.bar(figsize = (20,5), ax = axes[n])

    axes[n].set_ylabel(i)

    n-=1

axes[0].title.set_text("Figure 3. Distribution of foot traffic per station\n(sorted on AM Peak)")
fig, axes = plt.subplots(nrows = 4, ncols = 1, sharex=True)

n = 3

for i in ['AM Peak', 'Midday', 'PM Peak', 'Evening']:

    ft = df[df.period == i].groupby("enter").sum()

    if n == 3:

        ft.sort_values(by = 'avgrider', ascending = False).plot.bar(figsize = (20,5),ax = axes[n])

        axes[n].set_xlabel("")

        stationOrder = ft.sort_values(by = 'avgrider', ascending = False).index

    else:

        ft.reindex(stationOrder).plot.bar(figsize = (20,5), ax = axes[n])

    axes[n].set_ylabel(i)

    n-=1

axes[0].title.set_text("Figure 4. Distribution of foot traffic entering a station\n(sorted on AM Peak)")
fig, axes = plt.subplots(nrows = 4, ncols = 1, sharex=True)

n = 3

for i in ['AM Peak', 'Midday', 'PM Peak', 'Evening']:

    ft = df[df.period == i].groupby("exit").sum()

    if n == 3:

        ft.sort_values(by = 'avgrider', ascending = False).plot.bar(figsize = (20,5),ax = axes[n])

        axes[n].set_xlabel("")

        stationOrder = ft.sort_values(by = 'avgrider', ascending = False).index

    else:

        ft.reindex(stationOrder).plot.bar(figsize = (20,5), ax = axes[n])

    axes[n].set_ylabel(i)

    n-=1

axes[0].title.set_text("Figure 5. Distribution of foot traffic exiting a station\n(sorted on AM Peak)")