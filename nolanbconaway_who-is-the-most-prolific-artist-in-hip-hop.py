import sqlite3

import pandas as pd

import numpy as np

import matplotlib 

import matplotlib.pyplot as plt



#set matplotlib rc

matplotlib.rcParams.update({'font.size': 14})



# get the data...

con = sqlite3.connect('../input/database.sqlite')

torrents = pd.read_sql_query('SELECT * from torrents;', con)

con.close()



# only consider releases containing 'new' material

torrents = torrents[torrents.releaseType.isin(['single','ep','album','mixtape'])]

torrents = torrents[~torrents.artist.isin(['various artists'])]



# define a function to plot bar graphs

def plotbars(h, data, topn = 15):

    xticks = np.arange(topn)

    vals = data.sort_values(ascending = False)[:topn]

    

    # plotting

    h.bar(xticks-0.5, vals, width = 1.0, color = 'red')

    h.set_xticks([])

    h.axis([-0.5, topn-0.5, 0, max(vals) + max(vals)*0.05])

    for j in range(topn):  

        h.text(xticks[j], 0, vals.index[j] + ' ', 

            ha = 'right', va = 'top', rotation = 60)
# group by artist

grouping = torrents.groupby('artist')



# count overall number of releases

counts = grouping.size()



# get years of activity

years = grouping.groupYear.agg(['min', 'max'])

ranges = years['max'] - years['min'] + 1



# get number of releases per year

releases_per_year = counts / ranges



# plotting

f, ax = plt.subplots(1, 2, figsize=(15, 5))

plotbars(ax[0], counts)

ax[0].set_title('Number of Releases')

plotbars(ax[1], releases_per_year)

ax[1].set_title('Releases Per Year')



plt.show()