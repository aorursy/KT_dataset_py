import pandas as pd

import numpy as np



import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.colors import Normalize

from matplotlib.patches import Polygon

from matplotlib.collections import PatchCollection

from mpl_toolkits.basemap import Basemap



%matplotlib inline
crime = pd.read_csv("../input/SouthAfricaCrimeStats_v2.csv")
year = ['2005-2006', '2006-2007', '2007-2008', '2008-2009',

       '2009-2010', '2010-2011', '2011-2012', '2012-2013', '2013-2014',

       '2014-2015', '2015-2016']



mapping = dict(zip(year, [x.split("-")[0] for x in year]))



crime.rename(columns=mapping,inplace=True)

crime['Station'] = crime['Station'].str.upper()
crime.head()
crime['Category'].unique()
num_colors = 20

shapefile = '../input/Police_bounds'
def group(category):



    group = crime[crime['Category'] == category ][['Station','Category', '2005', '2006', '2007', '2008', '2009', 

    '2010','2011', '2012', '2013','2014', '2015']].groupby(['Station','Category'])['2005', '2006', '2007',

    '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'].sum()



    for x in group.columns:

        values = group[x].values

        cm = plt.get_cmap('Oranges')

        scheme = [cm(i / num_colors) for i in range(num_colors)]

        bins = np.linspace(values.min(), values.max(), num_colors)

        group['bin'] = np.digitize(values, bins) - 1





        fig = plt.figure(figsize=(12, 12))

        ax = fig.add_subplot(111, axisbg='w', frame_on=False)

        fig.suptitle('Count of {} in South Afrika in {} per district.'.format(category, x), fontsize=20, y=.95)

                     

        m = Basemap(llcrnrlat=-40,

                 urcrnrlat=-20,

                 llcrnrlon=10,

                 urcrnrlon=40,

                 lat_0=-25, projection='merc')

       

        m.drawcoastlines()

        m.drawcountries()

        m.readshapefile(shapefile,'units', color='#444444', linewidth=.2)



        for info, shape in zip(m.units_info, m.units):

            name = info['COMPNT_NM']

            if name not in group.index.levels[0]:

                color = '#dddddd'

            else:

                color = scheme[group[group.index.levels[0] == name]['bin'].values]



            patches = [Polygon(np.array(shape), True)]

            pc = PatchCollection(patches)

            pc.set_facecolor(color)

            ax.add_collection(pc)



        # Draw color legend.

        ax_legend = fig.add_axes([0.1, 0.3, 0.8, 0.03], zorder=3)

        cmap = mpl.colors.ListedColormap(scheme)

        cb = mpl.colorbar.ColorbarBase(ax_legend, cmap=cmap, ticks=bins, boundaries=bins, orientation='horizontal')

        cb.ax.set_xticklabels([str(round(i, 1)) for i in bins])



        plt.savefig('choropleth_{}_{}.png'.format(category,x), dpi=300, transparent=True)
crime['Category'].unique()
group('All theft not mentioned elsewhere')