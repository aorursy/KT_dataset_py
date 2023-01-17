#Import libraries: 



from urllib.request import urlopen

import glob

import cufflinks as cf

import pandas as pd

import numpy as np

#import pandas_bokeh

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt

from matplotlib.colors import BoundaryNorm

from matplotlib import cm

from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np

from matplotlib.ticker import MaxNLocator

import geopandas as gpd

from mpl_toolkits.basemap import Basemap

from matplotlib.patches import Polygon

from matplotlib.collections import PatchCollection
# The data could be downloaded here: 

data = pd.read_csv("https://opendata.dwd.de/climate_environment/CDC/regional_averages_DE/annual/air_temperature_mean/regional_averages_tm_year.txt",sep=";", skiprows=1)

#appended_data_sorted = appended_data.sort_values(['Jahr', 'Monat'], ascending=[True, True])
# Here I will try to build up a matrix to plot

# What you have to note is that the pcolormesh usually interpolates the data between 

# 4 corner points and therefore it seems like there is 1 point on each axis missing

%matplotlib inline

y, x = np.mgrid[slice(1, data.shape[1]-3 + 1, 1),

                slice(1, data.shape[0] + 1, 1)]

z = data.iloc[:,2:-1]



fig=plt.figure(figsize=(50, 6), dpi= 80, facecolor='w', edgecolor='k')



cmap = plt.get_cmap('jet')

levels = MaxNLocator(nbins=25).tick_values(z.min().min(), z.max().max())

norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

plt.pcolormesh(x,y,np.transpose(z), cmap=cm.coolwarm, norm=norm)

plt.xticks(np.arange(1,x.shape[1]+1), data.Jahr, rotation=70, fontsize=20)

plt.yticks(np.arange(1,x.shape[0]+1), data.columns[2:-1], rotation=0, fontsize=20)

plt.title('Near Surface Annual mean Air Temperature (c) in Germany', fontsize=50)

cbaxes = fig.add_axes([.91, 0.1, 0.01, 0.8]) 

cb = plt.colorbar( cax = cbaxes)  

cb.ax.tick_params(labelsize=25)
#Just another sort of plotting :



fig = plt.figure(figsize=(54, 10))

#fig.bbox_inches.from_bounds(1, 1, 20, 6)

ax = fig.gca(projection='3d')

ax.view_init(10, 250)





# Plot the surface.

surf = ax.plot_surface(x, y, np.transpose(z), cmap=cm.coolwarm,

                       linewidth=0, antialiased=False)



ax.zaxis.set_major_locator(LinearLocator(10))

ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))



plt.xticks(np.arange(1,x.shape[1]+1), data.Jahr, rotation=70, fontsize=20)

plt.yticks(np.arange(1,x.shape[0]+1), data.columns[2:-1], rotation=10, fontsize=20)



ax.set_zlim3d(6,11)





plt.show()
# For the sake of plotting I hav downloaded the Germany's shapefiles of districts divisions

# Downloaded from http://biogeo.ucdavis.edu/data/gadm2/shp/DEU_adm.zip

fname = '../input/germanyshp/DEU_adm1.shp'

map_df = gpd.read_file(fname)
def plot_map(year_start,year_end):

    

    fig= plt.figure(figsize=(10, 13))

    ax= fig.add_subplot(111)

    m=Basemap(projection='cyl',llcrnrlat=47.3024876979,llcrnrlon=5.98865807458,

                               urcrnrlat=54.983104153,urcrnrlon=15.0169958839,resolution='l')



#m.drawmapboundary(fill_color='aqua')

#m.fillcontinents(color='w',lake_color='aqua')

#m.drawcoastlines()

    m.readshapefile('../input/germanyshp/DEU_adm1','nomoi')





    dict1 = {}

    keys = range(16)

    values = data[(data.Jahr>year_start) & (data.Jahr<year_end)].iloc[:,2:-2].mean()



    dict1={1: values['Baden-Wuerttemberg'], 2: values['Bayern'], 3:values['Brandenburg/Berlin'],

           4: values['Brandenburg/Berlin'], 5: values['Niedersachsen/Hamburg/Bremen'],

           6: values['Niedersachsen/Hamburg/Bremen'],7:values['Hessen'],

           8:values['Mecklenburg-Vorpommern'],9:values['Niedersachsen'],10:values['Nordrhein-Westfalen'],

           11:values['Rheinland-Pfalz'],12:values['Saarland'],13:values['Sachsen-Anhalt'], 

           14:values['Sachsen'],15:values['Schleswig-Holstein'],16:values['Thueringen']}

                                                   

    colvals = dict1.values()



    

    cmap = plt.get_cmap('jet')

    levels = MaxNLocator(nbins=25).tick_values(z.min().min(), z.max().max())

    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    cmap = cm.coolwarm

    patches   = []



    for info, shape in zip(m.nomoi_info, m.nomoi):

        if info['ID_1'] in list(dict1.keys()):

            color=cmap(norm(dict1[info['ID_1']]))

            patches.append( Polygon(np.array(shape), True, color=color) )



    pc = PatchCollection(patches, match_original=True, edgecolor='k', linewidths=1., zorder=2)

    ax.add_collection(pc)



    #colorbar

   

    sm = plt.cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)

    sm.set_array(colvals)

    fig.colorbar(sm, ax=ax, shrink=.5, aspect=20)

    plt.title(str(year_start)+" to "+str(year_end), fontsize=30)

    ax.axis("off")

    plt.show()

    

    

def plot_map_diff(year_start1,year_end1,year_start2,year_end2):

    

    fig= plt.figure(figsize=(10, 13))

    ax= fig.add_subplot(111)

    m=Basemap(projection='cyl',llcrnrlat=47.3024876979,llcrnrlon=5.98865807458,

                               urcrnrlat=54.983104153,urcrnrlon=15.0169958839,resolution='l')



#m.drawmapboundary(fill_color='aqua')

#m.fillcontinents(color='w',lake_color='aqua')

#m.drawcoastlines()

    m.readshapefile('../input/germanyshp/DEU_adm1','nomoi')





    dict1 = {}

    keys = range(16)

    values1 = data[(data.Jahr>year_start1) & (data.Jahr<year_end1)].iloc[:,2:-2].mean()

    values2 = data[(data.Jahr>year_start2) & (data.Jahr<year_end2)].iloc[:,2:-2].mean()

    values = values2 -values1



    dict1={1: values['Baden-Wuerttemberg'], 2: values['Bayern'], 3:values['Brandenburg/Berlin'],

           4: values['Brandenburg/Berlin'], 5: values['Niedersachsen/Hamburg/Bremen'],

           6: values['Niedersachsen/Hamburg/Bremen'],7:values['Hessen'],

           8:values['Mecklenburg-Vorpommern'],9:values['Niedersachsen'],10:values['Nordrhein-Westfalen'],

           11:values['Rheinland-Pfalz'],12:values['Saarland'],13:values['Sachsen-Anhalt'], 

           14:values['Sachsen'],15:values['Schleswig-Holstein'],16:values['Thueringen']}

                                                   

    colvals = dict1.values()



    cmap = cm.coolwarm

    cmap = plt.get_cmap('jet')

    levels = MaxNLocator(nbins=25).tick_values(-2, 2)

    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    cmap = cm.coolwarm

    patches   = []



    for info, shape in zip(m.nomoi_info, m.nomoi):

        if info['ID_1'] in list(dict1.keys()):

            color=cmap(norm(dict1[info['ID_1']]))

            patches.append( Polygon(np.array(shape), True, color=color) )



    pc = PatchCollection(patches, match_original=True, edgecolor='k', linewidths=1., zorder=2)

    ax.add_collection(pc)



    #colorbar

    

    sm = plt.cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)

    sm.set_array(colvals)

    fig.colorbar(sm, ax=ax, shrink=.5, aspect=20)

    plt.title(str(year_start2)+" to "+str(year_end2)+" - "+ 

              str(year_start1)+" to "+str(year_end1), fontsize=30)

    ax.axis("off")

    plt.show()
plot_map(1881,1910)
plot_map(1989,2018)
plot_map_diff(1881,1910,1989,2018)
plot_map(1961,1990)
plot_map_diff(1961,1990,1989,2018)
plot_map_diff(1961,1990,1881,1910)