import numpy as np 

import pandas as pd 

import seaborn as sns

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.patches import Polygon

from matplotlib.collections import PatchCollection

from matplotlib.patches import PathPatch

import shapefile

from subprocess import check_output



%matplotlib inline

mpl.rcParams['figure.figsize'] = (16.0, 12.0)



print(check_output(["ls", "../input"]).decode("utf8"))
# Interautonomic migration flow per year, Autonomous Community of origin and destination

data_in_out = pd.read_excel('../input/Migrations_Spain_CA_in_out_per.xls', index_col=0)
# Interautonomic (CCAA) migration for the year 2008

data_in_out_2008 = data_in_out[data_in_out['Year'] == 2008]

data_in_out_2008 = data_in_out_2008.drop('Year', 1)



# Heatmap with the total data migration of people between you distinguish regions of Spain

sns.set(font_scale=0.5)

sns.heatmap(data_in_out_2008, annot=True) 
#Percentage of people who migrated in 2008 by autonomous community.

#The percentage is that of regions (C.A.) source

data_in_per = pd.read_excel('../input/Migrations_Spain_CA_source_per.xls', index_col=0)



data_in_per_2008 = data_in_per[data_in_per['Year'] == 2008]

data_in_per_2008 = data_in_per_2008.drop('Year', 1)



# Heatmap with the total data migration of people between you distinguish regions of Spain

sns.set(font_scale=0.5)

sns.heatmap(data_in_per_2008, annot=True) 
# Function for print map of Spain by colunm

def print_map_spain(year, data, color='Blues') : 

    fig = plt.figure(figsize=(8, 6))

    colors = []

    patches = []

    values = data[year]

        

    colorVotes = plt.get_cmap(color)



    # Los nombres de las CAs ya que son exactamente iguales en los ficheros

    cas = {"ndalu":u"Andalucía", "stur":"Asturias", "rag":u"Aragón", "aleare":"Baleares","anari":"Canarias",

            "antabr":"Cantabria", "y Le":u"Castilla y León", "ancha":"Castilla - La Mancha","atalu":u"Cataluña",

            "alencia":"Comunitat Valenciana", "xtrem":"Extremadura", "alic":"Galicia", "adrid":"Madrid", "urcia":"Murcia", 

            "avarra":"Navarra", "Vasco":u"País Vasco", "ioja":"La Rioja"}



    ax = plt.gca() # get current axes instance



    # Map

    map = Basemap(llcrnrlon=-10.5,llcrnrlat=35,urcrnrlon=4.,urcrnrlat=44.,

             resolution='i', projection='tmerc', lat_0 = 39.5, lon_0 = -3.25)

    

    # Download files of http://www.gadm.org/country (http://biogeo.ucdavis.edu/data/gadm2.8/shp/ESP_adm_shp.zip)

    # load the shapefile, use the name 'states'. 

    map.readshapefile('../input/ESP_adm2', 'spain')

    map.drawmapboundary(fill_color='aqua')

    map.drawcoastlines()



    for info, shape in zip(map.spain_info, map.spain):

        for abre_ca, ca in zip(cas.keys(), cas.values()) :

            if info['NAME_1'].find(abre_ca)>=0 :            

                value = float(data[year][data['CA'] == ca])

                # Regla de 3: Max - min -> 1 : value -min -> x

                value_adj = (value - values.min()) / (values.max() - values.min())

                color = colorVotes(value_adj)    

        colors.append(color)

        poly = Polygon(np.array(shape), facecolor=color, edgecolor=color)

        patches.append(poly)                

        ax.add_patch(poly)

 

    pc = PatchCollection(patches, cmap=colorVotes)

    ax.add_collection(pc)



    ax.text(1, 0, 'Migrations CA Spain\nTwitter: @SimOn_kxk\nhttp://labitacora.wordpress.com\nJulian Simon', ha='right', 

            va='bottom', color='#555555', transform=ax.transAxes)



    # Draw color legend.

    #ax_legend = fig.add_axes([0.35, 0.17, 0.3, 0.025], zorder=3)

    ax_legend = fig.add_axes([0.28, 0.19, 0.4, 0.03], zorder=3)

    cmap = mpl.cm.cool

    norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())

    cb1 = mpl.colorbar.ColorbarBase(ax_legend, cmap=colorVotes, norm=norm, orientation='horizontal')

    cb1.set_label('Percentage %')



    plt.show()
# Percentages of population of the different Autonomous Communities of Spain that have migrated in these years

data = pd.read_excel('../input/Migrations_Spain_CA_per.xls')
print_map_spain(year='Percentage_source_2008', data=data)
# I make the same map for 2012 to know if it follows the same trend

print_map_spain('Percentage_source_2012', data)
# Now we are going to show the percentages in relation to the inhabitants of the C.A. of destiny

# Year 2008

print_map_spain('Percentage_2008', data)
print_map_spain('Percentage_2010', data)
print_map_spain('Percentage_2012', data)
print_map_spain('Percentage_2014', data)
# Differences between arrivals and people who have migrated by C.A. no longer

data_diff = pd.read_excel('../input/Migrations_Spain_CA_diff.xls')
print_map_spain('Migrations_diff', data_diff[ data_diff ['Year'] == 2008], color='Greens')
sns.plt.title('People who migrate minus people who arrive')

sns.barplot(x="Year", y="Migrations_diff", hue="CA", data=data_diff)
# The percentage differences between arrivals and people who have migrated by C.A. no longer

sns.plt.title('% people who migrate minus people who arrive') 

sns.barplot(x="Year", y="Percentage_diff", hue="CA", data=data_diff)