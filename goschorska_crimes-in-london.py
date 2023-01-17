import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap

%matplotlib notebook
plt.style.use('seaborn-white')
# Importing the London Police Records dataset:
street = pd.read_csv('../input/london-police-records/london-street.csv')

street.drop(['Crime ID', 'Context'], axis=1, inplace=True) # removing mostly-emplty columns
street.dropna(inplace=True) # removing rows with emply fields

# Filtering for data from 2016 to reduce dataset size:
street['Year'] = street['Month'].apply(lambda x: x.split('-')[0].strip())
street = street.loc[street['Year'] == '2016']

# One of the columns in street contains 'LSOA codes'.
# In the UK, the Lower Layer Super Output Areas (LSOAs) are small
# geographic areas used by the Office for National Statistics

street.head()
# Preparing the data:

# Grouping data by location (LSOA code) and by time. A single data point
# correstonds to all crimes of a given type commited in one LSOA in one month,
# e.g all bicycle thefts in LSOA E01000001 in Jun 2016.

df1 = pd.crosstab([street['LSOA code'], street['Month']], 
                  street['Crime type']) # creates a column for each crime type
df1.drop('Other crime', axis=1, inplace=True)
df1.reset_index(inplace=True)
df1.columns.name = None # simplifies plot labelling
df1.head()
# Defining functions for the interactivity of the figure.

crimes = list(df1)[2:] # List crime types

# The figure contains a heatmap and a correspondng scatter plot.
# When clicked on, a tile in the heatmap is outlined and a corresponding
# scatterplot is generated.

def plotting(xlabel, ylabel, xint, yint):
    global outline
    outline.remove() # remove tile outline from previous onclick event
    scatter_ax.cla() # clear scatterplot axis
    s = sns.scatterplot(df1[xlabel], df1[ylabel], color='navy',
                    ax=scatter_ax, alpha = 0.2, s=80) # plot scatterplot
    #update names of x and y axis:
    scatter_ax.set_xlabel('%s (cases per month per LSOA)' % (xlabel))
    scatter_ax.set_ylabel('%s (cases per month per LSOA)' % (ylabel))
    title = ('More %s, more %s?' % (xlabel, ylabel)) # update plot's title
    heatmap_ax.set_title(title, fontsize=16, horizontalalignment='center')
    # draw tile outline:
    rect = Rectangle((xint, yint),1,1,fill=False,
                             edgecolor='navy',linewidth=2.0)
    outline = heatmap_ax.add_patch(rect)

def onclick(event): # Defines what happens whene mouse button is pressed
    # xdata, ydata = x, y coordinates of mouse in data coords
    # round down the coordinates by converting float into integer,
    # to index into the list of crimes:
    xint = int(event.xdata) 
    yint = int(event.ydata)
    xlabel = crimes[xint]
    ylabel = crimes[yint]
    
    plotting(xlabel, ylabel, xint, yint)
# Compute pairwise Pearson correlation between crime types.
# A Pearson correlation is a number between -1 (negative correlation)
# and 1 (positive correlation) that indicates the extent to which
# two variables are linearly related (0 for no correlation)

corr = df1.corr()
corr.head()
# corr will be plotted as heatmap with tile colours corresponding to Pearson scores
# Data in corr dataframe is duplicated
# To remove duplicates, generate a diagonal (istead of square) correlation matirix,
# with repeated data hidden with a triangular mask:

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# The scatter plot and heatmap's colorbar overlap the heatmap.
# Hence, defining overlapping axes:
f = plt.figure(figsize=(10, 10))
heatmap_ax = f.add_axes([0.11, 0.11, 0.7, 0.9])
cbar_ax = f.add_axes([0.62, 0.42, 0.3, 0.03])
scatter_ax = f.add_axes([0.62, 0.55, 0.3, 0.3])

# Figure's and plot's titles:
f.suptitle('Do rates of different crimes correlate?',
          fontsize=20, horizontalalignment='center',
          weight='semibold')
heatmap_ax.set_title('Click on tiles to see more details',
                     fontsize=16, horizontalalignment='center')
cbar_ax.set_title('Pearson correlation scores \n(1: positive correlation, 0: no correlation)',
                  horizontalalignment='center')

# A Diverging colour palette with 40 categories for the heatmap:
cmap = sns.color_palette("RdBu_r", 40)

# For labelling the heatmap (added line breaks into longer crime names):                 
labels = ['Bicycle theft', 'Burglary', 'Criminal damage\nand arson',
          'Drugs', 'Other theft', 'Possession of\nweapons',
          'Public order', 'Robbery', 'Shoplifting', 'Theft from\nthe person',
          'Vehicle crime', 'Violence and\nsexual offences']

# Drawing the heatmap using the Seaborn (sns) visualisation library,
# a high-level interface for drawing pretty statistical graphics:
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True,
            square=True, linewidths=.5, ax=heatmap_ax,
            xticklabels=labels, yticklabels=labels,cbar_ax = cbar_ax,
            cbar_kws = dict(orientation='horizontal'))

# Drawing a scatterplot corresponding to one tile in heatmap:
s = sns.scatterplot(df1['Other theft'], df1['Shoplifting'], ax=scatter_ax,
                alpha=0.2, s=80, color='navy')
# Hiding right y axis and upper x axis
scatter_ax.spines['right'].set_visible(False)
scatter_ax.spines['top'].set_visible(False)
# Lablelling the x and y axis:
scatter_ax.set_xlabel('Burglary (cases per month per LSOA)')
scatter_ax.set_ylabel('Shoplifting (cases per month per LSOA)')

# Adding an outline to the corresponding tile in the heatmap:
rect = Rectangle((3, 7),1,1,fill=False,
                         edgecolor='navy',linewidth=2.0)
outline = heatmap_ax.add_patch(rect)
  
# Add interactivity, triggered by pressing mouse bottom:
f.canvas.mpl_connect('button_press_event', onclick)

plt.show()
# Limiting the dataframe to cases of bike theft:
bikes = street[['Longitude','Latitude','Last outcome category']
              ].loc[street['Crime type'] == 'Bicycle theft']
bikes.head()
# 'Last outcomes' of bike theft and their frequencies:
outcomes = pd.DataFrame(bikes['Last outcome category'].value_counts())
# For each outcome I manually selected a colour for plotting. In particular,
# I picked warm colours (orange, red, brown) for the guilty party was found
# to make them stand out from unresolved or unseccessful cases
outcomes['Color'] = ['#00CED1','#6495ED','#66CDAA','#7B68EE','#FF4500',
                     '#DC143C','#8B0000','#A52A2A','#7FFFD4','#FF8C00',
                     '#FF6347','#00FF7F','#3CB371','#FF0000','#F4A460',
                     '#A0522D','#D2691E']
outcomes
# Plotting

# A detailed map of the roads of Greater London is too big
# to look well in the published version of this file.
# Hence, I "zoomed in" and displayed only teh central part of London.
# For a map of bike theft everywhere in Greater London,
# use the following coordinates:
# llcrnrlon=-0.526348576,
# llcrnrlat=51.277394014,
# urcrnrlon=0.347125976,
# urcrnrlat=51.712845414,

fig = plt.figure(figsize=(15, 10))
ax = fig.add_axes([0.25, 0.0, 1.0, 0.9])

fig.suptitle('Where not to park your bike in central London, \nand why not to expect to get it back when stollen',
          fontsize=20, horizontalalignment='left',
          weight='semibold')

# Basemap in a Matplotlib extention that allows ploting on maps.
# projection specifies the method of projecting the spherical Earth onto a flat surface
# Mercator ('merc') is a 2D cylindrical, conformal projection
# Every projection method causes distortions. lat_ts specifies the latitude of true scale
# llcrnrlon, llcrnrlat, urcrnrlon and urcrnrlat define the plotted region,
# e.g. llcrnrlor is the longitude of lower left corner of our map
m = Basemap(resolution='c',
            projection='merc',
            llcrnrlon=-0.15,
            llcrnrlat=51.49,
            urcrnrlon=-0.05,
            urcrnrlat=51.532,
            lat_ts=0,
            suppress_ticks=True);
m.drawmapboundary(color=None)

# Adding London roads from a shape file. This data comes from
# #https://download.geofabrik.de/europe/great-britain/england/greater-london.html
# and is an OpenStreetMap file.
# "shapefile" consists of a collection of files with a common filename prefix,
# stored in the same directory. The three mandatory files have filename
# extensions .shp, .shx, and .dbf
# Thus, when loading a "shapefile" provide a file name without extension:
m.readshapefile('../input/openstreetmap-greater-london-dec-2018/gis_osm_roads_free_1',
                'London', color='#ABB2B9')

lat = bikes['Latitude'].values
lon = bikes['Longitude'].values
colours = outcomes['Color'].to_dict() # a dict mappint outcome name to colour

# Plotting locations of bike theft instances, coloured by 'Last outcome Category':
x, y = m(lon, lat)  # transform coordinates for plotting on out map
# Seaborn (sns) library allows using a dataframe column to color a plot: 
sns.scatterplot(x, y, ax=ax, s=20, 
                hue=bikes['Last outcome category'],
                palette=colours)
# Specifying legend's position:
plt.legend(loc=0,bbox_to_anchor=(-0.25, 0.3, 0.25, 0.6))
m