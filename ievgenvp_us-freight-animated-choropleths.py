# load libraries and set basic options

import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import shapefile as shp

import plotly.graph_objs as go

import random

import io

import base64



from matplotlib.animation import FFMpegWriter

from matplotlib import gridspec

from matplotlib.patches import Polygon

from matplotlib import animation, rc, rcParams

from matplotlib.collections import PatchCollection

from mpl_toolkits.basemap import Basemap



from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from itertools import chain



#%matplotlib inline

from IPython.display import HTML, Image

rc('animation', html='html5')



init_notebook_mode(connected=True)

np.set_printoptions(formatter={'int_kind': '{:,}'.format})
# prepare commodity dictionary based on User Guide

com_list = ['Animals and Fish (live)', 'Cereal Grains (includes seed)',

            'Agricultural Products (excludes Animal Feed, Cereal Grains, and Forage Products)',

            'Animal Feed, Eggs, Honey, and Other Products of Animal Origin',

            'Meat, Poultry, Fish, Seafood, and Their Preparations',

            'Milled Grain Products and Preparations, and Bakery Products',

            'Other Prepared Foodstuffs, Fats and Oils',

            'Alcoholic Beverages and Denatured Alcohol',

            'Tobacco Products', 'Monumental or Building Stone',

            'Natural Sands', 'Gravel and Crushed Stone (excludes Dolomite and Slate)',

            'Other Non-Metallic Minerals not elsewhere classified',

            'Metallic Ores and Concentrates', 'Coal', 'Crude Petroleum',

            'Gasoline, Aviation Turbine Fuel, and Ethanol (includes Kerosene, and Fuel Alcohols)',

            'Fuel Oils (includes Diesel, Bunker C, and Biodiesel)',

            'Other Coal and Petroleum Products, not elsewhere classified',

            'Basic Chemicals', 'Pharmaceutical Products', 'Fertilizers',

            'Other Chemical Products and Preparations', 'Plastics and Rubber',

            'Logs and Other Wood in the Rough', 'Wood Products',

            'Pulp, Newsprint, Paper, and Paperboard', 'Paper or Paperboard Articles',

            'Printed Products', 'Textiles, Leather, and Articles of Textiles or Leather',

            'Non-Metallic Mineral Products', 

            'Base Metal in Primary or Semi-Finished Forms and in Finished Basic Shapes',

            'Articles of Base Metal', 'Machinery', 

            'Electronic and Other Electrical Equipment and Components, and Office Equipment',

            'Motorized and Other Vehicles (includes parts)', 'Transportation Equipment, not elsewhere classified',

            'Precision Instruments and Apparatus', 

            'Furniture, Mattresses and Mattress Supports, Lamps, Lighting Fittings, and Illuminated Signs',

            'Miscellaneous Manufactured Products', 'Waste and Scrap (excludes of agriculture or food)',

            'Mixed Freight', 'Commodity unknown']



com_dict = {}

for i in range(1,len(com_list)+1):

    com_dict[i] = com_list[i-1]



# make sure that dictionary has same numeration as User Guide

com_dict[99] = com_dict.pop(len(com_list))

com_dict[43] = com_dict.pop(42)



# dictionary of foreign trading partners

fr_dict = {801: "Canada", 802: "Mexico",

          803: "Rest of Americas", 804: "Europe",

          805: "Africa", 806: "SW & Central Asia",

          807: "Eastern Asia", 808: "SE Asia & Oceania",

          'total': "Total"}



# transportation mode dictionary and list

mode_dict = {1: "Truck", 2: "Rail", 3: "Water", 4: "Air",

             5: "Multimode and mail", 6: "Pipeline",

             7: "Other/unknown", 8: "No domestic mode"}



mode_list = ["Truck", "Rail", "Water", "Air",

             "Multimode and mail", "Pipeline",

             "Other/unknown"]



years = ['2012', '2013', '2014','2015','2020',

         '2025', '2030', '2035', '2040', '2045']
regions_df = pd.read_csv('../input/FAF4_Regional.csv',

                         dtype = {'dms_orig': str, 'dms_dest': str})

reader = shp.Reader('../input/CFS_AREA_shapefile_010215/CFS_AREA_shapefile_010215')
#Following sources helped a lot in making this animation:

#https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/

#http://louistiao.me/posts/notebooks/embedding-matplotlib-animations-in-jupyter-notebooks/

#http://louistiao.me/posts/notebooks/save-matplotlib-animations-as-gifs/

#http://matplotlib.org/api/_as_gen/matplotlib.animation.FuncAnimation.html#matplotlib.animation.FuncAnimation

#https://www.kaggle.com/kostyabahshetsyan/animated-choropleth-map-for-crime-statistic/code

#https://www.kaggle.com/jaeyoonpark/heatmap-animation-us-drought-map/code
# domestic origin by value 2012-2045 

dom_origin_df = regions_df.loc[pd.isnull(regions_df['fr_orig'])]
dff = dom_origin_df

category = 'dms_orig'

map_color = 'YlOrRd'

num_colors = 20

text = 'commodities with domestic origin'



# animated plot

cat_val_df = dff[[category,'value_2012',

                     'value_2013', 'value_2014',

                     'value_2015', 'value_2020',

                     'value_2025', 'value_2030',

                     'value_2035', 'value_2040',

                     'value_2045']].groupby(category, as_index = False).sum()



cat_val_df.columns = [category] + years



# color map for values origins (applicable to all periods)

values_df = cat_val_df[years]

values = values_df.values.reshape((cat_val_df.shape[0]*10,1))

    

# set up steady layer for the map

fig = plt.figure(figsize=(16, 8))

ax = fig.add_subplot(111)

    

# shapefile data and map setups

m = Basemap(width = 6000000, height = 4000000, projection = 'lcc',

                resolution = None, lat_1=27.,lat_2=32,lat_0=37,lon_0=-97.)

m.shadedrelief()

m.readshapefile('../input/CFS_AREA_shapefile_010215//CFS_AREA_shapefile_010215',

                    'regions', drawbounds = True, linewidth=.01)



# set up steady legend for the map

cm = plt.get_cmap(map_color)

scheme = [cm(q*1.0/num_colors) for q in range(num_colors)]

bins = np.linspace(values.min(), values.max(), num_colors)

ax_legend = fig.add_axes([0.8,0.12,0.02,0.77])

cmap = mpl.colors.ListedColormap(scheme)

cb = mpl.colorbar.ColorbarBase(ax_legend, cmap = cmap, ticks = bins,

                               boundaries = bins, orientation = 'vertical')

cb.ax.tick_params(labelsize=9)



# initial set up for polygons

chor_map = ax.plot([],[])[0]

for shape in m.regions:

        patches = [Polygon(np.array(shape), True)]

        pc = PatchCollection(patches)

        ax.add_collection(pc)



# animated elements of the map

def animate(j):

    year = years[j-1]

    fig.suptitle('Freight value of {} in year {}, USD million'.format(text, year), fontsize=20, y=.95)

    dms_orig_val_animated = cat_val_df.loc[:,[category, year]]

    dms_orig_val_animated.set_index(category, inplace = True)

    dms_orig_val_animated['bin'] = np.digitize(dms_orig_val_animated[year], bins) - 1

        

    for info, shape in zip(m.regions_info, m.regions):

        name = info['CFS07DDGEO']

        # some names of the regions are misstated in database or in shapefile

        # the purpose of if... loop is to make this names equal

        # -------------------------------------------------------------------

        if name == '349':

            name = '342'

        elif name == '100':

            name = '101'

        elif name == '330':

            name = '339'

        elif name == '310':

            name = '311'

        else:

            name

        # -------------------------------------------------------------------

        # Alaska and Hawaii were excluded from analysis to make map size reasonable

        if name not in ['020', '159', '151']: 

            color = scheme[dms_orig_val_animated.loc[name]['bin'].astype(int)]

            patches = [Polygon(np.array(shape), True)]

            pc = PatchCollection(patches)

            pc.set_facecolor(color)

            ax.add_collection(pc)

    return chor_map,



anim = animation.FuncAnimation(fig, func = animate, frames = 10,

                               repeat_delay = 2000, interval = 1000)



anim.save('freight_d_d.gif', writer='imagemagick')

plt.close()

Image(url='freight_d_d.gif', width = 1200, height = 1000)
# domestic destination by value 2012-2045 

dom_dest_df = regions_df.loc[pd.isnull(regions_df['fr_dest'])]
dff = dom_dest_df

category = 'dms_dest'

map_color = 'Greens'

num_colors = 20

text = 'commodities with domestic destinaton'



# animated plot

cat_val_df = dff[[category,'value_2012',

                     'value_2013', 'value_2014',

                     'value_2015', 'value_2020',

                     'value_2025', 'value_2030',

                     'value_2035', 'value_2040',

                     'value_2045']].groupby(category, as_index = False).sum()



cat_val_df.columns = [category] + years



# color map for values origins (applicable to all periods)

values_df = cat_val_df[years]

values = values_df.values.reshape((cat_val_df.shape[0]*10,1))

    

# set up steady layer for the map

fig = plt.figure(figsize=(16, 8))

ax = fig.add_subplot(111)

    

# shapefile data and map setups

m = Basemap(width = 6000000, height = 4000000, projection = 'lcc',

                resolution = None, lat_1=27.,lat_2=32,lat_0=37,lon_0=-97.)

m.shadedrelief()

m.readshapefile('../input/CFS_AREA_shapefile_010215//CFS_AREA_shapefile_010215',

                    'regions', drawbounds = True, linewidth=.01)



# set up steady legend for the map

cm = plt.get_cmap(map_color)

scheme = [cm(q*1.0/num_colors) for q in range(num_colors)]

bins = np.linspace(values.min(), values.max(), num_colors)

ax_legend = fig.add_axes([0.8,0.12,0.02,0.77])

cmap = mpl.colors.ListedColormap(scheme)

cb = mpl.colorbar.ColorbarBase(ax_legend, cmap = cmap, ticks = bins,

                               boundaries = bins, orientation = 'vertical')

cb.ax.tick_params(labelsize=9)



# initial set up for polygons

chor_map = ax.plot([],[])[0]

for shape in m.regions:

        patches = [Polygon(np.array(shape), True)]

        pc = PatchCollection(patches)

        ax.add_collection(pc)



# animated elements of the map

def animate(j):

    year = years[j-1]

    fig.suptitle('Freight value of {} in year {}, USD million'.format(text, year), fontsize=20, y=.95)

    dms_orig_val_animated = cat_val_df.loc[:,[category, year]]

    dms_orig_val_animated.set_index(category, inplace = True)

    dms_orig_val_animated['bin'] = np.digitize(dms_orig_val_animated[year], bins) - 1

        

    for info, shape in zip(m.regions_info, m.regions):

        name = info['CFS07DDGEO']

        # some names of the regions are misstated in database or in shapefile

        # the purpose of if... loop is to make this names equal

        # -------------------------------------------------------------------

        if name == '349':

            name = '342'

        elif name == '100':

            name = '101'

        elif name == '330':

            name = '339'

        elif name == '310':

            name = '311'

        else:

            name

        # -------------------------------------------------------------------

        # Alaska and Hawaii were excluded from analysis to make map size reasonable

        if name not in ['020', '159', '151']: 

            color = scheme[dms_orig_val_animated.loc[name]['bin'].astype(int)]

            patches = [Polygon(np.array(shape), True)]

            pc = PatchCollection(patches)

            pc.set_facecolor(color)

            ax.add_collection(pc)

    return chor_map,



anim = animation.FuncAnimation(fig, func = animate, frames = 10,

                               repeat_delay = 2000, interval = 1000)



anim.save('freight_d_o.gif', writer='imagemagick')

plt.close()

Image(url='freight_d_o.gif', width = 1200, height = 1000)
# dataframe for freight balance (outflow from regoin minus inflow to region)

dom_origin_bal_df = dom_origin_df[['dms_orig','value_2012',

                                'value_2013', 'value_2014',

                                'value_2015', 'value_2020',

                                'value_2025', 'value_2030',

                                'value_2035', 'value_2040',

                                'value_2045']].groupby('dms_orig', as_index = True).sum()



dom_dest_bal_df = dom_dest_df[['dms_dest','value_2012',

                                'value_2013', 'value_2014',

                                'value_2015', 'value_2020',

                                'value_2025', 'value_2030',

                                'value_2035', 'value_2040',

                                'value_2045']].groupby('dms_dest', as_index = True).sum()



dom_dest_bal_df = dom_dest_bal_df.apply(lambda x: x*(-1))



balance_df = dom_origin_bal_df.add(dom_dest_bal_df, fill_value = 0.0)

balance_df.reset_index(inplace = True)
dff = balance_df

category = 'dms_orig'

map_color = 'RdYlGn'

num_colors = 20

text = 'commodities input/output balance'



# animated plot

cat_val_df = dff[[category,'value_2012',

                     'value_2013', 'value_2014',

                     'value_2015', 'value_2020',

                     'value_2025', 'value_2030',

                     'value_2035', 'value_2040',

                     'value_2045']].groupby(category, as_index = False).sum()



cat_val_df.columns = [category] + years



# color map for values origins (applicable to all periods)

values_df = cat_val_df[years]

values = values_df.values.reshape((cat_val_df.shape[0]*10,1))

    

# set up steady layer for the map

fig = plt.figure(figsize=(16, 8))

ax = fig.add_subplot(111)

    

# shapefile data and map setups

m = Basemap(width = 6000000, height = 4000000, projection = 'lcc',

                resolution = None, lat_1=27.,lat_2=32,lat_0=37,lon_0=-97.)

m.shadedrelief()

m.readshapefile('../input/CFS_AREA_shapefile_010215//CFS_AREA_shapefile_010215',

                    'regions', drawbounds = True, linewidth=.01)



# set up steady legend for the map

cm = plt.get_cmap(map_color)

scheme = [cm(q*1.0/num_colors) for q in range(num_colors)]

bins = np.linspace(values.min(), values.max(), num_colors)

ax_legend = fig.add_axes([0.8,0.12,0.02,0.77])

cmap = mpl.colors.ListedColormap(scheme)

cb = mpl.colorbar.ColorbarBase(ax_legend, cmap = cmap, ticks = bins,

                               boundaries = bins, orientation = 'vertical')

cb.ax.tick_params(labelsize=9)



# initial set up for polygons

chor_map = ax.plot([],[])[0]

for shape in m.regions:

        patches = [Polygon(np.array(shape), True)]

        pc = PatchCollection(patches)

        ax.add_collection(pc)



# animated elements of the map

def animate(j):

    year = years[j-1]

    fig.suptitle('Freight value of {} in year {}, USD million'.format(text, year), fontsize=20, y=.95)

    dms_orig_val_animated = cat_val_df.loc[:,[category, year]]

    dms_orig_val_animated.set_index(category, inplace = True)

    dms_orig_val_animated['bin'] = np.digitize(dms_orig_val_animated[year], bins) - 1

        

    for info, shape in zip(m.regions_info, m.regions):

        name = info['CFS07DDGEO']

        # some names of the regions are misstated in database or in shapefile

        # the purpose of if... loop is to make this names equal

        # -------------------------------------------------------------------

        if name == '349':

            name = '342'

        elif name == '100':

            name = '101'

        elif name == '330':

            name = '339'

        elif name == '310':

            name = '311'

        else:

            name

        # -------------------------------------------------------------------

        # Alaska and Hawaii were excluded from analysis to make map size reasonable

        if name not in ['020', '159', '151']: 

            color = scheme[dms_orig_val_animated.loc[name]['bin'].astype(int)]

            patches = [Polygon(np.array(shape), True)]

            pc = PatchCollection(patches)

            pc.set_facecolor(color)

            ax.add_collection(pc)

    return chor_map,



anim = animation.FuncAnimation(fig, func = animate, frames = 10,

                               repeat_delay = 2000, interval = 1000)



anim.save('freight_bal.gif', writer='imagemagick')

plt.close()

Image(url='freight_bal.gif', width = 1200, height = 1000)
# origination dataframe

dom_origin_comm_df = dom_origin_df[['sctg2','value_2012',

                                'value_2013', 'value_2014',

                                'value_2015', 'value_2020',

                                'value_2025', 'value_2030',

                                'value_2035', 'value_2040',

                                'value_2045']].groupby('sctg2', as_index = False).sum()



dom_origin_comm_df.columns = ['sctg2'] + years

dom_origin_comm_df.loc['total'] = dom_origin_comm_df.sum()
# destination dataframe

dom_dest_comm_df = dom_dest_df[['sctg2','value_2012',

                                'value_2013', 'value_2014',

                                'value_2015', 'value_2020',

                                'value_2025', 'value_2030',

                                'value_2035', 'value_2040',

                                'value_2045']].groupby('sctg2', as_index = False).sum()



dom_dest_comm_df.columns = ['sctg2'] + years

dom_dest_comm_df.loc['total'] = dom_dest_comm_df.sum()
# list of top 10 commodities by consumption (2012 and 2045)

top_10_dest_2012 = dom_dest_comm_df[['sctg2','2012','2045']].sort_values('2012', axis = 0, ascending = False)

top_10_dest_2012 = top_10_dest_2012.head(11)

top_10_dest_2012.drop('total', inplace = True)



top_10_dest_2045 = dom_dest_comm_df[['sctg2','2012','2045']].sort_values('2045', axis = 0, ascending = False)

top_10_dest_2045 = top_10_dest_2045.head(11)

top_10_dest_2045.drop('total', inplace = True)



top_2012_dest = top_10_dest_2012['sctg2'].values.flatten().tolist()

top_2045_dest = top_10_dest_2045['sctg2'].values.flatten().tolist()

top_dest_list = [x for x in top_2012_dest or top_2045_dest]



top_dest_2012_2045 = dom_dest_comm_df.loc[dom_dest_comm_df['sctg2'].isin(top_dest_list), 

                                          ['sctg2']+years]

top_dest_2012_2045.reset_index(inplace = True, drop = True)

top_dest_2012_2045['Commodity'] = top_dest_2012_2045['sctg2'].apply(lambda x: com_dict[x])
# list of top 10 commodities by production (2012 and 2045)

top_10_origin_2012 = dom_origin_comm_df[['sctg2','2012','2045']].sort_values('2012', axis = 0, ascending = False)

top_10_origin_2012 = top_10_origin_2012.head(11)

top_10_origin_2012.drop('total', inplace = True)



top_10_origin_2045 = dom_origin_comm_df[['sctg2','2012','2045']].sort_values('2045', axis = 0, ascending = False)

top_10_origin_2045 = top_10_origin_2045.head(11)

top_10_origin_2045.drop('total', inplace = True)



top_2012_origin = top_10_origin_2012['sctg2'].values.flatten().tolist()

top_2045_origin = top_10_origin_2045['sctg2'].values.flatten().tolist()

top_origin_list = [x for x in top_2012_origin or top_2045_origin]



top_origin_2012_2045 = dom_origin_comm_df.loc[dom_origin_comm_df['sctg2'].isin(top_origin_list), 

                                              ['sctg2'] + years]

top_origin_2012_2045.reset_index(inplace = True, drop = True)

top_origin_2012_2045['Commodity'] = top_origin_2012_2045['sctg2'].apply(lambda x: com_dict[x])
# check assumption that top origin and top destination commodities lists contains same items

print (len([x for x in top_origin_list and top_dest_list]))

# As there are only 10 items in final list we can say that they contain same items.

# Thus, they can be used interchangeably as parameters of the function "line_plot".

# This will help to preserve color scheme during making different plots.
# function for plotting top 2012/2045 destinations/originations by commodity

# plotly web page with examples was the best source of information https://plot.ly/python/

def line_plot(category, sel_list, df, sel_dict, clustering_criterion, xAxis):

    data = []

    buttons = []

    for i in sel_list:

        r = random.randint(1,256)

        g = random.randint(1,256)

        b = random.randint(1,256)

        rgb = 'rgb({}, {}, {})'.format(r, g, b)

        trace = go.Scatter(x = ["year {}".format(x) for x in xAxis],

                           y = df.loc[df[category] == i, xAxis].apply(lambda x: x/1000000).values.flatten(),

                           name = '{}_{}'.format(" ".join(sel_dict[i].split(" ")[:2]), i),

                           line = dict(width = 2,

                                       dash = 'longdash'))

        data.extend([trace])



        buttons_upd = list([dict(label = '{}'.format(sel_dict[i]),

                                 method = 'update',

                                 args = [{'visible': [x==i for x in sel_list]}])])

        buttons.extend(buttons_upd)



    # button for reset / all items

    buttons_all = list([dict(label = 'All selected items',

                                 method = 'update',

                                 args = [{'visible': [True for x in sel_list]}])])

    buttons.extend(buttons_all)

    

    # set menues inside the plot

    update_menus = list([dict(active=-5,

                              buttons = buttons,

                              direction = 'down',

                              pad = {'r': 10, 't': 10},

                              showactive = True,

                              x = 0.001,

                              xanchor = 'left',

                              y = 1.1,

                              yanchor = 'top')])

    # Edit the layout

    layout = dict(title = '{}'.format(clustering_criterion),

                  xaxis = dict(title = 'Years',

                               nticks = len(xAxis)),

                  yaxis = dict(title = 'Value, trillion USD'),

                  updatemenus = update_menus,

                  showlegend = True)

         

    fig_top_10 = dict(data = data, layout = layout)

    iplot(fig_top_10)
# plot top 2012 vs 2045 destination by commodity

line_plot('sctg2', top_dest_list, top_origin_2012_2045, com_dict,

          'Top 10 commodities by production', years)
# plot top 2012 vs 2045 origination by commodity

line_plot('sctg2', top_dest_list, top_dest_2012_2045, com_dict,

          'Top 10 commodities by consumption', years)
# plot biggest commodity deficits and surpluses

dom_dest_comm_df_2 = dom_dest_comm_df.set_index(['sctg2'])

dom_origin_comm_df_2 = dom_origin_comm_df.set_index(['sctg2'])

dom_dest_comm_df_neg = dom_dest_comm_df_2.apply(lambda x: x*(-1))



comm_balance_df = dom_origin_comm_df_2.add(dom_dest_comm_df_neg, fill_value = 0.0)

comm_balance_df['max'] = comm_balance_df[years].max(axis = 1)

comm_balance_df['min'] = comm_balance_df[years].min(axis = 1)

comm_balance_df['abs_max'] = comm_balance_df[['max','min']].apply(lambda x: abs(x)).max(axis = 1)



selected_comm_bal = comm_balance_df.loc[comm_balance_df['abs_max'] >= 150000, years]

selected_comm_bal.reset_index(inplace = True)



bal = selected_comm_bal['sctg2'].values.flatten().tolist()

com_dict[1003] = "Total for all commodities"
# function for plotting balance

def balance_plot(category, sel_list, df, sel_dict, heading):

    data = []

    buttons = []



    # rgb for surplus

    r_s = 200

    g_s = 100

    b_s = 20



    # rgb for deficite

    r_d = 50

    g_d = 100

    b_d = 200



    for i in sel_list:

        y = df.loc[df[category] == i,years].apply(lambda x: x/1000000).values

        if np.sum(y)>=0:

            rgb = 'rgb({}, {}, {})'.format(r_s, g_s, b_s)

            r_s += 5

            g_s += 10

            b_s += 10

        else:

            rgb = 'rgb({}, {}, {})'.format(r_d, g_d, b_d)

            r_d += 10

            g_d += 10

            b_d += 5

        trace = go.Scatter(x = ["year {}".format(x) for x in years],

                       y = y.flatten(),

                       name = '{}_{}'.format(" ".join(sel_dict[i].split(" ")[:2]), i),

                       line = dict(color = (rgb), 

                                   width = 2))

        

    

        data.append(trace)

    

        buttons_upd = list([dict(label = '{}'.format(sel_dict[i]),

                                 method = 'update',

                                 args = [{'visible': [x==i for x in sel_list]}])])

        buttons.extend(buttons_upd)



    buttons_all = list([dict(label = 'All except total',

                         method = 'update',

                         args = [{'visible': [True for x in bal[:-1]]+[False]}])]) 



    buttons.extend(buttons_all)



    # set menues inside the plot

    update_menus = list([dict(active=-5,

                              buttons = buttons,

                              direction = 'down',

                              pad = {'r': 10, 't': 10},

                              showactive = True,

                              x = 0.001,

                              xanchor = 'left',

                              y = 1.1,

                              yanchor = 'top')])



    # Edit the layout

    layout = dict(title = heading,

                  xaxis = dict(title = 'Years'),

                  yaxis = dict(title = 'Value, trillion USD'),

                  updatemenus = update_menus,

                  showlegend = True)

         

    fig = dict(data = data, layout = layout)

    iplot(fig)
balance_plot('sctg2', bal, selected_comm_bal, com_dict, 

             'Balance of commodities consumed and produced')
# domestically originated goods exported to other countries

fr_dest_df = regions_df.loc[pd.notnull(regions_df['fr_dest'])]

fr_dest_list = fr_dest_df['fr_dest'].unique().flatten().tolist()

fr_dest_df = fr_dest_df[['fr_dest','value_2012',

                           'value_2013', 'value_2014',

                           'value_2015', 'value_2020',

                           'value_2025', 'value_2030',

                           'value_2035', 'value_2040',

                           'value_2045']].groupby('fr_dest', as_index = True).sum()



fr_dest_df.columns = years
# goods imported from other countries

fr_orig_df = regions_df.loc[pd.notnull(regions_df['fr_orig'])]

fr_orig_list = fr_orig_df['fr_orig'].unique().flatten().tolist()

fr_orig_df = fr_orig_df[['fr_orig','value_2012',

                           'value_2013', 'value_2014',

                           'value_2015', 'value_2020',

                           'value_2025', 'value_2030',

                           'value_2035', 'value_2040',

                           'value_2045']].groupby('fr_orig', as_index = True).sum()



fr_orig_df.columns = years
# fr balance df

fr_balance_df = fr_dest_df.add(fr_orig_df.apply(lambda x: x*(-1)))

fr_balance_df.loc['total'] = fr_balance_df.sum()

fr_balance_df.reset_index(inplace = True)
# reset index for fr dataframes

fr_dest_df.reset_index(inplace = True)

fr_orig_df.reset_index(inplace = True)
def fr_dest_orig_plot(category, list_fr, df, clustering_criterion):

    data = []

    for i in list_fr:



        trace = go.Bar(x = ["year {}".format(x) for x in years],

                       y = df.loc[df[category] == i,years].apply(lambda x: x/1000000).values.flatten(),

                       name = '{}'.format(fr_dict[i]))

        

        data.append(trace)

                       

    # Edit the layout

    layout = dict(title = 'Trading partners grouped by {}'.format(clustering_criterion),

                  xaxis = dict(title = 'Years'),

                  yaxis = dict(title = 'Value, trillion USD'),

                  showlegend = True,

                  barmode='stack')

         

    fig = dict(data = data, layout = layout)

    iplot(fig, filename='stacked-bar')
# plot overall change and structure of trading partners by export

fr_dest_orig_plot('fr_dest', fr_dest_list, fr_dest_df, 'export')
# plot trading partners by export

line_plot('fr_dest', fr_dest_list, fr_dest_df, fr_dict,

          'Trading partners by export', years)
# plot overall change and structure of trading partners by export

fr_dest_orig_plot('fr_orig', fr_dest_list, fr_orig_df, 'import') # fr_dest_list was applied to match colors with previous plots
# plot trading partners by export

line_plot('fr_orig', fr_dest_list, fr_orig_df, fr_dict,

          'Trading partners by import', years) # fr_dest_list was applied to match colors with previous plots
# plot balance for foreign destination / origination

fr_dest_list_total = fr_dest_list+['total']

balance_plot('fr_dest', fr_dest_list_total, fr_balance_df, fr_dict, 'Balance of foreign trade by commodities')
#print (regions_df)

ng_europe = regions_df[['value_2012',

                     'value_2013', 'value_2014',

                     'value_2015', 'value_2020',

                     'value_2025', 'value_2030',

                     'value_2035', 'value_2040',

                     'value_2045', 'fr_dest', 'sctg2']]

pd.to_numeric(ng_europe['fr_dest'], errors='coerce')

ng_europe.columns = [years + ['fr_dest','sctg2']]



ng_europe = ng_europe.loc[regions_df['fr_dest'] == 804, years+['sctg2']]

ng_europe = ng_europe.groupby('sctg2', as_index = False).sum()

ng_europe = ng_europe.sort_values('2045', axis = 0, ascending = False)

ng_europe_list = ng_europe['sctg2'].head(10).values.flatten().tolist()



line_plot('sctg2', top_dest_list, ng_europe, com_dict,

          'Natural gas export to Europe', years)
# domestic freight type (includes only transportation between entry/exit

# point and destination/origin point)

domestic_mode_df = regions_df[['dms_mode','value_2012',

                               'value_2013', 'value_2014',

                               'value_2015', 'value_2020',

                               'value_2025', 'value_2030',

                               'value_2035', 'value_2040',

                               'value_2045']].groupby('dms_mode', as_index = False).sum()



domestic_mode_df.columns = ['dms_mode'] + years



# exclude mode '8' as it is not domestic transportation mode

domestic_mode_df = domestic_mode_df.loc[domestic_mode_df['dms_mode']!=8,]

domestic_mode_df[years] = domestic_mode_df[years].div(domestic_mode_df[years].sum(axis=0),

                                                      axis=1).multiply(100)
# foreign fraight type (both import and export)

# domestic origin for export by value 2012-2045

for_dest_df = regions_df.loc[pd.notnull(regions_df['fr_dest'])]

for_outmode_df = for_dest_df[['fr_outmode','value_2012',

                               'value_2013', 'value_2014',

                               'value_2015', 'value_2020',

                               'value_2025', 'value_2030',

                               'value_2035', 'value_2040',

                               'value_2045']].groupby('fr_outmode', as_index = True).sum()



# domestic destination for import by value 2012-2045

for_orig_df = regions_df.loc[pd.notnull(regions_df['fr_orig'])]

for_inmode_df = for_orig_df[['fr_inmode','value_2012',

                                  'value_2013', 'value_2014',

                                  'value_2015', 'value_2020',

                                  'value_2025', 'value_2030',

                                  'value_2035', 'value_2040',

                                  'value_2045']].groupby('fr_inmode', as_index = True).sum()



for_mode_df = for_outmode_df.add(for_inmode_df)

for_mode_df.reset_index(inplace = True)

for_mode_df.columns = ['for_mode'] + years



for_mode_df[years] = for_mode_df[years].div(for_mode_df[years].sum(axis=0), axis=1).multiply(100)
# plotting pie charts for transportation modes

data_pie = []

pie_colors = ['rgb(100, 100, 100)',

              'rgb(230, 120, 40)',

              'rgb(110, 210, 220)',

              'rgb(220, 220, 220)',

              'rgb(180, 60, 110)',

              'rgb(80, 120, 150)',

              'rgb(125, 30, 120)']



# add year 2012 manually to ensure it appears on plot right after the code is executed

data_2012 = [{"values": domestic_mode_df['2012'].values.flatten().tolist(),

                 "labels": mode_list,

                 "domain": {"x": [0, .48]},

                 "marker": {"colors": pie_colors},

                 "hoverinfo": "label+percent",

                 "hole": .4,

                 "type": 'pie',

                 "visible": True},

           

                {"values": for_mode_df['2012'].values.flatten().tolist(),

                 "labels": mode_list,

                 "domain": {"x": [.52, 1]},

                 "marker": {"colors": pie_colors},

                 "hoverinfo": "label+percent",

                 "hole": .4,

                 "type": 'pie',

                 "visible": True}]

data_pie.extend(data_2012)



for i in years[1:]:    

    data_upd = [{"values": domestic_mode_df[i].values.flatten().tolist(),

                 "labels": mode_list,

                 "domain": {"x": [0, .48]},

                 "marker": {"colors": pie_colors},

                 "hoverinfo": "label+percent",

                 "hole": .4,

                 "type": 'pie',

                 "visible": False},

           

                {"values": for_mode_df[i].values.flatten().tolist(),

                 "labels": mode_list,

                 "domain": {"x": [.52, 1]},

                 "marker": {"colors": pie_colors},

                 "hoverinfo": "label+percent",

                 "hole": .4,

                 "type": 'pie',

                 "visible": False}]

    

    data_pie.extend(data_upd)

    

# set menues inside the plot

steps = []

yr = 0



for i in range(0,len(data_pie),2):

    step = dict(method = "restyle",

                args = ["visible", [False]*len(data_pie)],

                label = years[yr]) 

    step['args'][1][i] = True

    step['args'][1][i+1] = True

    steps.append(step)

    yr += 1



sliders = [dict(active = 0,

                currentvalue = {"prefix": "Year: ",

                               "visible": True},

                pad = {"t": 50},

                steps = steps)]



# Set the layout

layout = dict(title = 'Structure of transportation mode',

              annotations = [{"font": {"size": 20},

                              "showarrow": False,

                              "text": "DMT",

                              "x": 0.20,

                              "y": 0.5},

                             {"font": {"size": 20},

                              "showarrow": False,

                              "text": "FMT",

                              "x": 0.8,

                              "y": 0.5}],

              sliders = sliders)

         

fig = dict(data = data_pie, layout = layout)

iplot(fig, filename='donut')