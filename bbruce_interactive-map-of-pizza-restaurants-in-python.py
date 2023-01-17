# import packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# import Bokeh packages for interactive plots

from bokeh.plotting import figure, show, output_notebook

from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper

from bokeh.tile_providers import CARTODBPOSITRON_RETINA

from bokeh.palettes import Category20b, Category20c, Spectral

from bokeh.layouts import gridplot



output_notebook()
# read pizza data

pizza = pd.read_csv('../input/pizza-restaurants-and-the-pizza-they-sell/8358_1.csv')

pizza.head()
# have a look

pizza.info()
# extract interested columns

pizza_sub = pizza.copy()[['id', 'city', 'address', 'postalCode', 

                          'menus.name', 'latitude', 'longitude',

                          'menus.amountMax', 'menus.amountMin']]

pizza_sub.info()
# drop duplicates

pizza_sub = pizza_sub.drop_duplicates(subset=['id', 'menus.name'])



# count the pizza names

names_of_pizza = pizza_sub['menus.name'].value_counts()



# merge `names_of_pizza` to `pizza_sub`

pizza_sub = pd.merge(pizza_sub, names_of_pizza.to_frame(),

                     left_on='menus.name', right_index=True, how='left')
plt.figure(figsize=(14,3))

plt.plot(names_of_pizza.head(20), linestyle='none', markersize=15, marker='o')

plt.title('Top 20 Popular Pizza', fontsize=20)

plt.xticks(rotation=90)

plt.xlabel('Pizza Name', fontsize=15)

plt.ylabel('Counts', fontsize=15)

plt.grid(alpha=.3)

plt.margins(.05)

plt.show()
# read zipcode data

zipcode = pd.read_csv('../input/zip-code-income-tax-data-2014/14zpallagi.csv',

                      usecols=['STATE', 'zipcode'], dtype={'zipcode': 'str'})



zipcode.head()
# drop duplicates

zipcode = zipcode.drop_duplicates(['zipcode'])

zipcode.head()
# merge `pizza data` and `zipcode data`

pizza_merge = pd.merge(pizza_sub, zipcode, left_on='postalCode', right_on='zipcode', how='left')

pizza_merge.head()
# count the restaurants by State

counts = pizza_merge.STATE.value_counts()



# merge

pizza_clean = pd.merge(pizza_merge, counts.to_frame(), 

                       left_on='STATE', right_index=True, how='left')



# drop unused columns and missing values

pizza_clean = pizza_clean.drop(['id', 'postalCode', 'zipcode'], axis=1).dropna()



# rename columns

pizza_clean.columns = ['city', 'address', 'pizza_name',

                        'latitude', 'longitude', 

                        'menus_amountMax', 'menus_amountMin',

                        'pizza_counts', 'state', 'state_counts']



# set the size of dot for scatterplot later

pizza_clean['dot_size'] = pizza_clean.menus_amountMax**.55

pizza_clean.head()
# define functions for coordinate projection

import math



def lgn2x(a):

    return a * (math.pi/180) * 6378137



def lat2y(a):

    return math.log(math.tan(a * (math.pi/180)/2 + math.pi/4)) * 6378137
# project coordinates

pizza_clean['x'] = pizza_clean.longitude.apply(lambda row: lgn2x(row))

pizza_clean['y'] = pizza_clean.latitude.apply(lambda row: lat2y(row))



# drop unused columns

pizza_clean = pizza_clean.drop(['latitude', 'longitude'], axis=1)

pizza_clean.head()
# drop duplicated restaurants and keep the most expensive one

pizza_map = (pizza_clean.sort_values(['menus_amountMax'], ascending=[0])

             .drop_duplicates(subset=['city', 'address']))



# create ColumnDataSource

cds = ColumnDataSource(pizza_map)



# customize hover tool

hover = HoverTool(tooltips=[('City', '@city'),

                            ('Address', '@address'),

                            ('Pizza', '@pizza_name'),

                            ('Max price', '@menus_amountMax')],

                  mode='mouse')



# UPPER FIGURE

# initialize a figure

up = figure(title='Location of Pizza Restaurants in US',

           plot_width=780, plot_height=360,

           x_axis_location=None, y_axis_location=None, 

           tools=['pan', 'wheel_zoom', 'tap', 'reset', 'crosshair', hover])



# overlap map

up.add_tile(CARTODBPOSITRON_RETINA)



# create color mapper

mapper = CategoricalColorMapper(factors=pizza_map.state.unique(), 

                                palette=Category20b[20]+Category20c[20]+Spectral[4])



# add pizza location

scatter = up.circle('x', 'y', source=cds, size='dot_size',

                    color={'field': 'state','transform': mapper}, alpha=.5,

                    selection_color='black',

                    nonselection_fill_alpha=.1,

                    nonselection_fill_color='gray',)

                  



# BOTTOM FIGURE

# initialize a figure

down = figure(title='Number of Restaurants in each State (Click bar below)',

              x_range=pizza_map.state.unique(),

              plot_width=780, plot_height=200,

              tools=['tap', 'reset'])



# add restaurant counts

down.vbar(x='state', top='state_counts', source=cds, width=.7,

            color={'field': 'state','transform': mapper},

            selection_color='black',

            nonselection_fill_alpha=.1,

            nonselection_fill_color='gray',)





# set graph properties

down.xgrid.grid_line_color = None

down.xaxis.major_label_orientation = 'vertical'

down.xaxis.axis_label = 'State'

down.yaxis.axis_label = 'Count'

p = gridplot([[up], [down]], toolbar_location='left',)



# show the plot

show(p)