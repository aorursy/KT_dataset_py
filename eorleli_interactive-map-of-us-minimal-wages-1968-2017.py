import bokeh.sampledata

bokeh.sampledata.download() # In case you do not have this

import numpy as np

import pandas as pd

import random

import matplotlib.pyplot as plt

from bokeh.io import output_notebook, show

from bokeh.plotting import figure

from bokeh.plotting import gmap

from bokeh.tile_providers import CARTODBPOSITRON

from bokeh.models import WMTSTileSource, LinearColorMapper, LogColorMapper,ColumnDataSource, HoverTool, CustomJS, Slider, ColorBar, FixedTicker

from bokeh.sampledata.us_counties import data as counties

from bokeh.palettes import Viridis256,Viridis6, Greys256, Spectral5

from bokeh.transform import linear_cmap, factor_cmap

from bokeh.layouts import row, column, widgetbox

from matplotlib import colors as mcolors

from bokeh.sampledata.us_states import data as states
output_notebook()
#wage = pd.read_excel("Minimum_Wage_Data.xlsx", sheet_name = "Minimum_Wage_Data")

wage = pd.read_csv("../input/Minimum Wage Data.csv", encoding = "Windows-1252" )

wage.head()
wage.info()
number_of_states = np.unique(wage['State']).tolist()

print ("The states in the data are: ", number_of_states)

print ("\nTotal number of states: ", len(number_of_states))
state = pd.DataFrame(states)

state.head()
dic = dict(zip(state.columns.tolist(),(state.iloc[2,:]).tolist())) # dictionary like {'NV':'Nevada','AZ':'Arizona'...}

state = state.rename(columns=dic) # the 'rename' function requires dictionary to change the column names

state = state.reindex(sorted(state.columns),axis=1) # now I can reorder the columns to fit the 'wage' dataset

state.head()
print ("The states in the 'state' dataset are: ", list(set(state.iloc[2,:].tolist()).intersection(number_of_states)))

print ("\nNumber of states in the 'state' dataset: ",len((list(set(state.iloc[2,:].tolist()).intersection(number_of_states)))))
wage = wage[-wage["State"].isin(['Guam','U.S. Virgin Islands','Puerto Rico','Federal (FLSA)'])]

wage = wage.reset_index()



wage1968 = wage[wage['Year']==1968]

wage1968.index = range(len(wage1968)) # as I removed four rows, I have to shift up the indices
url = 'http://a.basemaps.cartocdn.com/rastertiles/voyager/{Z}/{X}/{Y}.png'

attribution = "Tiles by Carto, under CC BY 3.0. Data by OSM, under ODbL"

USA = x_range,y_range = ((-13884029,-7453304), (2698291,6455972))

def wgs84_to_web_mercator(df, lon="lon", lat="lat"):

    """Converts decimal longitude/latitude to Web Mercator format"""

    k = 6378137

    df["x"] = df[lon] * (k * np.pi/180.0)

    df["y"] = np.log(np.tan((90 + df[lat]) * np.pi/360.0)) * k

    return df



number_of_unique_states = len(np.unique(wage['State']).tolist())

for i in range(0,number_of_unique_states):

    new = wgs84_to_web_mercator(pd.DataFrame(dict(lon=state.iloc[1,i], lat=state.iloc[0,i])))

    state.iloc[0,i] = new['y'].tolist()

    state.iloc[1,i] = new['x'].tolist()

    

state_xs = [i for i in state.loc["lons",:].tolist()]

state_ys = [i for i in state.loc["lats",:].tolist()]

statename = wage1968["State"]



nd = dict(x = state_xs, y = state_ys, name = statename)



total_years = np.unique(wage['Year'])

number_of_total_years = len(total_years)

for i in range(0,number_of_total_years): 

    nd[str(np.unique(wage['Year'])[i])] = wage[wage['Year']==np.unique(wage['Year'])[i]]["High.2018"].tolist()

    

nd['used'] = nd['1968']

source = ColumnDataSource(nd)
TOOLS = "pan,wheel_zoom,reset,hover,save"

color_mapper = LogColorMapper(palette=Greys256,low=0, high=15)

Greys256.reverse()

p = figure(x_range=x_range, y_range=y_range, x_axis_type="mercator", y_axis_type="mercator", tools = TOOLS, 

        tooltips=[("State","@name"),("Salary","@used")])

p.add_tile(WMTSTileSource(url=url, attribution=attribution))        

renderer = p.patches('x', 'y', source=source, fill_color={'field': 'used', 'transform': color_mapper})     

p.hover.point_policy = "follow_mouse"

  

callback = CustomJS(args=dict(source=source,plot=p,color_mapper = color_mapper,renderer = renderer), code="""

    var data = source.data;

    var year = year.value;

    used = data['used']

    should_be = data[String(year)]

    for (i = 0; i < should_be.length; i++) {

    used[i] = should_be[i];

    } 

    source.change.emit()

""")



year_slider = Slider(start=1968, end=2017, value=1968, step=1, title="year")

callback.args['year'] = year_slider

year_slider.js_on_change('value', callback)

layout = column(year_slider,p)



ticker = FixedTicker(ticks = [0,3,6,9,12,15])

color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, ticker = ticker, major_label_text_font_size='10pt', border_line_color=None, location=(0,0))



p.add_layout(color_bar, 'right')



show(layout)
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)



listy = []

for item in colors.keys():

    listy.append(item)

l = listy[0:number_of_unique_states]



list_of_years = [np.unique(wage.Year).tolist()]*number_of_unique_states

salary_list = []

for i in range(number_of_unique_states): 

    salary_list.append(wage[wage.State==np.unique(wage.State)[i]]["High.2018"].tolist())



wage_d = dict(Years = list_of_years, Salary = salary_list, State = np.unique(wage.State).tolist(),c = l)

source2 = ColumnDataSource(wage_d)



p = figure(plot_width=800, plot_height=600, tools = TOOLS, tooltips=[("State","@State")])

p.multi_line(xs = 'Years',ys ='Salary',source=source2, line_width  = 2, line_color = 'c')



p.xaxis.axis_label = 'Year'

p.xaxis.axis_label_text_font_size = "12pt" 

p.xaxis.major_label_text_font_size = "12pt"

p.xaxis.axis_label_text_font_style = "normal"         

p.yaxis.axis_label = "Minimal salary adjusted to CPI 2018"

p.yaxis.major_label_text_font_size = "12pt"

p.yaxis.axis_label_text_font_style = "normal"  

p.yaxis.axis_label_text_font_size = "12pt" 

p.title.text = "Minimal salary adjusted to CPI 2018 in US states"

p.title.text_font_size = "12pt"

p.title.text_font_style = "bold"  



show(p)