import numpy as np 

import pandas as pd 

import os

import seaborn as sns

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

files = os.listdir('/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Data/Stocks/')

stocks_path = '/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Data/Stocks/'

files[0]
stock_names = [f.split('.')[0] for f in files[:10]]

stock_names
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols

symbols = get_nasdaq_symbols()

symbols.info()
symbols.head()
symbols[symbols.index.str.startswith('AIMT')]
import pprint

for s in stock_names:

    print(s)

    print(symbols[symbols.index.str.startswith(s.upper())])

    print('*'*10)
from scipy.stats import normaltest

import sys

stocks_data = []

for f,stock_name in zip(files, stock_names):

    # print(stocks_path+f)



    try:

        df = pd.read_csv(stocks_path+f, parse_dates=[0])

        df['stock'] = stock_name

        stocks_data.append(df)

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:

            print(stock_name, col, normaltest(df[col].pct_change().dropna()))

            # Distributions are not normal https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/    

    except:

        print("Unexpected error:", sys.exc_info()[0])





stocks_data[0].info()
df_stocks = pd.concat(stocks_data, ignore_index=True)

df_stocks.info()
_=df_stocks.groupby('stock')['Open'].describe().plot(kind='bar', subplots=True, figsize=(15,15))
!head -n 5 '/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Data/Stocks/icui.us.txt'
stocks_data[0].head()
df_stocks.set_index('Date').groupby('stock')['Open'].plot(lw=1, legend=True, 

       figsize=(14, 6));
# Import figure from bokeh.plotting

from bokeh.plotting import figure

# Import output_file and show from bokeh.io

from bokeh.io import output_file,show

from bokeh.plotting import figure, output_notebook, show

# Create the figure: p

p = figure(x_axis_type='datetime')

# Add a circle glyph to the figure p

p.circle(x=stocks_data[0]['Date'], y=stocks_data[0]['Open'])

# Call the output_file() function and specify the name of the file

output_notebook()

# Display the plot

show(p)

# from bokeh.models import ColumnDataSource, Select

# from bokeh.plotting import figure

# from bokeh.layouts import row

# from bokeh.io import curdoc

# from bokeh.io import output_file, show

# from bokeh.io import output_notebook, push_notebook, show

# from bokeh.layouts import widgetbox

# output_notebook()

# # https://stackoverflow.com/questions/55482198/how-to-show-curdoc-dashboard-in-jupyter-notebook

# #def make_doc(doc):

# # Create ColumnDataSource: source

# source = ColumnDataSource(data={

#     'x' : stocks_data[0]['Date'],

#     'y' : stocks_data[0]['Open']

# })

# # Create a new plot: plot

# plot = figure()

# # Add circles to the plot

# plot.circle('x', 'y', source=source)

# # Create a dropdown Select widget: select    

# select = Select(title="Open", options=stock_names, value=stock_names[0])



# # Define a callback function: update_plot

# def update_plot(attr, old, new):

#     # If the new Selection is 'female_literacy', update 'y' to female_literacy

# #     if new == 'female_literacy':

# #         source.data = {

# #             'x' : fertility,

# #             'y' : female_literacy

# #         }

# #     # Else, update 'y' to population

# #     else:

# #         source.data = {

# #             'x' : fertility,

# #             'y' : population

# #         }

#    # if old!=new:

#     ind = stock_names.index(new)

#     source.data = {

#         'x' : stocks_data[ind]['Date'],

#         'y' : stocks_data[ind]['Open']

#     }

#     output_notebook()

# #     layout = row(select, plot)

# #     curdoc().add_root(layout)

# #     handle = show(layout, notebook_handle=True)

# #     push_notebook(handle=handle)

#     show(row(plot,widgetbox([select])))



# # Attach the update_plot callback to the 'value' property of select

# select.on_change('value', update_plot)

# # Create layout and add to current document

# # layout = row(select, plot)

# # curdoc().add_root(layout)

# # #show(plot, notebook_handle=True)    

# # handle = show(layout, notebook_handle=True)

# # push_notebook(handle=handle)



# show(row(plot,widgetbox([select])))
from ipywidgets import interact, interactive, fixed, interact_manual

import ipywidgets as widgets

import matplotlib.pyplot as plt

@interact (st= stock_names)

def plot(st):

    ind = stock_names.index(st)

    _=stocks_data[ind].set_index('Date')[['Open', 'High', 'Low', 'Close']].plot(kind='line',rot=45) 
from ipywidgets import interact, interactive, fixed, interact_manual

import ipywidgets as widgets

import matplotlib.pyplot as plt

@interact (st= stock_names)

def plot(st):

    ind = stock_names.index(st)

    # _=stocks_data[ind].set_index('Date')[['Open', 'High', 'Low', 'Close']].plot(kind='line',rot=45) 

    data = stocks_data[ind]

    ax = sns.boxplot(x=data['Date'].dt.year, y=data['Open'])

    plt.xticks(rotation=45)

    plt.show()
from ipywidgets import interact, interactive, fixed, interact_manual

import ipywidgets as widgets

import matplotlib.pyplot as plt

@interact (st= stock_names)

def plot(st):

    ind = stock_names.index(st)

    # _=stocks_data[ind].set_index('Date')[['Open', 'High', 'Low', 'Close']].plot(kind='line',rot=45) 

    data = stocks_data[ind]

    ax = sns.boxplot(x=data['Date'].dt.month, y=data['Open'])

    plt.xticks(rotation=45)

    plt.show()
from ipywidgets import interact, interactive, fixed, interact_manual

import ipywidgets as widgets

import matplotlib.pyplot as plt

@interact (st= stock_names)

def plot(st):

    ind = stock_names.index(st)

    _=stocks_data[ind][['Open', 'High', 'Low', 'Close']].plot(kind='box') 

from ipywidgets import interact, interactive, fixed, interact_manual

import ipywidgets as widgets

import matplotlib.pyplot as plt

@interact (st= stock_names)

def plot(st):

    ind = stock_names.index(st)

    _=stocks_data[ind][['Open', 'High', 'Low', 'Close']].plot(kind='hist') 
from ipywidgets import interact, interactive, fixed, interact_manual

import ipywidgets as widgets

import matplotlib.pyplot as plt

@interact (st= stock_names)

def plot(st):

    ind = stock_names.index(st)

    _=stocks_data[ind][['Volume']].plot(kind='hist') 
from ipywidgets import interact

import numpy as np

from bokeh.models.annotations import Title

from bokeh.io import push_notebook, show, output_notebook

from bokeh.plotting import figure

from bokeh.models import HoverTool

output_notebook()



x = stocks_data[0]['Date']

y = stocks_data[0]['Open']

t = Title()

t.text = stock_names[0]

TOOLTIPS = [

    

    ("(Date,Open)", "($x, $y)"),

    

]

p = figure(title=t, 

           #plot_height=300, plot_width=600,

           #background_fill_color='#efefef',

           #tooltips=TOOLTIPS,

           x_axis_type='datetime')

hover = HoverTool(tooltips=[ ("(Date,Open)", "($x, $y)"),],

                  formatters=dict(fruits='datetime'))

p.add_tools(hover)



r = p.circle(x, y,  line_width=1.5, alpha=0.8)

p.title = t

p.yaxis.axis_label = 'Open'

def update(st, yaxis):

    ind = stock_names.index(st)

    # r.data_source.data['y'] = A * func(w * x + phi)

#     print(len(stocks_data[ind]['Date']))

#     print(len(stocks_data[ind]['Open']))

    r.data_source.data = {'x':stocks_data[ind]['Date'], 'y':stocks_data[ind][yaxis]}

    t.text = st

    p.title = t

    p.yaxis.axis_label = yaxis

#     r.data_source.data['x'] = stocks_data[ind]['Date']

#     r.data_source.data['y'] = stocks_data[ind]['Open']

    push_notebook()





show(p, notebook_handle=True)

interact(update, st=stock_names, yaxis=['Open', 'High', 'Low', 'Close', 'Volume'])
import numpy as np

from bokeh.models.annotations import Title

from bokeh.io import push_notebook, show, output_notebook

from bokeh.layouts import row

from bokeh.plotting import figure

from math import pi

output_notebook()



x = stocks_data[0]['Date']

y1 = stocks_data[0]['Open']

y2 = stocks_data[0]['High']

y3 = stocks_data[0]['Low']

y4 = stocks_data[0]['Close']

t = Title()

t.text = stock_names[0]



p1 = figure(title=t, 

           plot_height=200, plot_width=200,

           #background_fill_color='#efefef',

           x_axis_type='datetime')

r1 = p1.circle(x, y1,  size=2, color="navy", alpha=0.2)

p1.title = t

p1.yaxis.axis_label = 'Open'

p1.xaxis.major_label_orientation = pi/4



p2 = figure(plot_height=200, plot_width=200, x_axis_type='datetime')

r2 = p2.diamond(x, y2,  size=2, color="firebrick", alpha=0.2)

p2.yaxis.axis_label = 'High'

p2.xaxis.major_label_orientation = pi/4



p3 = figure(plot_height=200, plot_width=200, x_axis_type='datetime')

r3 = p3.square(x, y3,  size=2, color="olive", alpha=0.2)

p3.yaxis.axis_label = 'Low'

p3.xaxis.major_label_orientation = pi/4



p4 = figure(plot_height=200, plot_width=200, x_axis_type='datetime')

r4 = p4.triangle(x, y4,  size=2, color="pink", alpha=0.2)

p4.yaxis.axis_label = 'Close'

p4.xaxis.major_label_orientation = pi/4



p4.x_range = p3.x_range = p2.x_range = p1.x_range

p4.y_range = p3.y_range = p2.y_range = p1.y_range





def update_with_time(st):

    ind = stock_names.index(st)

    # r.data_source.data['y'] = A * func(w * x + phi)

#     print(len(stocks_data[ind]['Date']))

#     print(len(stocks_data[ind]['Open']))

    r1.data_source.data = {'x':stocks_data[ind]['Date'], 'y':stocks_data[ind]['Open']}

    r2.data_source.data = {'x':stocks_data[ind]['Date'], 'y':stocks_data[ind]['High']}

    r3.data_source.data = {'x':stocks_data[ind]['Date'], 'y':stocks_data[ind]['Low']}

    r4.data_source.data = {'x':stocks_data[ind]['Date'], 'y':stocks_data[ind]['Close']}

    t.text = st

    p1.title = t

    #p1.yaxis.axis_label = 'Open'

#     r.data_source.data['x'] = stocks_data[ind]['Date']

#     r.data_source.data['y'] = stocks_data[ind]['Open']

    push_notebook()





show(row(p1, p2, p3, p4), notebook_handle=True)
import time

for s in stock_names:

    #print(s)

    update_with_time(s)

    time.sleep(2)
import numpy as np

from bokeh.models.annotations import Title

from bokeh.io import push_notebook, show, output_notebook

from bokeh.layouts import gridplot

from bokeh.plotting import figure

from math import pi

output_notebook()



x = stocks_data[0]['Date']

y1 = stocks_data[0]['Open']

y2 = stocks_data[0]['High']

y3 = stocks_data[0]['Low']

y4 = stocks_data[0]['Close']

y5 =  stocks_data[0]['Volume']

t = Title()

t.text = stock_names[0]



p1 = figure(title=t, 

           plot_height=200, plot_width=200,

           #background_fill_color='#efefef',

           x_axis_type='datetime')

r1 = p1.circle(x, y1,  size=2, color="navy", alpha=0.2)

p1.title = t

p1.yaxis.axis_label = 'Open'

p1.xaxis.major_label_orientation = pi/4



p2 = figure(plot_height=200, plot_width=200, x_axis_type='datetime')

r2 = p2.diamond(x, y2,  size=2, color="firebrick", alpha=0.2)

p2.yaxis.axis_label = 'High'

p2.xaxis.major_label_orientation = pi/4



p3 = figure(plot_height=200, plot_width=200, x_axis_type='datetime')

r3 = p3.square(x, y3,  size=2, color="olive", alpha=0.2)

p3.yaxis.axis_label = 'Low'

p3.xaxis.major_label_orientation = pi/4



p4 = figure(plot_height=200, plot_width=200, x_axis_type='datetime')

r4 = p4.triangle(x, y4,  size=2, color="pink", alpha=0.2)

p4.yaxis.axis_label = 'Close'

p4.xaxis.major_label_orientation = pi/4



p5 = figure(

     plot_height=200, plot_width=800, 

            x_axis_type='datetime')

r5 = p5.triangle(x, y5,  size=2, color="red", alpha=0.2)

p5.yaxis.axis_label = 'Volume'

p5.xaxis.major_label_orientation = pi/4





p5.x_range = p4.x_range = p3.x_range = p2.x_range = p1.x_range

#p4.y_range = p3.y_range = p2.y_range = p1.y_range





def update_with_time2(st):

    ind = stock_names.index(st)

    # r.data_source.data['y'] = A * func(w * x + phi)

#     print(len(stocks_data[ind]['Date']))

#     print(len(stocks_data[ind]['Open']))

    r1.data_source.data = {'x':stocks_data[ind]['Date'], 'y':stocks_data[ind]['Open']}

    r2.data_source.data = {'x':stocks_data[ind]['Date'], 'y':stocks_data[ind]['High']}

    r3.data_source.data = {'x':stocks_data[ind]['Date'], 'y':stocks_data[ind]['Low']}

    r4.data_source.data = {'x':stocks_data[ind]['Date'], 'y':stocks_data[ind]['Close']}

    r5.data_source.data = {'x':stocks_data[ind]['Date'], 'y':stocks_data[ind]['Volume']}

    t.text = st

    p1.title = t

    #p1.yaxis.axis_label = 'Open'

#     r.data_source.data['x'] = stocks_data[ind]['Date']

#     r.data_source.data['y'] = stocks_data[ind]['Open']

    push_notebook()





show(gridplot([[row([p1, p2, p3, p4])], [row([p5])]], sizing_mode='stretch_width'), notebook_handle=True)
import time

for s in stock_names:

    #print(s)

    update_with_time2(s)

    time.sleep(2)
from bokeh.layouts import column

from bokeh.models import CustomJS, ColumnDataSource, Slider, Select, Dropdown

from bokeh.plotting import Figure, output_file, show



output_file("js_on_change.html")



x = [x*0.005 for x in range(0, 200)]

y = x



source = ColumnDataSource(data=dict(x=x, y=y))



plot = Figure(plot_width=400, plot_height=400)

plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)



callback = CustomJS(args=dict(source=source), code="""

    var data = source.data;

    var f = cb_obj.value

    var x = data['x']

    var y = data['y']

    for (var i = 0; i < x.length; i++) {

        y[i] = Math.pow(x[i], f)

    }

    source.change.emit();

""")



menu = [("Item 1", "item_1"), ("Item 2", "item_2"), None, ("Item 3", "item_3")]

dropdown = Dropdown(label="Dropdown button", button_type="warning", menu=menu)

select = Select(title="Option:", value="foo", options=["foo", "bar", "baz", "quux"])



slider = Slider(start=0.1, end=4, value=1, step=.1, title="power")

slider.js_on_change('value', callback)



layout = column(dropdown, select, slider, plot)



show(layout)
from bokeh.models import ColumnDataSource, Select

from bokeh.plotting import figure

from bokeh.layouts import row, column

from bokeh.io import curdoc

from bokeh.io import output_file, show

from bokeh.io import output_notebook, push_notebook, show

from bokeh.layouts import widgetbox

output_notebook()

# https://stackoverflow.com/questions/55482198/how-to-show-curdoc-dashboard-in-jupyter-notebook

#def make_doc(doc):

# Create ColumnDataSource: source

source = ColumnDataSource(data={

    'x' : stocks_data[0]['Date'],

    'y' : stocks_data[0]['Open']

})

# Create a new plot: plot

plot = figure()

# Add circles to the plot

plot.circle('x', 'y', source=source)

# Create a dropdown Select widget: select    

select = Select(title="Stocks", options=stock_names, value=stock_names[0])

handle = show(column(plot,widgetbox([select])),  notebook_handle=True)

# Define a callback function: update_plot

def update_plot(attr, old, new):

    print(old)

    # If the new Selection is 'female_literacy', update 'y' to female_literacy

#     if new == 'female_literacy':

#         source.data = {

#             'x' : fertility,

#             'y' : female_literacy

#         }

#     # Else, update 'y' to population

#     else:

#         source.data = {

#             'x' : fertility,

#             'y' : population

#         }

   # if old!=new:

    ind = stock_names.index(new)

    source.data = {

        'x' : stocks_data[ind]['Date'],

        'y' : stocks_data[ind]['Open']

    }

    # output_notebook()

#     layout = row(select, plot)

#     curdoc().add_root(layout)

#     handle = show(layout, notebook_handle=True)

#     push_notebook(handle=handle)

   # show(row(plot,widgetbox([select])))

    push_notebook(handle)



# Attach the update_plot callback to the 'value' property of select

select.on_change('value', update_plot)

# Create layout and add to current document

# layout = row(select, plot)

# curdoc().add_root(layout)

# #show(plot, notebook_handle=True)    

# handle = show(layout, notebook_handle=True)

# push_notebook(handle=handle)







import yaml



from bokeh.layouts import column

from bokeh.models import ColumnDataSource, Slider

from bokeh.plotting import figure

from bokeh.themes import Theme

from bokeh.io import show, output_notebook



from bokeh.sampledata.sea_surface_temperature import sea_surface_temperature



output_notebook()



def modify_doc(doc):

    df = sea_surface_temperature.copy()

    source = ColumnDataSource(data=df)



    plot = figure(x_axis_type='datetime', y_range=(0, 25),

                  y_axis_label='Temperature (Celsius)',

                  title="Sea Surface Temperature at 43.18, -70.43")

    plot.line('time', 'temperature', source=source)



    def callback(attr, old, new):

        if new == 0:

            data = df

        else:

            data = df.rolling('{0}D'.format(new)).mean()

        source.data = ColumnDataSource(data=data).data



    slider = Slider(start=0, end=30, value=0, step=1, title="Smoothing by N Days")

    slider.on_change('value', callback)



    doc.add_root(column(slider, plot))



    doc.theme = Theme(json=yaml.load("""

        attrs:

            Figure:

                background_fill_color: "#DDDDDD"

                outline_line_color: white

                toolbar_location: above

                height: 500

                width: 800

            Grid:

                grid_line_dash: [6, 4]

                grid_line_color: white

    """))





show(modify_doc)