# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/all_stocks_5yr.csv')

df['date'] = pd.to_datetime( df['date'])
from bokeh.io import output_notebook, push_notebook, show

from bokeh.plotting import figure, ColumnDataSource

from bokeh.layouts import widgetbox
df.columns
output_notebook()
companies = {'Apple':'AAPL',

             'Google':'GOOG',

             'Amazon':'AMZN'}

initVal = 'Amazon'

df_init = df[ df['Name'] == companies[initVal]]

dataSrc = ColumnDataSource( data = df_init)



TOOLS = "xbox_select,lasso_select,pan,help"

plot1 = figure( title='Volume', tools=TOOLS, x_axis_type='datetime', x_axis_label='Date',

               y_axis_label='Volume', active_drag="xbox_select",

              plot_width=800, plot_height=200)

plot1.line( x='date', y='volume', source=dataSrc, color='teal')

# For brushing behavior with xbox_select tool, the line plot doesn't work - need a scatterplot

plot1.circle( x='date', y='volume', source=dataSrc, color='teal', size=2, alpha=.3)



plot2 = figure( title='Hi - Low', tools=TOOLS, x_axis_type='datetime', x_axis_label='Date',

               y_axis_label='Value', active_drag="xbox_select",

              plot_width=800, plot_height=400)

#plot2.circle( x='date', y='low', source=dataSrc, color='dodgerblue', size=8, alpha=.3)

plot2.line( x='date', y='low', source=dataSrc, color='dodgerblue', alpha=.3)

plot2.line( x='date', y='high', source=dataSrc, color='coral', alpha=.3)

plot2.x_range = plot2.x_range
# This callback is used with the "interact" function rather than "on_change", which supposedly requires bokeh server

def update( company):

    dataSrc.data = df[ df['Name'] == company]

    push_notebook() # reflect changes in the ui
list(zip( companies.keys(), companies.values()))
from ipywidgets import interactive_output

import ipywidgets as widgets

widgetSelect = widgets.Dropdown(

    options=list( zip( companies.keys(), companies.values())),

    value=companies[initVal],

    description='Company:',

    disabled=False

    )

select_co = interactive_output( update, {"company":widgetSelect})

#select_co = interactive( update, company=companies.keys())
from bokeh.layouts import gridplot, layout, row, column

display( widgetSelect)

show( gridplot([[plot1], [plot2]]), notebook_handle=True)