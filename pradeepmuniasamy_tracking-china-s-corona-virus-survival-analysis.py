!pip install pygmt
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import geopandas as gpd



from bokeh.plotting import output_notebook, figure, show

from bokeh.models import ColumnDataSource, Div, Select, Button, ColorBar, CustomJS

from bokeh.layouts import row, column, layout

from bokeh.transform import cumsum, linear_cmap

from bokeh.palettes import Blues8, Spectral3

from bokeh.plotting import figure, output_file, show





output_notebook()





import plotly.graph_objects as go

import plotly.express as px

import plotly.offline as py

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)



import gc # Garbage Collector
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
corona_df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")

death_df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_2019_ncov_deaths.csv")

confirmed_df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_2019_ncov_confirmed.csv")

recovered_df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_2019_ncov_recovered.csv")
corona_df.head()
death_df.head()
confirmed_df.head()
recovered_df.head()
dates = corona_df.Date.unique()

confirmed_df = confirmed_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date', value_name='Confirmed')

death_df = death_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date', value_name='Death')

recovered_df = recovered_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date', value_name='Recovered')
Corona_df = pd.DataFrame()

Corona_df = pd.concat([confirmed_df, death_df['Death'], recovered_df['Recovered']], axis=1, sort=False)

Corona_df = Corona_df.fillna(0)



## Splitting Date and Time into Data and Time

Corona_df['Time'] = pd.to_datetime(Corona_df['Date']).dt.time

Corona_df['Date'] = pd.to_datetime(Corona_df['Date']).dt.date
Corona_df
Corona_df.info()
Disease_through_Country = pd.DataFrame()

Disease_through_Country = Corona_df.groupby(["Country/Region"]).sum().reset_index()

Disease_through_Country = Disease_through_Country.drop(['Lat','Long'],axis=1)
Names = ["Confirmed","Death","Recovered"]

for i in Names:

    Disease_through_Country[i+"_percentage"] = Disease_through_Country[i]/Disease_through_Country[Names].sum(axis=1)*100

    Disease_through_Country[i+"_angle"] = Disease_through_Country[i+"_percentage"]/100 * 2*np.pi
Disease_through_Country_plot = pd.DataFrame({'class': ["Confirmed","Death","Recovered"],

                                              'percent': [float('nan'), float('nan'), float('nan')],

                                              'angle': [float('nan'), float('nan'), float('nan')],

                                              'color': [ '#718dbf', '#e84d60','#c9d9d3']})

Disease_through_Country_plot
# Create the ColumnDataSource objects "s2" and "s2_plot"

s2 = ColumnDataSource(Disease_through_Country)

s2_plot = ColumnDataSource(Disease_through_Country_plot)



# Create the Figure object "p2"

p2 = figure(plot_width=275, plot_height=350, y_range=(-0.5, 0.7),toolbar_location=None, tools=['hover'], tooltips='@percent{0.0}%')



# Add circular sectors to "p2"

p2.wedge(x=0, y=0, radius=0.8, source=s2_plot,start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),fill_color='color', line_color=None, legend='class')



# Change parameters of "p2"

p2.axis.visible = False

p2.grid.grid_line_color = None

p2.legend.orientation = 'horizontal'

p2.legend.location = 'top_center'



# Create the custom JavaScript callback

callback2 = CustomJS(args=dict(s2=s2, s2_plot=s2_plot), code='''

    var ang = ['Confirmed_angle', 'Death_angle','Recovered_percentage'];

    var per = ['Confirmed_percentage',  'Death_percentage','Recovered_percentage'];

    if (cb_obj.value != 'Please choose...') {

        var disease = s2.data['Country/Region'];

        var ind = disease.indexOf(cb_obj.value);

        for (var i = 0; i < ang.length; i++) {

            s2_plot.data['angle'][i] = s2.data[ang[i]][ind];

            s2_plot.data['percent'][i] = s2.data[per[i]][ind];



        }

    }

    else {

        for (var i = 0; i < ang.length; i++) {

            s2_plot.data['angle'][i] = undefined;

            s2_plot.data['percent'][i] = undefined;

        }



    }

    s2_plot.change.emit();

''')



# When changing the value of the dropdown menu execute "callback2"

options = ['Please choose...'] + list(s2.data['Country/Region'])

select = Select(title='Country ', value=options[0], options=options)

select.js_on_change('value', callback2)



# Display "select" and "p2" as a column

show(column(select, p2))
Disease_through_Country = pd.DataFrame()

Disease_through_Country = Corona_df.groupby(["Country/Region","Province/State"]).sum().reset_index()

Disease_through_Country = Disease_through_Country.drop(['Lat','Long'],axis=1)

Disease_through_Country = Disease_through_Country.loc[Disease_through_Country["Country/Region"]=="Mainland China"]







Names = ["Confirmed","Death","Recovered"]

for i in Names:

    Disease_through_Country[i+"_percentage"] = Disease_through_Country[i]/Disease_through_Country[Names].sum(axis=1)*100

    Disease_through_Country[i+"_angle"] = Disease_through_Country[i+"_percentage"]/100 * 2*np.pi

    

    

Disease_through_Country_plot = pd.DataFrame({'class': ["Confirmed","Death","Recovered"],

                                              'percent': [float('nan'), float('nan'), float('nan')],

                                              'angle': [float('nan'), float('nan'), float('nan')],

                                              'color': [ '#718dbf', '#e84d60','#c9d9d3']})

Disease_through_Country_plot
# Create the ColumnDataSource objects "s2" and "s2_plot"

s2 = ColumnDataSource(Disease_through_Country)

s2_plot = ColumnDataSource(Disease_through_Country_plot)



# Create the Figure object "p2"

p2 = figure(plot_width=275, plot_height=350, y_range=(-0.5, 0.7),toolbar_location=None, tools=['hover'], tooltips='@percent{0.0}%')



# Add circular sectors to "p2"

p2.wedge(x=0, y=0, radius=0.8, source=s2_plot,start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),fill_color='color', line_color=None, legend='class')



# Change parameters of "p2"

p2.axis.visible = False

p2.grid.grid_line_color = None

p2.legend.orientation = 'horizontal'

p2.legend.location = 'top_center'



# Create the custom JavaScript callback

callback2 = CustomJS(args=dict(s2=s2, s2_plot=s2_plot), code='''

    var ang = ['Confirmed_angle', 'Death_angle','Recovered_percentage'];

    var per = ['Confirmed_percentage',  'Death_percentage','Recovered_percentage'];

    if (cb_obj.value != 'Please choose...') {

        var disease = s2.data['Province/State'];

        var ind = disease.indexOf(cb_obj.value);

        for (var i = 0; i < ang.length; i++) {

            s2_plot.data['angle'][i] = s2.data[ang[i]][ind];

            s2_plot.data['percent'][i] = s2.data[per[i]][ind];



        }

    }

    else {

        for (var i = 0; i < ang.length; i++) {

            s2_plot.data['angle'][i] = undefined;

            s2_plot.data['percent'][i] = undefined;

        }



    }

    s2_plot.change.emit();

''')



# When changing the value of the dropdown menu execute "callback2"

options = ['Please choose...'] + list(s2.data['Province/State'])

select = Select(title='Regions of China', value=options[0], options=options)

select.js_on_change('value', callback2)



# Display "select" and "p2" as a column

show(column(select, p2))
Data = Corona_df.groupby("Date").sum()

source = ColumnDataSource(Data)



p = figure(x_axis_type='datetime')



p.line(x='Date', y='Confirmed', line_width=2, source=source, legend_label='Confirmed Corona Cases')

p.line(x='Date', y='Death', line_width=2, source=source, color=Spectral3[1], legend_label='Death by Corona')

p.line(x='Date', y='Recovered', line_width=2, source=source, color=Spectral3[2], legend_label='Recovered from Corona')



p.yaxis.axis_label = 'Kilotons of Munitions Dropped'



show(p)