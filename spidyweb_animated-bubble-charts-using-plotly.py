# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd 

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



import plotly.express as px

import plotly.graph_objects as go

import math





from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd



data = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv'

                     , encoding='latin1')



data.head()
print("year : ", data['year'].unique())
print("state : ", data['state'].unique())
print("month : ", data['month'].unique())
#creating a dictionary with translations of months

month_map={'Janeiro': 'January', 'Fevereiro': 'February', 'Março': 'March', 'Abril': 'April', 'Maio': 'May',

          'Junho': 'June', 'Julho': 'July', 'Agosto': 'August', 'Setembro': 'September', 'Outubro': 'October',

          'Novembro': 'November', 'Dezembro': 'December'}

#mapping our translated months

data['month']=data['month'].map(month_map)

#checking the month column for the second time after the changes were made

data.month.unique()
data.head()
# Define the dataset and the columns

dataset = data

x_column = 'month'

y_column = 'number'

bubble_column = 'state'

time_column = 'year'
# Get the years in the dataset

years = dataset[time_column].unique()



# Make the grid

grid = pd.DataFrame()

col_name_template = '{year}+{header}_grid'

for year in years:

    dataset_by_year = dataset[(dataset['year'] == int(year))]

    for col_name in [x_column, y_column, bubble_column]:

        # Each column name is unique

        temp = col_name_template.format(

            year=year, header=col_name

        )

        if dataset_by_year[col_name].size != 0:

            grid = grid.append({'value': list(dataset_by_year[col_name]), 'key': temp}, 

                               ignore_index=True)



grid.sample(10)


# Define figure

figure = {

    'data': [],

    'layout': {},

    'frames': []

}



# Get the earliest year

year = min(years)



# Make the trace

trace = {

    'x': grid.loc[grid['key']==col_name_template.format(

        year=year, header=x_column

    ), 'value'].values[0], 

    'y': grid.loc[grid['key']==col_name_template.format(

        year=year, header=y_column

    ), 'value'].values[0],

    'mode': 'markers',

    'text': grid.loc[grid['key']==col_name_template.format(

        year=year, header=bubble_column

    ), 'value'].values[0]

}

# Append the trace to the figure

figure['data'].append(trace)



# Plot the figure

iplot(figure, config={'scrollzoom': True})




# Modify the layout

figure['layout']['xaxis'] = {'title': 'Month'}   

figure['layout']['yaxis'] = {'title': 'Fires in numbers'} 

figure['layout']['title'] = 'Forest Fires in Brazil(1998-2018)'

figure['layout']['showlegend'] = False

figure['layout']['hovermode'] = 'closest'

iplot(figure, config={'scrollzoom': True})
#Adding animated time frames



#Next we add frames for each year resulting in an animated graph, though not interactive yet.



for year in years:

    # Make a frame for each year

    frame = {'data': [], 'name': str(year)}

    

    # Make a trace for each frame

    trace = {

        'x': grid.loc[grid['key']==col_name_template.format(

            year=year, header=x_column

        ), 'value'].values[0],

        'y': grid.loc[grid['key']==col_name_template.format(

            year=year, header=y_column

        ), 'value'].values[0],

        'mode': 'markers',

        'text': grid.loc[grid['key']==col_name_template.format(

            year=year, header=bubble_column

        ), 'value'].values[0],

        'type': 'scatter'

    }

    # Add trace to the frame

    frame['data'].append(trace)

    # Add frame to the figure

    figure['frames'].append(frame) 



iplot(figure, config={'scrollzoom': True})



#The animation happened only once, right after executing the code. To be able to make it interactive, we add a slider bar for the time.
#Adding Play and Pause Button

figure['layout']['updatemenus'] = [

    {

        'buttons': [

            {

                'args': [None, {'frame': {'duration': 500, 'redraw': False},

                         'fromcurrent': True, 'transition': {'duration': 300, 

                                                             'easing': 'quadratic-in-out'}}],

                'label': 'Play',

                'method': 'animate'

            },

            {

                'args': [[None], {'frame': {'duration':0, 'redraw': False}, 'mode': 'immediate',

                'transition': {'duration': 0}}],

                'label': 'Pause',

                'method': 'animate'

            }

        ],

        'direction': 'left',

        'pad': {'r': 10, 't': 87},

        'showactive': False,

        'type': 'buttons',

        'x': 0.1,

        'xanchor': 'right',

        'y': 0,

        'yanchor': 'top'

    }

]

iplot(figure, config={'scrollzoom': True})
import plotly.graph_objects as go



dataset = data



years =  ["1998", "1999", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011",

 "2012", "2013", "2014", "2015", "2016", "2017"]



# make list of states

states = []

for state in dataset["state"]:

    if state not in states:

        states.append(state)

# make figure

fig_dict = {

    "data": [],

    "layout": {},

    "frames": []

}



# fill in most of layout

fig_dict["layout"]["xaxis"] = {"title": "Month"}

fig_dict["layout"]["yaxis"] = {"title": "Fires in Numbers"}

fig_dict["layout"]["hovermode"] = "closest"

fig_dict["layout"]["sliders"] = {

    "args": [

        "transition", {

            "duration": 400,

            "easing": "cubic-in-out"

        }

    ],

    "initialValue": "1998",

    "plotlycommand": "animate",

    "values": years,

    "visible": True

}

fig_dict["layout"]["updatemenus"] = [

    {

        "buttons": [

            {

                "args": [None, {"frame": {"duration": 500, "redraw": False},

                                "fromcurrent": True, "transition": {"duration": 300,

                                                                    "easing": "quadratic-in-out"}}],

                "label": "Play",

                "method": "animate"

            },

            {

                "args": [[None], {"frame": {"duration": 0, "redraw": False},

                                  "mode": "immediate",

                                  "transition": {"duration": 0}}],

                "label": "Pause",

                "method": "animate"

            }

        ],

        "direction": "left",

        "pad": {"r": 10, "t": 87},

        "showactive": False,

        "type": "buttons",

        "x": 0.1,

        "xanchor": "right",

        "y": 0,

        "yanchor": "top"

    }

]



sliders_dict = {

    "active": 0,

    "yanchor": "top",

    "xanchor": "left",

    "currentvalue": {

        "font": {"size": 20},

        "prefix": "Year:",

        "visible": True,

        "xanchor": "right"

    },

    "transition": {"duration": 300, "easing": "cubic-in-out"},

    "pad": {"b": 10, "t": 50},

    "len": 0.9,

    "x": 0.1,

    "y": 0,

    "steps": []

}



# make data

year = 1998

for state in states:

    dataset_by_year = dataset[dataset["year"] == year]

    dataset_by_year_and_cont = dataset_by_year[

        dataset_by_year["state"] == state]



    data_dict = {

        "x": list(dataset_by_year_and_cont["month"]),

        "y": list(dataset_by_year_and_cont["number"]),

        "mode": "markers",

        "text": list(dataset_by_year_and_cont["state"]),

        "marker": {

            "sizemode": "area",

            "sizeref": 2,

            "size": list(dataset_by_year_and_cont["number"]

            )

        },

        "name": state

    }

    fig_dict["data"].append(data_dict)



# make frames

for year in years:

    frame = {"data": [], "name": str(year)}

    for state in states:

        dataset_by_year = dataset[dataset["year"] == int(year)]

        dataset_by_year_and_cont = dataset_by_year[

            dataset_by_year["state"] == state]



        data_dict = {

            "x": list(dataset_by_year_and_cont["month"]),

            "y": list(dataset_by_year_and_cont["number"]),

            "mode": "markers",

            "text": list(dataset_by_year_and_cont["state"]),

            "marker": {

                "sizemode": "area",

                "sizeref": 2,

            "size": list(dataset_by_year_and_cont["number"]

                )

            },

            "name": state

        }

        frame["data"].append(data_dict)



    fig_dict["frames"].append(frame)

    slider_step = {"args": [

        [year],

        {"frame": {"duration": 300, "redraw": False},

         "mode": "immediate",

         "transition": {"duration": 400}}

    ],

        "label": year,

        "method": "animate"}

    sliders_dict["steps"].append(slider_step)





fig_dict["layout"]["sliders"] = [sliders_dict]



fig = go.Figure(fig_dict)



fig.show()


