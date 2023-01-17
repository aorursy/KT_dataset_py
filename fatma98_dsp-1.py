from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML
from matplotlib import rcParams
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation, rc
from IPython.display import HTML
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

data= pd.read_csv('../input/data-after-preprocessing/Processed data.csv',
                         parse_dates=['Date'])

data.head()

#Color, Labels
colors = dict(zip(
    ['North America', 'South America', 'Asia','Australia', 'Africa', 'Europe'],
    ['#adb0ff', '#ffb3ff', '#90d595', '#e48381', '#aafbff', '#f7bb5f']))

group_lk = data.set_index('country')['continent'].to_dict()
#**********************************************************************************************************************
def draw_barchart(year):
       dff = data[data['Date'].eq(year)].sort_values(by='Confirmed', ascending=True).tail(15)
       ax.clear()
       ax.barh(dff['country'], dff['Confirmed'], color=[colors[group_lk[x]] for x in dff['country']])
       dx = dff['Confirmed'].max() / 200
       for i, (value, name) in enumerate(zip(dff['Confirmed'], dff['country'])):
              ax.text(value - dx, i, name, size=10, weight=600, ha='right', va='bottom')
              ax.text(value - dx, i - .25, group_lk[name], size=8, color='#444444', ha='right', va='baseline')
              ax.text(value + dx, i, f'{value:,.0f}', size=10, ha='left', va='center')
       # ... polished styles
       ax.text(1, 0.4, year, transform=ax.transAxes, color='#777777', size=45, ha='right', weight=800)
       ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
       ax.xaxis.set_ticks_position('top')
       ax.tick_params(axis='x', colors='#777777', labelsize=12)
       ax.set_yticks([])
       ax.margins(0, 0.01)
       ax.grid(which='major', axis='x', linestyle='-')
       ax.set_axisbelow(True)
       ax.text(0, 1.12, 'COVID_19 cases through time',
       transform=ax.transAxes, size=12, weight=1000, ha='left')
       plt.box(False)

       
      



x=data["Date"].dt.strftime('%Y-%m-%d').unique()
fig, ax = plt.subplots(figsize=(15, 4))
animator = animation.FuncAnimation(fig, draw_barchart, frames=x)
HTML(animator.to_html5_video())



#********************************** Bubble plot *************************************
dataset=data
x_column = 'Active'
y_column ='Recovered'
bubble_column = 'country'
time_column = 'Date'
size_column = 'Deaths'
#****************************************************************************************
# make list of continents
continents = []
for continent in dataset["continent"]:
    if continent not in continents:
        continents.append(continent)
#*****************************************************************************************
# make figure
fig_dict = {
    "data": [],
    "layout": {},
    "frames": []
}
#******************************************************************************************
# fill in most of layout
# Get the max and min range of both axes
xmin = min(np.log10(dataset[x_column]))*0.2
xmax = max(np.log10(dataset[x_column]))*1.2
ymin = -40000
ymax = max(dataset[y_column])*1.5

# Modify the layout
fig_dict['layout']['xaxis'] = {'title': 'Number of Active cases','type': 'log',
                            'range': [xmin, xmax]}
fig_dict['layout']['yaxis'] = {'title': 'Number of recovered cases',
                             'range': [ymin, ymax]}
fig_dict['layout']['title'] = 'COVID_19 visualization'
fig_dict['layout']['showlegend'] = True
fig_dict['layout']['hovermode'] = 'closest'
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
#**********************************************************************************************************************
sliders_dict = {
    "active": 0,
    "yanchor": "top",
    "xanchor": "left",
    "currentvalue": {
        "font": {"size": 20},
        "prefix": "day:",
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
#**********************************************************************************************************************
# # make data
# Get the days in the dataset
days = dataset[time_column].dt.strftime('%Y-%m-%d').unique()

day = min(days)
for continent in continents:
    dataset_by_year = dataset[dataset["Date"] == day]
    dataset_by_year_and_cont = dataset_by_year[
        dataset_by_year["continent"] == continent]

    data_dict = {
        "x": list(dataset_by_year_and_cont["Active"]),
        "y": list(dataset_by_year_and_cont["Recovered"]),
        "mode": "markers",
        "text": list(dataset_by_year_and_cont["country"]),
        "marker": {
            "sizemode": "area",
            "sizeref": 100,
            "size": list(dataset_by_year_and_cont["Deaths"])
        },
        "name": continent
    }
    fig_dict["data"].append(data_dict)
#*********************************************************************************************************************
# make frames
for day in days:
    frame = {"data": [], "name": str(day)}
    for continent in continents:
        dataset_by_year = dataset[dataset["Date"].dt.strftime('%Y-%m-%d')== str(day)]
        dataset_by_year_and_cont = dataset_by_year[
            dataset_by_year["continent"] == continent]

        data_dict = {
            "x": list(dataset_by_year_and_cont["Active"]),
            "y": list(dataset_by_year_and_cont["Recovered"]),
            "mode": "markers",
            "text": list(dataset_by_year_and_cont["country"]),
            "marker": {
                "sizemode": "area",
                "sizeref": 100,
                "size": list(dataset_by_year_and_cont["Deaths"])
            },
            "name": continent
        }
        frame["data"].append(data_dict)
#**********************************************************************************************************************

    fig_dict["frames"].append(frame)
    slider_step = {"args":[
        [day],
        {"frame": {"duration": 300, "redraw": False},
         "mode": "immediate",
         "transition": {"duration": 300}}
    ],
        "label": str(day),
        "method": "animate"}
    sliders_dict["steps"].append(slider_step)

fig_dict["layout"]["sliders"] = [sliders_dict]

fig = go.Figure(fig_dict)

iplot(fig , config={'scrollzoom': True})

import plotly.express as px
fig = px.choropleth(data, locations=data['country'], locationmode='country names', color=np.log(data["Confirmed"]),
                    hover_name=data['country'], animation_frame=data["Date"].dt.strftime('%Y-%m-%d'),
                    title='Cases over time', color_continuous_scale=px.colors.sequential.Oranges)
fig.update(layout_coloraxis_showscale=False)
fig.show()
dataset=data
x_column2 = 'Deaths'
y_column2 ='Recovered'
bubble_column2 = 'country'
time_column2 = 'Date'
size_column2 = 'Confirmed'
#****************************************************************************************
# make list of continents
continents2 = []
for continent in dataset["continent"]:
    if continent not in continents2:
        continents2.append(continent)
#*****************************************************************************************
# make figure
fig_dict2 = {
    "data": [],
    "layout": {},
    "frames": []
}
#******************************************************************************************
# fill in most of layout
# Get the max and min range of both axes
xmin2 = min(np.log10(dataset[x_column2]))*0.2
xmax2 = max(np.log10(dataset[x_column2]))*1.2
ymin2 = -40000
ymax2 = max(dataset[y_column2])*1.5

# Modify the layout
fig_dict2['layout']['xaxis'] = {'title': 'Number of Death','type': 'log',
                            'range': [xmin2, xmax2]}
fig_dict2['layout']['yaxis'] = {'title': 'Number of recovered',
                             'range': [ymin2, ymax2]}
fig_dict2['layout']['title'] = 'COVID_19 visualization'
fig_dict2['layout']['showlegend'] = False
fig_dict2['layout']['hovermode'] = 'closest'
fig_dict2["layout"]["updatemenus"] = [
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
        "showactive": True,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }
]
#**********************************************************************************************************************
sliders_dict2 = {
    "active": 0,
    "yanchor": "top",
    "xanchor": "left",
    "currentvalue": {
        "font": {"size": 20},
        "prefix": "day:",
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
#**********************************************************************************************************************
# # make data
# Get the days in the dataset
days = dataset[time_column2].dt.strftime('%Y-%m-%d').unique()

day = min(days)
for continent in continents2:
    dataset_by_year = dataset[dataset["Date"] == day]
    dataset_by_year_and_cont = dataset_by_year[
        dataset_by_year["continent"] == continent]

    data_dict2 = {
        "x": list(dataset_by_year_and_cont["Deaths"]),
        "y": list(dataset_by_year_and_cont["Recovered"]),
        "mode": "markers",
        "text": list(dataset_by_year_and_cont["country"]),
        "marker": {
            "sizemode": "area",
            "sizeref": 100,
            "size": list(dataset_by_year_and_cont["Confirmed"])
        },
        "name": continent
    }
    fig_dict2["data"].append(data_dict2)
#*********************************************************************************************************************
# make frames
for day in days:
    frame = {"data": [], "name": str(day)}
    for continent in continents2:
        dataset_by_year = dataset[dataset["Date"].dt.strftime('%Y-%m-%d')== str(day)]
        dataset_by_year_and_cont = dataset_by_year[
            dataset_by_year["continent"] == continent]

        data_dict2 = {
            "x": list(dataset_by_year_and_cont["Deaths"]),
            "y": list(dataset_by_year_and_cont["Recovered"]),
            "mode": "markers",
            "text": list(dataset_by_year_and_cont["country"]),
            "marker": {
                "sizemode": "area",
                "sizeref": 500,
                "size": list(dataset_by_year_and_cont["Confirmed"])
            },
            "name": continent
        }
        frame["data"].append(data_dict2)
#**********************************************************************************************************************

    fig_dict2["frames"].append(frame)
    slider_step = {"args":[
        [day],
        {"frame": {"duration": 300, "redraw": False},
         "mode": "immediate",
         "transition": {"duration": 300}}
    ],
        "label": str(day),
        "method": "animate"}
    sliders_dict2["steps"].append(slider_step)

fig_dict2["layout"]["sliders"] = [sliders_dict2]

fig2 = go.Figure(fig_dict2)

iplot(fig2 , config={'scrollzoom': True})

