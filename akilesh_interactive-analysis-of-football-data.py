import sqlite3

import pandas as pd



# Establish the connection to the db

cnx = sqlite3.connect('../input/database.sqlite')
# Loading the players table

players = pd.read_sql_query("SELECT * from Player", cnx)



# some basic validation

print(len(players['player_api_id']))

print(len(players['player_api_id'].unique()))
player_attributes = pd.read_sql_query("SELECT * from Player_Attributes", cnx)

player_attributes.head()
player_attributes['date'] = pd.to_datetime(player_attributes['date'])

player_attribute_dates = player_attributes[['id', 'player_api_id', 'date']]
pd.options.mode.chained_assignment = None

player_attribute_dates["rank"] = player_attribute_dates.groupby("player_api_id")["date"].rank(method="first", ascending=False)

player_attribute_dates = player_attribute_dates[player_attribute_dates['rank'] == 1.0]
assert len(player_attributes['player_api_id'].unique()) == len(player_attribute_dates['player_api_id'])

player_attributes.head()
players_attrs = player_attribute_dates.merge(player_attributes, on=['id', 'player_api_id', 'date'], how='left')

player_info = pd.merge(players_attrs, players, on=['player_api_id', 'player_fifa_api_id'], how='left')

player_info.head()
%%capture

!jupyter lab clean

!pip install ipywidgets

!jupyter nbextension enable --py widgetsnbextension
%matplotlib notebook

%matplotlib inline

import seaborn as sns

from ipywidgets import *

import numpy as np

import functools

import matplotlib.pyplot as plt
required_columns = ['player_name', 'height', 'weight']

required_numeric_columns = ['overall_rating',

       'potential', 'crossing', 'finishing', 'heading_accuracy',

       'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',

       'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',

       'long_passing', 'ball_control', 'acceleration', 'sprint_speed',

       'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',

       'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',

       'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',

       'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',

       'gk_reflexes']
def conjunction(*conditions):

    return functools.reduce(np.logical_or, conditions)
def get_desc():

    return """

This is an interactive dashboard to visualize the football dataset. 

The output is a simple table with the player name, height and weight.



The default value for all the sliders are set to 0.



<h4>How to use this dashboard?</h4>

<h7> Let us say you want to know the list of players whose crossing rating is above 90 and whose overall rating is above 70.

To find that, move the slider for crossing to 90 and the slider for overall rating to 70. The table below will change dynamically.

The table is sorted in descending order with the column/label that has the maximum value. In this example, the output is

sorted in descending order with the "crossing" column followed by the "overall rating" column.



By default the output table will show the player_name, height, weight and overall_rating. You can add more columns to show by 

choosing the columns in the multiple selection box using the "ctrl" or "shift" key.

"""
def common_function(data):

    required_columns = ['player_name', 'height', 'weight']

    columns_to_display = required_columns + [column for column in data['columns_to_show']]

    del data['columns_to_show']

    if 'filter_by' in data.keys():

        del data['filter_by']

    columns = data.keys()

    comps = [player_info[column] > data[column] for column in columns]

    result = comps[0]

    for comp in comps[1:]:

        result &= comp

    df = player_info[result]

    re_order_numeric_columns = [item[0] for item in sorted(data.items(), key=lambda x:x[1], reverse=True)]

    df.sort_values(re_order_numeric_columns, ascending=False,inplace=True)

    return df

def plot(**data):

    columns_to_display = required_columns + [column for column in data['columns_to_show']]

    df = common_function(data)

    display(df[columns_to_display])

        

def height_plot(**data):

    df = common_function(data)

    plt.figure(figsize=(8, 3))

    chart = sns.countplot(data=df, x='height')

    chart.set_xticklabels(chart.get_xticklabels(), rotation=65, horizontalalignment='right')

    title = f"Height Distribution"

    chart.set_title(title)

    

def weight_plot(**data):

    df = common_function(data)

    plt.figure(figsize=(8, 3))

    chart = sns.countplot(data=df, x='weight')

    chart.set_xticklabels(chart.get_xticklabels(), rotation=65, horizontalalignment='right')

    title = f"Weight Distribution"

    chart.set_title(title)





sliders = {}

plt.style.use('seaborn')

%config InlineBackend.figure_format = 'svg'



style = {'description_width': 'initial'}

for column in required_numeric_columns:

    sliders[column] = IntSlider(description=f'{column}', min=0, max=100, step=1, value=0, style=style)

slider_displays = widgets.VBox(list(sliders.values()))



columns_to_show = widgets.SelectMultiple(

    options=required_numeric_columns,

    value=['overall_rating'],

    rows=4,

    description='Columns',

    disabled=False,

    layout=widgets.Layout(border='1px solid black')

)





columns_to_show_checkboxes = [widgets.Checkbox(value=False, description=column, disabled=False,  style=style) 

                              for column in required_numeric_columns]



sliders['columns_to_show'] = columns_to_show



dashboard_desc = get_desc()





title = widgets.HTML(

    value="<H2 style=\"font-family:Verdana\"><center>Interactive visualization of the Football dataset</center></H2>",

)

description = widgets.HTML(

    value=f"<p style=\"font-family:Arial\">{dashboard_desc}</p><br>",

)



columns_to_show_title = widgets.HTML(

    value="<H5 style=\"font-family:Verdana\"><left>Columns to show</left></H5>",

)

break_widget = widgets.HTML(

    value="<br>",

)



out = Output(layout=Layout(border='1px solid black'))



plot_output = widgets.interactive_output(plot, sliders)

with out:

    display(plot_output)

    

h_pixels = '200px'

avg_height_plot = widgets.interactive_output(height_plot, sliders)

h_out = Output(layout=Layout(margin_left="0px",

    border='1px solid black',                       

    align_items='stretch'))

with h_out:

    display(avg_height_plot)



avg_weight_plot = widgets.interactive_output(weight_plot, sliders)

w_out = Output(layout=Layout(margin_left="0px",

    border='1px solid black',

    align_items='stretch'))

with w_out:

    display(avg_weight_plot)



column_1 = VBox([slider_displays])

column_2 = VBox([HBox([out]), h_out, w_out], layout=Layout(margin_left="0px",

    border='1px solid black'))

line_break = widgets.Output(layout={'border': '1px solid black'})

title_widget = widgets.HBox([title])

description_widget = widgets.HBox([description])

dashboard = widgets.VBox([

    title_widget,

    description_widget,

    line_break,

    columns_to_show,

    HBox([column_1, column_2])])

display(dashboard)