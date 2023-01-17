# default

import numpy as np 

import pandas as pd



# visualization

import missingno as msno

import altair as alt



# util

import os

import warnings

warnings.filterwarnings("ignore")
alt.renderers.enable('kaggle')
%%time

PATH = "/kaggle/input/ufcdata/"

data = pd.read_csv(PATH+'data.csv')
print(data.shape)
pd.options.display.max_columns = None # for see all columns

data.head(3)
col = data.columns

RedCol, BlueCol, PlayCol = [], [], []



for i in col:

    if 'R_' == i[:2]: RedCol.append(i)

    elif 'B_' in i[:2]: BlueCol.append(i)

    else : PlayCol.append(i)

print(f"Red Fighter Info : {len(RedCol)}\nBlue Fighter Info : {len(BlueCol)}\nPlay Info : {len(PlayCol)}")
play_info = data[PlayCol]



play_info.head()
play_info['year'] = play_info['date'].apply(lambda x : x.split('-')[0])

play_counts = pd.DataFrame(play_info['year'].value_counts().sort_index())

play_counts['count'] = play_counts['year']

play_counts['year'] = play_counts.index





play_counts.head()
alt.Chart(play_counts).mark_area(

    color="lightblue",

).encode(

    x='year',

    y='count'

)

alt.Chart(play_counts).mark_area(

    color="lightblue",

    interpolate='step-after',

    line=True

).encode(

    x='year',

    y='count'

)

weight_class_count = pd.DataFrame(data['weight_class'].value_counts())

weight_class_count['class'] = weight_class_count.index



alt.Chart(weight_class_count).mark_bar(

    color="#564d8d"

).encode(

    y='class',

    x='weight_class'

)

class_bar = alt.Chart(weight_class_count).mark_bar(

    color="#564d8d"

).encode(

    y='class',

    x='weight_class'

)



class_text = class_bar.mark_text(

    align='left',

    baseline='middle',

    dx=3

).encode(

    text='weight_class'

)



(class_bar + class_text)

class_bar = alt.Chart(weight_class_count).mark_bar(

    color="#564d8d"

).encode(

    x='class',

    y='weight_class'

)



rule = alt.Chart(weight_class_count).mark_rule(color='red').encode(

    y='mean(weight_class)'

)



(class_bar + rule).properties(width=600)

print(len([i for i in RedCol if 'att' in i]))

print([i for i in RedCol if 'att' in i])
# Unfortunately, Altair can visualize max 5000 points

# I will visualize latest 5000 matches

alt.Chart(data[-5000:]).mark_circle(size=10).encode(

    x='R_avg_BODY_att',

    y='B_avg_BODY_att',

    color='Winner',

).properties(

    width=500, 

    height=500,

    title='Average Body Attack'

).interactive()
brush = alt.selection(type='interval')

base = alt.Chart(data[-5000:]).add_selection(brush)



points = base.mark_point(opacity=0.8).encode(

    x='R_avg_HEAD_att',

    y='B_avg_HEAD_att',

    color='Winner',

).properties(

    width=500, 

    height=500,

    title='Average Head Attack'

)



# Configure the ticks

tick_axis = alt.Axis(labels=False, domain=False, ticks=False)



x_ticks = base.mark_tick().encode(

    alt.X('R_avg_HEAD_att', axis=tick_axis),

    alt.Y('Winner', title='', axis=tick_axis),

    color=alt.condition(brush, 'Winner', alt.value('lightgrey'))

).properties(

    width=500, 

)



y_ticks = base.mark_tick().encode(

    alt.X('Winner', title='', axis=tick_axis),

    alt.Y('B_avg_HEAD_att', axis=tick_axis),

    color=alt.condition(brush, 'Winner', alt.value('lightgrey'))

).properties(

    height=500

)



# Build the chart

y_ticks | (points & x_ticks )
play_winner = play_info.groupby('year')['Winner'].value_counts().reset_index(name='counts')



# Check for Data Shape

display(play_winner.head())



# 1. Stacked Area Chart



alt.Chart(play_winner).mark_area().encode(

    x='year',

    y='counts',

    color='Winner'

).properties(

    width=800,

    title='Red or Blue, Who Win the GAME?'

)



# 2. Normalized Stacked Area Chart



alt.Chart(play_winner).mark_area().encode(

    x='year',

    y=alt.Y("counts", stack="normalize"),

    color='Winner'

).properties(

    width=600,

    title='Red or Blue, Who Win the GAME?'

)



# 3. Streamgraph



alt.Chart(play_winner).mark_area().encode(

    alt.X('year:T',axis=alt.Axis(format='%Y', domain=False, tickSize=0)),

    alt.Y('counts:Q', stack='center', axis=None),

    alt.Color('Winner')

).properties(

    width=600

)
# 4. Draw in col or row



alt.Chart(play_winner).mark_area().encode(

    x='year',

    y='counts',

    color='Winner',

    column='Winner' # you can change this as row

)
# 5. No Stacked & Set Opacity



alt.Chart(play_winner).mark_area(opacity=0.5).encode(

    x='year',

    y=alt.Y('counts:Q', stack=None),

    color='Winner',

).properties(

    width=600

)


