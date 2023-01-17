# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from plotly import tools

init_notebook_mode(connected=True)  

import plotly.figure_factory as ff

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
pitches = pd.read_csv('../input/pitches.csv')

pitches.head()
games = pd.read_csv('../input/games.csv')

games.head()
atbats = pd.read_csv('../input/atbats.csv')

atbats.head()
player_name = pd.read_csv('../input/player_names.csv')

player_name.head()
games.dtypes
atbats.dtypes
pitches.dtypes
pitches['ab_id'] = pitches['ab_id'].astype(int)
player_name.rename(columns={'id': 'batter_id'}, inplace=True)
new_df = pd.merge(pitches, atbats,  how='left', left_on='ab_id', right_on = 'ab_id')

new_df.head()
new_df1 = pd.merge(new_df, games,  how='left', left_on='g_id', right_on = 'g_id')

new_df1.head()
new_df2 = pd.merge(new_df1, player_name,  how='left', left_on='batter_id', right_on = 'batter_id')

new_df2.head()
new_df2['Batters Name'] = new_df2[['first_name', 'last_name']].apply(lambda x: ' '.join(x), axis=1)
new_df2.drop(['first_name', 'last_name'], axis=1, inplace=True)
player_name.rename(columns={'batter_id': 'pitcher_id'}, inplace=True)
final_df = pd.merge(new_df2, player_name,  how='left', left_on='pitcher_id', right_on = 'pitcher_id')

final_df.head()
final_df['Pitchers Name'] = final_df[['first_name', 'last_name']].apply(lambda x: ' '.join(x), axis=1)
final_df.drop(['first_name', 'last_name'], axis=1, inplace=True)
final_df['Pitchers Name'].value_counts()
final_df['pitch_type'] = final_df['pitch_type'].map({'FF': 'Four Seam Fastball', 'SL': 'Slider', 'FT': 'Two seam fastball', 'CH': 'Changeup', 'SI': 'Sinker', 'CU': 'Curveball', 'FC': 'Cutter', 'KC': 'Knuckle Curve', 'FS': 'Splitter','KN': 'Knuckleball', 'EP': 'Eephus', 'FO': 'Pitch Out', 'PO': 'Pitch Out', 'SC': 'Screwball', 'UN': 'Unidentified', 'FA': 'Fastball', 'IN': 'Intentional Ball'})
final_df['code'] = final_df['code'].map({'B': 'Ball', '*B': 'Ball in dirt', 'S': 'Swinging Strike', 'C': 'Called Strike', 'F': 'Foul', 'T': 'Foul Tip', 'L': 'Foul Bunt', 'I': 'Intentional Ball', 'W': 'Blocked','M': 'Missed Bunt', 'P': 'Pitch Out', 'Q': 'Swinging Pitch Out', 'R': 'Foul Pitch Out', 'X': 'In play out(s)', 'D': 'In play no out', 'E': 'In play runs'})
final_df.head()
grp = final_df.groupby(['Pitchers Name'])[["s_count"]].sum()
grp1 = final_df.groupby(['Batters Name'])[["b_count"]].sum()
grp.head()
ERA = grp.s_count / len (final_df)
BA = grp1.b_count / len (final_df) * 100
ERA.sort_values(ascending=False)
BA.sort_values(ascending=False)
Max_Scherzer = final_df[final_df['Pitchers Name'] == 'Max Scherzer']

Max_Scherzer.head()
Max_Scherzer['pitch_type'].value_counts() / len(Max_Scherzer) * 100
Max_Scherzer['event'].value_counts() / len(Max_Scherzer) * 100
size = [20, 40, 60, 80, 100, 80, 60, 40, 20, 40]

data = [dict(

  type = 'scatter',

  x = Max_Scherzer['event'],

  y = Max_Scherzer['pitch_type'],

  mode='markers',

    marker=dict(

        size=size,

        sizemode='area',

        sizeref=2.*max(size)/(40.**2),

        sizemin=4

    ),

    transforms = [dict(

        type = 'groupby',

        groups = Max_Scherzer['pitch_type'],

   

  )]

)]



py.iplot({'data': data}, validate=False)
import random

def random_colors(number_of_colors):

    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

                 for i in range(number_of_colors)]

    return color
trace0 = go.Scatter(

    x = Max_Scherzer.px[Max_Scherzer.pitch_type == 'Four Seam Fastball'],

    y = Max_Scherzer.pz[Max_Scherzer.pitch_type == 'Four Seam Fastball'],

    name = 'Four Seam FastBall',

    mode = 'markers',

    marker = dict(

        size = 10,

        color = 'rgba(152, 0, 0, .8)',

        line = dict(

            width = 2,

            color = 'rgba(152, 0, 0, .8)'

        )

    )

)



trace1 = go.Scatter(

    x = Max_Scherzer.px[Max_Scherzer.pitch_type == 'Slider'],

    y = Max_Scherzer.pz[Max_Scherzer.pitch_type == 'Slider'],

    name = 'Slider',

    mode = 'markers',

    marker = dict(

        size = 10,

        color = 'rgba(22, 0, 0, .8)',

        line = dict(

            width = 2,

            color = 'rgba(22, 0, 0, .8)'

        )

    )

)



trace2 = go.Scatter(

    x = Max_Scherzer.px[Max_Scherzer.pitch_type == 'Changeup'],

    y = Max_Scherzer.pz[Max_Scherzer.pitch_type == 'Changeup'],

    name = 'Changeup',

    mode = 'markers',

    marker = dict(

        size = 10,

        color = 'rgba(224, 0, 0, .8)',

        line = dict(

            width = 2,

            color = 'rgba(224, 0, 0, .8)'

        )

    )

)



trace3 = go.Scatter(

    x = Max_Scherzer.px[Max_Scherzer.pitch_type == 'Curveball'],

    y = Max_Scherzer.pz[Max_Scherzer.pitch_type == 'Curveball'],

    name = 'Curveball',

    mode = 'markers',

    marker = dict(

        size = 10,

        color = 'rgba(22, 1, 0, .8)',

        line = dict(

            width = 2,

            color = 'rgba(22, 1, 0, .8)'

        )

    )

)



trace4 = go.Scatter(

    x = Max_Scherzer.px[Max_Scherzer.pitch_type == 'Cutter'],

    y = Max_Scherzer.pz[Max_Scherzer.pitch_type == 'Cutter'],

    name = 'Cutter',

    mode = 'markers',

    marker = dict(

        size = 10,

        color = 'rgba(2, 1, 0, .8)',

        line = dict(

            width = 2,

            color = 'rgba(2, 1, 0, .8)'

        )

    )

)



trace5 = go.Scatter(

    x = Max_Scherzer.px[Max_Scherzer.pitch_type == 'Two Seam Fastball'],

    y = Max_Scherzer.pz[Max_Scherzer.pitch_type == 'Two Seam Fastball'],

    name = 'Two Seam Fastball',

    mode = 'markers',

    marker = dict(

        size = 10,

        color = 'rgba(222, 1, 0, .8)',

        line = dict(

            width = 2,

            color = 'rgba(222, 1, 0, .8)'

        )

    )

)







data = [trace0, trace1,trace2, trace3, trace4, trace5]



layout = dict(title = 'Pitch types of Max Scherzer ',

              plot_bgcolor='rgb(50,205,50)',

              yaxis = dict(zeroline = False),

              xaxis = dict(zeroline = False)

             )



fig = dict(data=data, layout=layout)

py.iplot(fig, filename='styled-scatter')
ax = sns.lineplot(x="inning", y="start_speed", hue="pitch_type", data=Max_Scherzer)
Justin_Verlander = final_df[final_df['Pitchers Name'] == 'Justin Verlander']

Justin_Verlander.head()
Justin_Verlander['pitch_type'].value_counts() / len(Justin_Verlander) * 100
Justin_Verlander['event'].value_counts() / len(Justin_Verlander) * 100
size = [20, 40, 60, 80, 100, 80, 60, 40, 20, 40]

data = [dict(

  type = 'scatter',

  x = Justin_Verlander['event'],

  y = Justin_Verlander['pitch_type'],

  mode='markers',

    marker=dict(

        size=size,

        sizemode='area',

        sizeref=2.*max(size)/(40.**2),

        sizemin=4

    ),

    transforms = [dict(

        type = 'groupby',

        groups = Justin_Verlander['pitch_type'],

   

  )]

)]



py.iplot({'data': data}, validate=False)
trace0 = go.Scatter(

    x = Justin_Verlander.px[Justin_Verlander.pitch_type == 'Four Seam Fastball'],

    y = Justin_Verlander.pz[Justin_Verlander.pitch_type == 'Four Seam Fastball'],

    name = 'Four Seam FastBall',

    mode = 'markers',

    marker = dict(

        size = 10,

        color = 'rgba(152, 0, 0, .8)',

        line = dict(

            width = 2,

            color = 'rgba(152, 0, 0, .8)'

        )

    )

)



trace1 = go.Scatter(

    x = Justin_Verlander.px[Justin_Verlander.pitch_type == 'Slider'],

    y = Justin_Verlander.pz[Justin_Verlander.pitch_type == 'Slider'],

    name = 'Slider',

    mode = 'markers',

    marker = dict(

        size = 10,

        color = 'rgba(22, 0, 0, .8)',

        line = dict(

            width = 2,

            color = 'rgba(22, 0, 0, .8)'

        )

    )

)



trace2 = go.Scatter(

    x = Justin_Verlander.px[Justin_Verlander.pitch_type == 'Changeup'],

    y = Justin_Verlander.pz[Justin_Verlander.pitch_type == 'Changeup'],

    name = 'Changeup',

    mode = 'markers',

    marker = dict(

        size = 10,

        color = 'rgba(224, 0, 0, .8)',

        line = dict(

            width = 2,

            color = 'rgba(224, 0, 0, .8)'

        )

    )

)



trace3 = go.Scatter(

    x = Justin_Verlander.px[Justin_Verlander.pitch_type == 'Curveball'],

    y = Justin_Verlander.pz[Justin_Verlander.pitch_type == 'Curveball'],

    name = 'Curveball',

    mode = 'markers',

    marker = dict(

        size = 10,

        color = 'rgba(22, 1, 0, .8)',

        line = dict(

            width = 2,

            color = 'rgba(22, 1, 0, .8)'

        )

    )

)



trace4 = go.Scatter(

    x = Justin_Verlander.px[Justin_Verlander.pitch_type == 'Cutter'],

    y = Justin_Verlander.pz[Justin_Verlander.pitch_type == 'Cutter'],

    name = 'Cutter',

    mode = 'markers',

    marker = dict(

        size = 10,

        color = 'rgba(2, 1, 0, .8)',

        line = dict(

            width = 2,

            color = 'rgba(2, 1, 0, .8)'

        )

    )

)



trace5 = go.Scatter(

    x = Justin_Verlander.px[Justin_Verlander.pitch_type == 'Two Seam Fastball'],

    y = Justin_Verlander.pz[Justin_Verlander.pitch_type == 'Two Seam Fastball'],

    name = 'Two Seam Fastball',

    mode = 'markers',

    marker = dict(

        size = 10,

        color = 'rgba(222, 1, 0, .8)',

        line = dict(

            width = 2,

            color = 'rgba(222, 1, 0, .8)'

        )

    )

)







data = [trace0, trace1,trace2, trace3, trace4, trace5]



layout = dict(title = 'Pitch types of Justin Verlander ',

              plot_bgcolor='rgb(50,205,50)',

              yaxis = dict(zeroline = False),

              xaxis = dict(zeroline = False)

             )



fig = dict(data=data, layout=layout)

py.iplot(fig, filename='styled-scatter')
ax = sns.lineplot(x="inning", y="start_speed", hue="pitch_type", data=Justin_Verlander)
data = [

    go.Scatterpolargl(

      r = Justin_Verlander.pz,

      theta = Justin_Verlander.spin_dir,

      mode = "markers",

      name = "Justin Verlander",

      marker = dict(

        color = "rgb(27,158,119)",

        size = 15,

        line = dict(

          color = "white"

        ),

        opacity = 0.7

      )

    ),

    go.Scatterpolargl(

      r = Max_Scherzer.pz,

      theta = Max_Scherzer.spin_dir,

      mode = "markers",

      name = "Max Schrezer",

      marker = dict(

        color = "rgb(217,95,2)",

        size = 20,

        line = dict(

          color = "white"

        ),

        opacity = 0.7

      )

    ),

]



layout = go.Layout(

    title = "Justin Verlander vs Max Scherzer pitch spin",

    font = dict(

      size = 15

    ),

    showlegend = False,

    polar = dict(

      bgcolor = "rgb(223, 223, 223)",

      angularaxis = dict(

        tickwidth = 2,

        linewidth = 3,

        layer = "below traces"

      ),

      radialaxis = dict(

        side = "counterclockwise",

        showline = True,

        linewidth = 2,

        tickwidth = 2,

        gridcolor = "white",

        gridwidth = 2

      )

    ),

    paper_bgcolor = "rgb(223, 223, 223)"

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='polar-webgl')
Joey_Votto = final_df[final_df['Batters Name'] == 'Joey Votto']

Joey_Votto.head()
Joey_Votto['code'].value_counts() / len(Joey_Votto) * 100
Joey_Votto['event'].value_counts() / len(Joey_Votto) * 100
size = [20, 40, 60, 80, 100, 80, 60, 40, 20, 40]

data = [dict(

  type = 'scatter',

  x = Joey_Votto['event'],

  y = Joey_Votto['code'],

  mode='markers',

    marker=dict(

        size=size,

        sizemode='area',

        sizeref=2.*max(size)/(40.**2),

        sizemin=4

    ),

    transforms = [dict(

        type = 'groupby',

        groups = Joey_Votto['code'],

   

  )]

)]



py.iplot({'data': data}, validate=False)
ax = sns.lineplot(x="inning", y="p_score", hue="code", data=Joey_Votto)
trace1 = go.Scatter3d(

    x = Joey_Votto.x[Joey_Votto['event'] == 'Home Run'],

    y = Joey_Votto.y[Joey_Votto['event'] == 'Home Run'],

    z = Joey_Votto.z0[Joey_Votto['event'] == 'Home Run'],

    text = 'Home Run',

    mode = 'markers',

    marker = dict(

        sizemode = 'diameter',

        sizeref = 750, # info on sizeref: https://plot.ly/python/reference/#scatter-marker-sizeref

        color = random_colors(1000),

        )

)

data=[trace1]



layout=go.Layout(width=800, height=800, title = 'Joey Votto Home Run Zone',

              scene = dict(xaxis=dict(title='X axis',

                                      titlefont=dict(color='Orange')),

                            yaxis=dict(title='Y axis',

                                       titlefont=dict(color='rgb(220, 220, 220)')),

                            zaxis=dict(title='Z axis',

                                       titlefont=dict(color='rgb(220, 220, 220)')),

                            bgcolor = 'rgb(50,205,50)'

                           )

             )



fig=go.Figure(data=data, layout=layout)

py.iplot(fig, filename='solar_system_planet_size')
PaulGoldschmidt = final_df[final_df['Batters Name'] == 'Paul Goldschmidt']

PaulGoldschmidt.head()
PaulGoldschmidt['code'].value_counts() / len(PaulGoldschmidt) * 100
PaulGoldschmidt['event'].value_counts() / len(PaulGoldschmidt) * 100
size = [20, 40, 60, 80, 100, 80, 60, 40, 20, 40]

data = [dict(

  type = 'scatter',

  x = PaulGoldschmidt['event'],

  y = PaulGoldschmidt['code'],

  mode='markers',

    marker=dict(

        size=size,

        sizemode='area',

        sizeref=2.*max(size)/(40.**2),

        sizemin=4

    ),

    transforms = [dict(

        type = 'groupby',

        groups = PaulGoldschmidt['code'],

   

  )]

)]



py.iplot({'data': data}, validate=False)
ax = sns.lineplot(x="inning", y="p_score", hue="code", data=PaulGoldschmidt)
trace1 = go.Scatter3d(

    x = PaulGoldschmidt.x[PaulGoldschmidt['event'] == 'Home Run'],

    y = PaulGoldschmidt.y[PaulGoldschmidt['event'] == 'Home Run'],

    z = PaulGoldschmidt.z0[PaulGoldschmidt['event'] == 'Home Run'],

    text = 'Home Run',

    mode = 'markers',

    marker = dict(

        sizemode = 'diameter',

        sizeref = 750, # info on sizeref: https://plot.ly/python/reference/#scatter-marker-sizeref

        color = random_colors(1000),

        )

)

data=[trace1]



layout=go.Layout(width=800, height=800, title = 'Paul Goldschmidt Home Run Zone',

              scene = dict(xaxis=dict(title='X axis',

                                      titlefont=dict(color='Orange')),

                            yaxis=dict(title='Y axis',

                                       titlefont=dict(color='rgb(220, 220, 220)')),

                            zaxis=dict(title='Z axis',

                                       titlefont=dict(color='rgb(220, 220, 220)')),

                            bgcolor = 'rgb(50,205,50)'

                           )

             )



fig=go.Figure(data=data, layout=layout)

py.iplot(fig, filename='solar_system_planet_size')