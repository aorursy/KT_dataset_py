# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/wv-counties-covid19-data/IncidRtLam_20-Oct-2020.csv')
df['R_minus'] = df['R_mean'] - df['R_low']

df['R_plus'] = df['Rhigh'] - df['R_mean']
import plotly.express as px

import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x = df['R_mean'],

                                y = df['dailycases_p100k_7d_avg'],

                                error_x=dict(type='data',

                                            symmetric=False,

                                            array=df['R_plus'],

                                            arrayminus=df['R_minus']),

                                    mode='markers',

                                    name='Counties',

                                    marker=dict(size=df['inf_potential_p100k'],

                                                color=df['population']),

                                    hovertext = df['county_name']

                               )

               )

fig.update_layout(

        hoverlabel=dict(

                        bgcolor="white",

                        font_size=16,

                        font_family="Rockwell",

        ),

    font_family="Ariel",

    font_color='Black',

    title={

        'text': 'WV Counties',

        'font_size' : 20,

        'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    xaxis_title={

        'text' : "R rate",

        'font_size' : 16

    },

    yaxis_title={

        'text' : "Daily Cases per 100k (7 Day Average)",

        'font_size' : 16

    },

)

fig.show()
fig = px.scatter(df, x="R_mean", y="dailycases_p100k_7d_avg",

                 hover_name="county_name", hover_data=["R_mean", "dailycases_p100k_7d_avg"],

                 error_x="R_plus", error_x_minus="R_minus",

                 color = 'population',

                 size = 'inf_potential_p100k',

                 title = 'WV Counties',

                 labels={

                     "R_mean": "R rate (mean)",

                     "dailycases_p100k_7d_avg": "Daily cases per 100k (7 day average)",

                     "inf_potential_p100k" : "Inf Potential per 100k",

                     "population" : "Population"

                 },)

fig.update_layout(

                autosize=False,

                width=800,

                height=1000,

                hoverlabel=dict(

                        bgcolor="white",

                        font_size=16,

                        font_family="Rockwell",

        ),

                title={

                    'y':0.9,

                    'x':0.5,

                    'xanchor': 'center',

                    'yanchor': 'top'},

                coloraxis = {'showscale' : False})

fig.update_layout(showlegend=False)



fig.show()
fig = go.Figure(data=go.Scatter(x = df['R_mean'],

                                y = df['dailycases_p100k_7d_avg'],

                                error_x=dict(type='data',

                                            symmetric=False,

                                            array=df['R_plus'],

                                            arrayminus=df['R_minus']

                                            ),

                                    mode='markers',

                                    marker=dict(size=df['inf_potential_p100k'],

                                                color=df['population']

                                               ),

                                    hovertemplate = df['county_name'] +

                                            '<br>R Rate (mean): %{x:.2f}'+

                                            '<br>Daily Cases: %{y}<br><extra></extra>',

                               )

               )

fig.update_layout(

        autosize=False,

        width=900,

        height=1100,

        hoverlabel=dict(

                        bgcolor="white",

                        font_size=16,

                        font_family="Rockwell",

        ),

    font_family="Ariel",

    font_color='Black',

    title={

        'text': 'WV Counties',

        'font_size' : 20,

        'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    xaxis_title={

        'text' : "R rate",

        'font_size' : 16

    },

    yaxis_title={

        'text' : "Daily Cases per 100k (7 Day Average)",

        'font_size' : 16

    },

)

fig.show()
fig = go.Figure(data=go.Scatter(x = df['R_mean'],

                                y = df['dailycases_p100k_7d_avg'],

                                error_x=dict(type='data',

                                            symmetric=False,

                                            array=df['R_plus'],

                                            arrayminus=df['R_minus']

                                            ),

                                    mode='markers',

                                    marker=dict(size=df['inf_potential_p100k'] + 10,

                                                color=df['population']

                                               ),

                                    hovertemplate = df['county_name'] +

                                            '<br>R Rate (mean): %{x:.2f}'+

                                            '<br>Daily Cases: %{y}<br><extra></extra>',

                               )

               )

fig.update_layout(

        autosize=False,

        width=800,

        height=1000,

        hoverlabel=dict(

                        bgcolor="white",

                        font_size=16,

                        font_family="Rockwell",

        ),

    font_family="Ariel",

    font_color='Black',

    title={

        'text': 'WV Counties',

        'font_size' : 20,

        'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    xaxis_title={

        'text' : "R rate",

        'font_size' : 16

    },

    yaxis_title={

        'text' : "Daily Cases per 100k (7 Day Average)",

        'font_size' : 16

    },

                shapes = [dict(

                            type = 'line',

                            x0 = -0.2,

                            x1 = 5,

                            y0 = 25,

                            y1 = 25,

                            line = dict(

                            color = 'Black',

                            dash = 'dash'))

                        ],

                 annotations=[

             dict(text="High Risk",x = 4, y=25)

                 ],

             showlegend=False

            )





fig.show()