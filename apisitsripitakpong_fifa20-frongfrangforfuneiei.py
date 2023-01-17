# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Data Manipulation 

import numpy as np

import pandas as pd



# Visualization 

import matplotlib.pyplot as plt

import seaborn as sns



#datetime

import datetime as dt



#warnings

import warnings

warnings.filterwarnings("ignore")





#plotly

import plotly.graph_objects as go

import plotly.figure_factory as ff

import plotly.express as px

from plotly.subplots import make_subplots
fifa=pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_20.csv')

fifa.head()
fifa.dob=pd.to_datetime(fifa.dob)
fifa_potential=fifa[(fifa.potential>85 )& (fifa.overall<80)]

fifa_potential.head()
fifa_potential_ready=fifa_potential[(fifa_potential.overall<80)&(fifa_potential.overall>70)]

fifa_potential.head()
position="GK"

fifa_potential_ready_GK=fifa_potential_ready[fifa.player_positions.str.contains(position)]

fifa_potential_ready_GK
fig = go.Figure(data=[

            go.Bar(name='overall', x=fifa_potential_ready_GK.short_name, y=fifa_potential_ready_GK.overall,text=fifa_potential_ready_GK.overall,textposition='auto'),

            go.Bar(name='potential', x=fifa_potential_ready_GK.short_name, y=fifa_potential_ready_GK.potential,text=fifa_potential_ready_GK.potential,textposition='auto')

            ])

fig.update_layout(title='Top potential GK in FIFA 20',

                   xaxis_title='player name ',

                   yaxis_title='Rating')

fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"

fig.data[1].marker.line.width = 1

fig.data[1].marker.line.color = "black"

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=fifa_potential_ready_GK.short_name, y=fifa_potential_ready_GK.value_eur,

                    mode='lines+markers',

                    ))

fig.update_layout(title=' Top Potential GK player values(Euros)',

                   xaxis_title='player name ',

                   yaxis_title='Value')

fig.show()
position="LB"

position1="LWB"

fifa_potential_LB=fifa_potential_ready[(fifa.player_positions.str.contains(position))|(fifa.player_positions.str.contains(position1))]

fifa_potential_LB
fig = go.Figure(data=[

            go.Bar(name='overall', x=fifa_potential_LB.short_name, y=fifa_potential_LB.overall,text=fifa_potential_LB.overall,textposition='auto'),

            go.Bar(name='potential', x=fifa_potential_LB.short_name, y=fifa_potential_LB.potential,text=fifa_potential_LB.potential,textposition='auto')

            ])

fig.update_layout(title='Top potential LB in FIFA 20 ',

                   xaxis_title='player name ',

                   yaxis_title='Rating')

fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"

fig.data[1].marker.line.width = 1

fig.data[1].marker.line.color = "black"

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=fifa_potential_LB.short_name, y=fifa_potential_LB.value_eur,

                    mode='lines+markers',

                    ))

fig.update_layout(title=' Top Potential LB player values(Euros)',

                   xaxis_title='player name ',

                   yaxis_title='Value')

fig.show()
position="RB"

position1="RWB"

fifa_potential_RB=fifa_potential_ready[(fifa.player_positions.str.contains(position))|(fifa.player_positions.str.contains(position1))]
fifa_potential_RB
fig = go.Figure(data=[

            go.Bar(name='overall', x=fifa_potential_RB.short_name, y=fifa_potential_RB.overall,text=fifa_potential_RB.overall,textposition='auto'),

            go.Bar(name='potential', x=fifa_potential_RB.short_name, y=fifa_potential_RB.potential,text=fifa_potential_RB.potential,textposition='auto')

            ])

fig.update_layout(title='Top potential RB in FIFA 20 ',

                   xaxis_title='player name ',

                   yaxis_title='Rating')

fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"

fig.data[1].marker.line.width = 1

fig.data[1].marker.line.color = "black"

fig.show() 
fig = go.Figure()

fig.add_trace(go.Scatter(x=fifa_potential_RB.short_name, y=fifa_potential_RB.value_eur,

                    mode='lines+markers',

                    ))

fig.update_layout(title=' Top Potential RB player values(Euros)',

                   xaxis_title='player name ',

                   yaxis_title='Value')

fig.show()
position="CB"

fifa_potential_cb=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fifa_potential_cb
fig = go.Figure(data=[

            go.Bar(name='overall', x=fifa_potential_cb.short_name, y=fifa_potential_cb.overall,text=fifa_potential_cb.overall,textposition='auto'),

            go.Bar(name='potential', x=fifa_potential_cb.short_name, y=fifa_potential_cb.potential,text=fifa_potential_cb.potential,textposition='auto')

            ])

fig.update_layout(title='Top potential CB in FIFA 20 ',

                   xaxis_title='player name ',

                   yaxis_title='Rating')

fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"

fig.data[1].marker.line.width = 1

fig.data[1].marker.line.color = "black"

fig.show() 
fig = go.Figure()

fig.add_trace(go.Scatter(x=fifa_potential_cb.short_name, y=fifa_potential_cb.value_eur,

                    mode='lines+markers',

                    ))

fig.update_layout(title=' Top Potential CB player values(Euros)',

                   xaxis_title='player name ',

                   yaxis_title='Value')

fig.show()
position="CM"

fifa_potential_cm=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fifa_potential_cm
fig = go.Figure(data=[

            go.Bar(name='overall', x=fifa_potential_cm.short_name, y=fifa_potential_cm.overall,text=fifa_potential_cm.overall,textposition='auto'),

            go.Bar(name='potential', x=fifa_potential_cm.short_name, y=fifa_potential_cm.potential,text=fifa_potential_cm.potential,textposition='auto')

            ])

fig.update_layout(title='Top potential CM in FIFA 20 ',

                   xaxis_title='player name ',

                   yaxis_title='Rating')

fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"

fig.data[1].marker.line.width = 1

fig.data[1].marker.line.color = "black"

fig.show() 
fig = go.Figure()

fig.add_trace(go.Scatter(x=fifa_potential_cm.short_name, y=fifa_potential_cm.value_eur,

                    mode='lines+markers',

                    ))

fig.update_layout(title=' Top Potential CM player values(Euros)',

                   xaxis_title='player name ',

                   yaxis_title='Value')

fig.show()
position="CAM"

fifa_potential_cam=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fifa_potential_cam
fig = go.Figure(data=[

            go.Bar(name='overall', x=fifa_potential_cam.short_name, y=fifa_potential_cam.overall,text=fifa_potential_cam.overall,textposition='auto'),

            go.Bar(name='potential', x=fifa_potential_cam.short_name, y=fifa_potential_cam.potential,text=fifa_potential_cam.potential,textposition='auto')

            ])

fig.update_layout(title='Top potential CAM in FIFA 20 ',

                   xaxis_title='player name ',

                   yaxis_title='Rating')

fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"

fig.data[1].marker.line.width = 1

fig.data[1].marker.line.color = "black"

fig.show() 
fig = go.Figure()

fig.add_trace(go.Scatter(x=fifa_potential_cam.short_name, y=fifa_potential_cam.value_eur,

                    mode='lines+markers',

                    ))

fig.update_layout(title=' Top Potential CAM player values(Euros)',

                   xaxis_title='player name ',

                   yaxis_title='Value')

fig.show()
position="CDM"

fifa_potential_cdm=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fifa_potential_cdm
fig = go.Figure(data=[

            go.Bar(name='overall', x=fifa_potential_cdm.short_name, y=fifa_potential_cdm.overall,text=fifa_potential_cdm.overall,textposition='auto'),

            go.Bar(name='potential', x=fifa_potential_cdm.short_name, y=fifa_potential_cdm.potential,text=fifa_potential_cdm.potential,textposition='auto')

            ])

fig.update_layout(title='Top potential CDM in FIFA 20 ',

                   xaxis_title='player name ',

                   yaxis_title='Rating')

fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"

fig.data[1].marker.line.width = 1

fig.data[1].marker.line.color = "black"

fig.show() 
fig = go.Figure()

fig.add_trace(go.Scatter(x=fifa_potential_cdm.short_name, y=fifa_potential_cdm.value_eur,

                    mode='lines+markers',

                    ))

fig.update_layout(title=' Top Potential CDM player values(Euros)',

                   xaxis_title='player name ',

                   yaxis_title='Value')

fig.show()
position="LW"

fifa_potential_lw=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fifa_potential_lw
fig = go.Figure(data=[

            go.Bar(name='overall', x=fifa_potential_lw.short_name, y=fifa_potential_lw.overall,text=fifa_potential_lw.overall,textposition='auto'),

            go.Bar(name='potential', x=fifa_potential_lw.short_name, y=fifa_potential_lw.potential,text=fifa_potential_lw.potential,textposition='auto')

            ])

fig.update_layout(title='Top potential LW in FIFA 20 ',

                   xaxis_title='player name ',

                   yaxis_title='Rating')

fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"

fig.data[1].marker.line.width = 1

fig.data[1].marker.line.color = "black"

fig.show() 
fig = go.Figure()

fig.add_trace(go.Scatter(x=fifa_potential_lw.short_name, y=fifa_potential_lw.value_eur,

                    mode='lines+markers',

                    ))

fig.update_layout(title=' Top Potential LW player values(Euros)',

                   xaxis_title='player name ',

                   yaxis_title='Value')

fig.show()
position="RW"

fifa_potential_rw=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fifa_potential_rw
fig = go.Figure(data=[

            go.Bar(name='overall', x=fifa_potential_rw.short_name, y=fifa_potential_rw.overall,text=fifa_potential_rw.overall,textposition='auto'),

            go.Bar(name='potential', x=fifa_potential_rw.short_name, y=fifa_potential_rw.potential,text=fifa_potential_rw.potential,textposition='auto')

            ])

fig.update_layout(title='Top potential RW in FIFA 20 ',

                   xaxis_title='player name ',

                   yaxis_title='Rating')

fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"

fig.data[1].marker.line.width = 1

fig.data[1].marker.line.color = "black"

fig.show() 
fig = go.Figure()

fig.add_trace(go.Scatter(x=fifa_potential_rw.short_name, y=fifa_potential_rw.value_eur,

                    mode='lines+markers',

                    ))

fig.update_layout(title=' Top Potential RW player values(Euros)',

                   xaxis_title='player name ',

                   yaxis_title='Value')

fig.show()
position="ST"

fifa_potential_st=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fifa_potential_st
fig = go.Figure(data=[

            go.Bar(name='overall', x=fifa_potential_st.short_name, y=fifa_potential_st.overall,text=fifa_potential_st.overall,textposition='auto'),

            go.Bar(name='potential', x=fifa_potential_st.short_name, y=fifa_potential_st.potential,text=fifa_potential_st.potential,textposition='auto')

            ])

fig.update_layout(title='Top potential ST in FIFA 20 ',

                   xaxis_title='player name ',

                   yaxis_title='Rating')

fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"

fig.data[1].marker.line.width = 1

fig.data[1].marker.line.color = "black"

fig.show() 
fig = go.Figure()

fig.add_trace(go.Scatter(x=fifa_potential_st.short_name, y=fifa_potential_st.value_eur,

                    mode='lines+markers',

                    ))

fig.update_layout(title=' Top Potential ST player values(Euros)',

                   xaxis_title='player name ',

                   yaxis_title='Value')

fig.show()
position="CF"

fifa_potential_cf=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fifa_potential_cf
fig = go.Figure(data=[

            go.Bar(name='overall', x=fifa_potential_cf.short_name, y=fifa_potential_cf.overall,text=fifa_potential_cf.overall,textposition='auto'),

            go.Bar(name='potential', x=fifa_potential_cf.short_name, y=fifa_potential_cf.potential,text=fifa_potential_cf.potential,textposition='auto')

            ])

fig.update_layout(title='Top potential CF in FIFA 20 ',

                   xaxis_title='player name ',

                   yaxis_title='Rating')

fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"

fig.data[1].marker.line.width = 1

fig.data[1].marker.line.color = "black"

fig.show() 
fig = go.Figure()

fig.add_trace(go.Scatter(x=fifa_potential_cf.short_name, y=fifa_potential_cf.value_eur,

                    mode='lines+markers',

                    ))

fig.update_layout(title=' Top Potential CF player values(Euros)',

                   xaxis_title='player name ',

                   yaxis_title='Value')

fig.show()