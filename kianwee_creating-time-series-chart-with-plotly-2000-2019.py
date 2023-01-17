import pandas as pd

import numpy as np

import plotly

import plotly.graph_objects as go

data = pd.read_csv('../input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv')

data.head()

data= data.replace('ND', np.nan)

data = data.dropna()

data
country_lst = list(data.columns[2:])

type(country_lst[1])
updatemenu_lst = []

for a,b in enumerate(country_lst):

    visible = [False] * len(country_lst)

    visible[0] = True

visible
# Initialise figure 

fig = go.Figure()

fig.update_yaxes(automargin=True)

# Add Traces



for k,v in enumerate(country_lst):

    colour_lst = ['#91930b', '#6cdc93', '#935049', '#acbc09', '#0b92d3', '#dc8845', '#a60c7c', '#4a31f7', '#d8191c', '#e86f71','#efd4f3','#2e0e88','#7d4c26','#0bc039','#fa378c','#54f1e5','#7a0a8b','#43142d','#beaef4','#04b919','#91dde5','#2a850d']

    fig.add_trace(

    go.Scatter(x= data['Time Serie'],

                    y= data[country_lst[k]],

                    name= country_lst[k],

                    line=dict(color=colour_lst[k])))

    

    

fig.update_layout(

    updatemenus=[

        dict(

            active=0,

            buttons=list([

                dict(label= 'AUS/US',

                     method="update",

                     args=[{"visible": [True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]},

                           {"title": country_lst[0]}]),

                dict(label='EUR/US',

                     method="update",

                     args=[{"visible": [False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]},

                           {"title": country_lst[1]}]),

                dict(label='NZ/US',

                     method="update",

                     args=[{"visible": [False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]},

                           {"title": country_lst[2]}]),

                dict(label='UK/US',

                     method="update",

                     args=[{"visible": [False, False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]},

                           {"title": country_lst[3]}]),

                dict(label='BRA/US',

                     method="update",

                     args=[{"visible": [False, False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]},

                           {"title": country_lst[4]}]),

                dict(label='CAN/US',

                     method="update",

                     args=[{"visible": [False, False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]},

                           {"title": country_lst[5]}]),

                dict(label='CHINA/US',

                     method="update",

                     args=[{"visible": [False, False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]},

                           {"title": country_lst[6]}]),

                dict(label='HK/US',

                     method="update",

                     args=[{"visible": [False, False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False]},

                           {"title": country_lst[7]}]),

                dict(label='INDIA/US',

                     method="update",

                     args=[{"visible": [False, False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False]},

                           {"title": country_lst[8]}]),

                dict(label='KOR/US',

                     method="update",

                     args=[{"visible": [False, False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False]},

                           {"title": country_lst[9]}]),

                dict(label='MEX/US',

                     method="update",

                     args=[{"visible": [False, False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False]},

                           {"title": country_lst[10]}]),

                dict(label='SOUTHAFRICA/US',

                     method="update",

                     args=[{"visible": [False, False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False]},

                           {"title": country_lst[11]}]),

                dict(label='SG/US',

                     method="update",

                     args=[{"visible": [False, False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False]},

                           {"title": country_lst[12]}]),

                dict(label='DEN/US',

                     method="update",

                     args=[{"visible": [False, False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False]},

                           {"title": country_lst[13]}]),

                dict(label='JAP/US',

                     method="update",

                     args=[{"visible": [False, False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False]},

                           {"title": country_lst[14]}]),

                dict(label='MY/US',

                     method="update",

                     args=[{"visible": [False, False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False]},

                           {"title": country_lst[15]}]),

                dict(label='NOR/US',

                     method="update",

                     args=[{"visible": [False, False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False]},

                           {"title": country_lst[16]}]),

                dict(label='SWE/US',

                     method="update",

                     args=[{"visible": [False, False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False]},

                           {"title": country_lst[17]}]),

                dict(label='SRI/US',

                     method="update",

                     args=[{"visible": [False, False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False]},

                           {"title": country_lst[18]}]),

                dict(label='CHE/US',

                     method="update",

                     args=[{"visible": [False, False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False]},

                           {"title": country_lst[19]}]),

                dict(label='TAI/US',

                     method="update",

                     args=[{"visible": [False, False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False]},

                           {"title": country_lst[20]}]),

                dict(label='THAI/US',

                     method="update",

                     args=[{"visible": [False, False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True]},

                           {"title": country_lst[21]}]),

                dict(label='Asia & Oceania',

                     method="update",

                     args=[{"visible": [True, False,True,False,False,False,True,True,True,True,False,False,True,False,True,True,False,False,True,False,True,True]},

                           {"title": 'Asia & Oceania'}]),

                dict(label='Europe',

                     method="update",

                     args=[{"visible": [False, True,False,True,False,False,False,False,False,False,False,False,False,True,False,False,True,True,False,True,False,False]},

                           {"title": 'Europe'}]),

            

            ]),

        )

    ])
