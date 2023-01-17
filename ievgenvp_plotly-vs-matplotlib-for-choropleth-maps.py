import numpy as np

import pandas as pd

import plotly.graph_objs as go



from IPython.display import HTML

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
states_dict = {'01': "AL", '02': "AK", "04": "AZ", "05": "AR",

              "06": "CA", "08": "CO", "09": "CT", "10": "DE",

              "11": "DC", "12": "FL", "13": "GA", "15": "HI",

              "16": "ID", "17": "IL", "18": "IN", "19": "IA",

              "20": "KS", "21": "KY", "22": "LA", "23": "ME",

              "24": "MD", "25": "MA", "26": "MI", "27": "MN",

              "28": "MS", "29": "MO", "30": "MT", "31": "NE",

              "32": "NV", "33": "NH", "34": "NJ", "35": "NM",

              "36": "NY", "37": "NC", "38": "ND", "39": "OH",

              "40": "OK", "41": "OR", "42": "PA", "44": "RI",

              "45": "SC", "46": "SD", "47": "TN", "48": "TX",

              "49": "UT", "50": "VT", "51": "VA", "53": "WA",

              "54": "WV", "55": "WI", "56": "WY"}



years = ['2012', '2013', '2014','2015','2020',

         '2025', '2030', '2035', '2040', '2045']
states_df = pd.read_csv('../input/FAF4.4_State.csv',

                         dtype = {'dms_orig': str, 'dms_dest': str})
states_df['dms_orig'].replace(states_dict, inplace = True)

states_df['dms_dest'].replace(states_dict, inplace = True)
# domestic origin by value 2012-2045 

dom_origin_df = states_df.loc[pd.isnull(states_df['fr_orig'])]



# domestic destination by value 2012-2045 

dom_dest_df = states_df.loc[pd.isnull(states_df['fr_dest'])]



# dataframe for freight balance (outflow from regoin minus inflow to region)

dom_origin_bal_df = dom_origin_df[['dms_orig','value_2012',

                                'value_2013', 'value_2014',

                                'value_2015', 'value_2020',

                                'value_2025', 'value_2030',

                                'value_2035', 'value_2040',

                                'value_2045']].groupby('dms_orig', as_index = True).sum()



dom_dest_bal_df = dom_dest_df[['dms_dest','value_2012',

                                'value_2013', 'value_2014',

                                'value_2015', 'value_2020',

                                'value_2025', 'value_2030',

                                'value_2035', 'value_2040',

                                'value_2045']].groupby('dms_dest', as_index = True).sum()



dom_dest_bal_df = dom_dest_bal_df.apply(lambda x: x*(-1))



balance_df = dom_origin_bal_df.add(dom_dest_bal_df, fill_value = 0.0)

balance_df.reset_index(inplace = True)

balance_df.columns = ['state']+years
scl = [[0.0, 'rgb(84,39,143)'], [0.1, 'rgb(117,107,177)'], [0.2, 'rgb(158,154,200)'],

       [0.3, 'rgb(188,189,220)'], [0.4, '218,218,235)'], [0.5, 'rgb(240,240,240)'],

       [0.6, 'rgb(255,214,151)'],[0.8, 'rgb(250,195,104)'], [0.9, 'rgb(250,177,58)'],

       [1.0, 'rgb(252,153,6)']]



data_bal = []



data_2012 = [dict(type='choropleth',

                colorscale = scl,

                autocolorscale = False,

                locations = balance_df['state'],

                z = balance_df['2012'].astype(float)/1000,

                locationmode = 'USA-states',

                text = balance_df['state'],

                marker = dict(line = dict(color = 'rgb(255,255,255)',

                                          width = 2)),

                visible = True,

                colorbar = dict(title = "Billions USD"))]

    

data_bal.extend(data_2012)



for i in years[1:]:

    data_upd = [dict(type='choropleth',

                      colorscale = scl,

                      autocolorscale = False,

                      locations = balance_df['state'],

                      z = balance_df[i].astype(float)/1000,

                      locationmode = 'USA-states',

                      text = balance_df['state'],

                      marker = dict(line = dict(color = 'rgb(255,255,255)',

                                                width = 2)),

                      visible = False,

                      colorbar = dict(title = "Billions USD"))]

    

    data_bal.extend(data_upd)





# set menues inside the plot

steps = []

yr = 0

for i in range(0,len(data_bal)):

    step = dict(method = "restyle",

                args = ["visible", [False]*len(data_bal)],

                label = years[yr]) 

    step['args'][1][i] = True

    steps.append(step)

    yr += 1



sliders = [dict(active = 10,

                currentvalue = {"prefix": "Year: "},

                pad = {"t": 50},

                steps = steps)]



# Set the layout

layout = dict(title = 'Production / consumption balance per state',

              geo = dict(scope='usa',

                         projection=dict( type='albers usa' ),

                         showlakes = True,

                         lakecolor = 'rgb(255, 255, 255)'),

              sliders = sliders)



fig = dict(data=data_bal, layout=layout)

iplot( fig, filename='d3-cloropleth-map')