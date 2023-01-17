import numpy as np

import pandas as pd

import pickle



import plotly.graph_objects as go

from ipywidgets import widgets
data_path = '../input/ashrae-data-wrangling-csv-to-pickle/'

with open(data_path + 'X_train.pickle', 'rb') as f:

    X_train = pickle.load(f)
def get_data(building_id, meter, xy):

    return X_train[(X_train['building_id']==str(building_id)) &\

                   (X_train['meter']==meter)][xy]



building = widgets.IntText(

    value=1249,

    min=1,

    max=1448,

    step=1,

    description='Building:',

    disabled=False

)



refresh = widgets.Button(

    description='Refresh',

    disabled=False,

    button_style='info',

    tooltip='Refresh',

    icon='refresh'

)



results = widgets.HTML(

    value=''

)



electricity = go.Scatter(x=[],

                         y=[],

                         name='Electricity')



hotwater = go.Scatter(x=[],

                      y=[],

                      name='Hot water')



chilledwater = go.Scatter(x=[],

                          y=[],

                          name='Chilled water')



steam = go.Scatter(x=[],

                   y=[],

                   name='Steam')



g = go.FigureWidget(data=[electricity, hotwater, chilledwater, steam],

                    layout=go.Layout(

                        title=dict(

                            text=f'Energy for building {building.value}'

                        ),

                        xaxis=go.layout.XAxis(

                                rangeselector=dict(

                                    buttons=list([

                                        dict(count=7,

                                             label="1w",

                                             step="day",

                                             stepmode="backward"),

                                        dict(count=1,

                                             label="1m",

                                             step="month",

                                             stepmode="backward"),

                                        dict(count=3,

                                             label="3m",

                                             step="month",

                                             stepmode="backward"),

                                        dict(count=6,

                                             label="6m",

                                             step="month",

                                             stepmode="backward"),

                                        dict(step="all")

                                    ])

                                ),

                                rangeslider=dict(

                                    visible=True

                                ),

                                type="date"

                            ),

                        height=800

                    )

                   )



def validate():

    if 0 <= building.value <= 1448:

        return True

    else:

        return False



    

def response(change):

    if validate():

        

        refresh.button_style = 'warning'

        electricity_ts = get_data(building.value, 'electricity', 'timestamp')

        electricity_reading = get_data(building.value, 'electricity', 'meter_reading')

        hotwater_ts = get_data(building.value, 'hotwater', 'timestamp')

        hotwater_reading = get_data(building.value, 'hotwater', 'meter_reading')

        chilledwater_ts = get_data(building.value, 'chilledwater', 'timestamp')

        chilledwater_reading = get_data(building.value, 'chilledwater', 'meter_reading')

        steam_ts = get_data(building.value, 'steam', 'timestamp')

        steam_reading = get_data(building.value, 'steam', 'meter_reading')

        with g.batch_update():

            g.layout.title.text = f'Energy for building {building.value}'

            g.data[0].x = electricity_ts

            g.data[0].y = electricity_reading

            g.data[1].x = hotwater_ts

            g.data[1].y = hotwater_reading

            g.data[2].x = chilledwater_ts

            g.data[2].y = chilledwater_reading

            g.data[3].x = steam_ts

            g.data[3].y = steam_reading

        zero_nan = pd.DataFrame(columns=['Energy aspect', 'Zero count', 'NaN count'])

        if len(electricity_ts) > 0:

            zero_nan = zero_nan.append({'Energy aspect': 'Electricity',

                                        'Zero count': (electricity_reading == 0).sum(),

                                        'NaN count': 366 * 24 - len(electricity_ts)},

                                       ignore_index=True)

        if len(hotwater_ts) > 0:

            zero_nan = zero_nan.append({'Energy aspect': 'Hot water',

                                        'Zero count': (hotwater_reading == 0).sum(),

                                        'NaN count': 366 * 24 - len(hotwater_ts)},

                                       ignore_index=True)

        if len(chilledwater_ts) > 0:

            zero_nan = zero_nan.append({'Energy aspect': 'Chilled water ',

                                        'Zero count': (chilledwater_reading == 0).sum(),

                                        'NaN count': 366 * 24 - len(chilledwater_ts)},

                                       ignore_index=True)

        if len(steam_ts) > 0:

            zero_nan = zero_nan.append({'Energy aspect': 'Steam ',

                                        'Zero count': (steam_reading == 0).sum(),

                                        'NaN count': 366 * 24 - len(steam_ts)},

                                       ignore_index=True)

        results.value = f"{zero_nan.style.hide_index().set_table_attributes('class=''table''').render()}"            

        refresh.button_style = 'info'



building.observe(response, names='value')

refresh.observe(response, names='value')



response('refresh')



control = widgets.HBox([building, refresh])

widgets.VBox([control,

             g,

             results])