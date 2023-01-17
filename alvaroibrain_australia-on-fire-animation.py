import numpy as np

import pandas as pd



import plotly.graph_objects as go

from kaggle_secrets import UserSecretsClient



user_secrets = UserSecretsClient()

MAPBOX_TOKEN = user_secrets.get_secret("mapbox_token")

MAPBOX_STYLE = user_secrets.get_secret("mapbox_style")
fires = pd.read_csv('../input/fires-from-space-australia-and-new-zeland/fire_archive_M6_96619.csv')
def to_hour_str(number):

    string = '{:04d}'.format(number)

    string = string[:2] + ':' + string[2:]

    return string



fires['acq_datestring'] = fires.apply(lambda r: r['acq_date'] + " " + to_hour_str(r['acq_time']), axis=1)
fires.head()
times = fires.groupby(['acq_date'])['acq_date'].count().index.tolist()

frames_data = [fires.loc[fires['acq_date'] == t] for t in times]
frames= [go.Frame(data=[go.Densitymapbox(lat=f['latitude'], lon=f['longitude'], z=f['brightness'], radius=10)], name=str(f.iloc[0]['acq_date'])) for f in frames_data]
buttons=[

         dict(label="Play",method="animate",args=[None, {'fromcurrent':True, "transition": {"duration": 30, "easing": "quadratic-in-out"}}]),

         dict(label="Pause",method="animate",args=[[None], {"frame": {"duration": 0, "redraw": False},"mode": "immediate", "transition": {"duration": 0}}])

]





sliders_dict = {

    'active':0,

    'currentvalue': dict(font=dict(size=16), prefix='Time: ', visible=True),

    "transition": {"duration": 300, "easing": "cubic-in-out"},

    'x': 0,

    'steps': []

}



for i,t in enumerate(times):

    slider_step = {"args": [

                        [t],

                        {"frame": {"duration": 300, "redraw": False},

                         #"mode": "immediate",

                         "transition": {"duration": 30, "easing": "quadratic-in-out"}}

                    ],

            "label": t,

            "method": "animate",

            "value": t

    }

    sliders_dict['steps'].append(slider_step)

    
fig = go.Figure(data = [go.Densitymapbox(lat=fires['latitude'], lon=fires['longitude'], z=fires['brightness'], radius=1, colorscale='Hot', zmax=400, zmin=0)],

               layout=go.Layout(updatemenus=[dict(type="buttons", buttons=buttons,showactive=True)] ), 

               frames=frames

)



fig.update_layout(mapbox_style=MAPBOX_STYLE, 

                  mapbox_accesstoken=MAPBOX_TOKEN,

                  mapbox_center_lon=135,

                  mapbox_center_lat=-25.34,

                  mapbox_zoom=3.5)



"""fig.update_layout(mapbox_style="stamen-terrain", 

                  mapbox_center_lon=135,

                  mapbox_center_lat=-25.34,

                  mapbox_zoom=3.5)"""





fig.update_layout(sliders=[sliders_dict],

                 title='Australia fires over time')



fig.update_layout(height=850)

fig.show()