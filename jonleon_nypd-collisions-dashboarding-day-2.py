import numpy as np

import pandas as pd



import plotly.plotly as py

import plotly.graph_objs as go



from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)



my_style = 'mapbox://styles/jonleon/cjpwcy37n21mx2rnzxq5uob4i'

mapbox_access_token = 'pk.eyJ1Ijoiam9ubGVvbiIsImEiOiJjanB1eGJjYzYwYmZpNGFsZW9oa2hxZGJzIn0.Bs58Ngdyl3BZ4CTCXTJ0BA'



import warnings

warnings.filterwarnings('ignore')



collisions = pd.read_csv('../input/nypd-motor-vehicle-collisions.csv')



# Convert to datetime format

collisions['date_parsed'] = pd.to_datetime(collisions['DATE'], format="%Y-%m-%d")

collisions_by_month = collisions['date_parsed'].value_counts().resample('m').sum()



locations = go.Scattermapbox(

    lat=collisions['LATITUDE'][:10001],

    lon=collisions['LONGITUDE'][:10001],

    mode='markers',

    marker=dict(

        size=4,

        color='gold',

        opacity=0.8

    ),

    text=('Date: '+collisions['date_parsed'][:10001].astype(str)+

          '</br>Injured: '+collisions['NUMBER OF PERSONS INJURED'][:10001].astype(str)+

          '</br>Killed: '+collisions['NUMBER OF PERSONS KILLED'][:10001].astype(str)

    ),

    name='locations'

)



dates = go.Scatter(

    x=collisions_by_month.index,

    y=collisions_by_month.values,

    line=dict(color='gold'),

    name='dates'

)



data = [locations, dates]



layout = dict(

    title='NYPD Motor Vehicle Collisions',

    titlefont=dict(

        size=20,

        family="Raleway, Roman, Arial"

    ),

    autosize=True,

    hovermode='closest',

    mapbox=dict(

        domain=dict(x = [0, 0.55],

#                     y= [0, 0.9]

        ),

        accesstoken=mapbox_access_token,

        bearing=0,

        center=dict(

            lat=40.7,

            lon=-73.9

        ),

        pitch=0,

        zoom=8.5,

        style=my_style,

    ),

    xaxis = dict(

        domain = [0.6, 1]

    ),

#     yaxis = dict(

#         domain = [0, 0.9]

#     )

)



annotations =  [

    {

      "x": 0.3, 

      "y": 1.0, 

      "font": {"size": 12, "family":"Raleway, Roman, Arial"}, 

      "showarrow": False, 

      "text": "10K Most Recent", 

      "xanchor": "center", 

      "xref": "paper", 

      "yanchor": "bottom", 

      "yref": "paper"

    }, 

    {

      "x": 0.8, 

      "y": 1.0, 

      "font": {"size": 12, "family":"Raleway, Roman, Arial"}, 

      "showarrow": False, 

      "text": "By Month", 

      "xanchor": "center", 

      "xref": "paper", 

      "yanchor": "bottom", 

      "yref": "paper"

    }

]



layout['annotations'] = annotations



fig = dict(data=data, layout=layout)

fig['layout'].update(showlegend=False, plot_bgcolor='black', paper_bgcolor='black', font=dict(color= 'white'))



iplot(fig)