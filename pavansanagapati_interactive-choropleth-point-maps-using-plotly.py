# Import Libraries
import numpy as np 
import pandas as pd 
#Import plotly libraries
import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
import plotly
import plotly.plotly as py
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
plotly.offline.init_notebook_mode(connected=True)
# Let us use the dataset available at https://www.4shared.com/s/fvepo3gOAei for this map
states= pd.read_csv('../input/us-states/States.csv')
states.columns = ['code','region','pop','SATV','SATM','percent','dollars','pay']
states['text']='SATv '+ states['SATV'].astype(str)+'SATm '+ states['SATM'].astype(str) +'<br>'+'State '+ states['code']
data = [dict(type='choropleth',autocolorscale=False, locations = states['code'], z=states['dollars'],
             locationmode='USA-states',text = states['text'], colorscale = 'Viridis', colorbar = dict(title='thousand dollars'))]
layout = dict(title='State Spending on Public Education in $k/student',
              geo = dict(scope='usa',projection = dict(type ='albers usa'),
                        showlakes =True,lakecolor='rgb(66,165,245)'))

layout
plotly.offline.iplot({
    "data": data,
    "layout": layout
})
# Let us use the dataset available at https://www.4shared.com/s/fjUlUogqqei for this map

snow= pd.read_csv('../input/snow-inventory/snow_inventory.csv')

snow.columns = ['stn_id','lat','long','elev','code']

snow.head()
snow.shape
snow_sample = snow.sample(n=400,random_state=25,axis=0)
snow_sample.head()
data1 = [dict(type='scattergeo',lat = snow_sample['lat'],lon = snow_sample['long'],
             marker = dict(size = 9, autocolorscale=False,colorscale = 'Viridis',
            color = snow_sample['elev'], colorbar = dict(title='Elevation (m)')))]


layout1 = dict(title='NOAA Weather Snowfall Station Elevations',
              geo = dict(scope='usa',projection = dict(type ='albers usa'),showland = True,
                    landcolor="rgb(250,250,250)",subunitcolor = "rgb(217,217,217)",
                     countrycolor = "rgb(217,217,217)",countrywidth =0.5, subunitwidth=0.5))
plotly.offline.iplot({
    "data": data1,
    "layout": layout1
})
