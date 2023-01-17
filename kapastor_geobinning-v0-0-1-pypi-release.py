!pip install geobinning==1.1.4

import geobinning as gb
import numpy as np 
import pandas as pd 
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
df = pd.read_csv('../input/petfinder-animal-shelters-database/petfinder_shelters.csv')
points =  df[['longitude','latitude']].values.tolist()
import pandas as pd
fp = gb.geobin(counties,points)
import plotly.express as px


fig = px.choropleth(fp, geojson=counties, locations='id', color='bins',
                           color_continuous_scale='Viridis',
                           scope="usa"
                   )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()