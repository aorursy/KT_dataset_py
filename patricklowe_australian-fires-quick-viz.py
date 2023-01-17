import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
"""

latitude	- Estimated Centre of fire

longitude	- Estimated Centre of fire

bright_ti4	- VIIRS I-4 channel brightness temperature of the fire pixel measured in Kelvin.

scan	    - Reflects Actual pixel size

track	    - Reflects Actual pixel size

acq_date	- date of MODIS Acc

acq_time	- Time of satellite overpass (UTC)

satellite	- Aqua or Terra

instrument	- 

confidence	- 0-100 scale of confidence of fire

version	    - NRT (near real time), or Standard Processing

bright_ti5  - I5 Channel Brightness temp of fire pixel, Kelvin

frp	        - Fire Radiative Power, megawatts

daynight    - Day / Night

"""



import pandas as pd

import numpy as np

import datetime

import plotly.express as px

from plotly.offline import plot



# Read in data and view description

df = pd.read_csv("../input/fires-from-space-australia-and-new-zeland/fire_nrt_M6_96619.csv")



# Style 1

fig = px.density_mapbox(df, 

                        lat ='latitude', 

                        lon ='longitude', 

                        z = 'brightness', 

                        color_continuous_scale  = 'solar',

                        range_color = [300,507],

                        radius = 2,

                        center = dict(lat=-25, lon=140), 

                        zoom = 2.5,

                        mapbox_style = "carto-darkmatter",

                        animation_frame = "acq_date",

                        )

fig.update_layout(title='Australian Fires - Oct 2019 to Jan 2020')

#plot( fig )

fig.show()