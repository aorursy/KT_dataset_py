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
# import pandas as pd

# state_data = pd.read_csv("/kaggle/input/unemployment-data/states.csv")

# state_data.head()
import pandas as pd

cancer = pd.read_csv("/kaggle/input/cancer2/cancer2.csv")

cancer.head()
import folium

 

# Load the shape of the zone (US states)

# Find the original file here: https://github.com/python-visualization/folium/tree/master/examples/data

# You have to download this file and set the directory where you saved it

state_geo = "/kaggle/input/usstates/us-states.json"



# Initialize the map:

m = folium.Map(location=[37, -102], zoom_start=5)

 

# Add the color for the chloropleth:

m.choropleth(

 geo_data=state_geo,

 name='choropleth',

 data=cancer,

 columns=['State', 'Total.Rate'],

 key_on='feature.id',

 fill_color='YlGn',

 fill_opacity=0.7,

 line_opacity=0.2,

 legend_name='Cancer Rate (%)'

)

folium.LayerControl().add_to(m)



m