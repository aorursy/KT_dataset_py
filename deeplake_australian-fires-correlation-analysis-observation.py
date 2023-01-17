# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import datetime
import plotly.express as px
# (_nrt_) file has Near Real Time data.
# NRT data are replaced with standard quality data when they are available - hence pickup nrt data for analysis

# latitude  - Estimated Centre of fire
# longitude - Estimated Centre of fire
# scan      - Reflects Actual pixel size
# track     - Reflects Actual pixel size
# acq_date  - date of MODIS Acc
# acq_time  - Time of satellite overpass (UTC)
# satellite - Aqua or Terra
# confidence- 0-100 scale of confidence of fire
# frp       - Fire Radiative Power, megawatts
# daynight  - Day / Night

# Read in data and view description
df_nrt = pd.read_csv("../input/fires-from-space-australia-and-new-zeland/fire_nrt_M6_96619.csv")
df_nrt
df_nrt.describe()
# satellite, instrument, version - may not be relevant to the fire outbreak, hence dropping
# bright_t31 data shows same nature with brightness, hence dropping

df_nrt=df_nrt.drop(['acq_time','satellite','instrument','bright_t31','version'],axis=1)
figure= plt.figure(figsize=(10,10))
sns.heatmap(df_nrt.corr(), annot=True)
sns.pairplot(df_nrt)

# confidence shows relatively strong correlation with brightness,
# but hard to find factors that shows strong correlation.
# density_mapbox
fig = px.density_mapbox(df_nrt, 
                        lat ='latitude', lon ='longitude', z = 'brightness', 
                        center = dict(lat=-25, lon=140), zoom = 2.5,
                        hover_name="brightness", hover_data=["scan", "track"],
                        animation_frame = "acq_date",
                        mapbox_style = "carto-darkmatter",
                        color_continuous_scale  = 'solar',
                        range_color = [300,507],
                        radius = 10,
                        )
fig.update_layout(title='Australian Fires: Oct 2019 to Jan 2020')
fig.show()

# observation - the fire spreads out to densly populated regions overtime
# update layout - to see if there's a relationship between mountaineous landscapes and fire

fig.update_layout(
    mapbox_style="white-bg",
    mapbox_layers=[
        {
            "below": 'traces',
            "sourcetype": "raster",
            "sourceattribution": "United States Geological Survey",
            "source": [
                "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
            ]
        }
      ])
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
# 1. Limitation in data - hard to find factors that shows strong correlation with fire.
#
# 2. Observation - the fire spreads out to densly populated regions overtime,
#
# 3. and once it spreads out to the mountaineous regions, it tends not to die out easily.
#    hence it's critical to manage the fire at early phase.

# It's hard to find factors in the given datasets to apply and train & test with machine learning models.