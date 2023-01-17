# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly

import plotly.plotly as py

import plotly.offline as offline

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

from plotly.graph_objs import Scatter, Figure, Layout

from plotly import tools

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
red_cam = pd.read_csv("../input/red-light-camera-locations.csv")

red_vlt = pd.read_csv("../input/red-light-camera-violations.csv")

spd_cam = pd.read_csv("../input/speed-camera-locations.csv")

spd_vlt = pd.read_csv("../input/speed-camera-violations.csv")
red_cam.shape
red_cam['THIRD APPROACH'].isnull().sum()
red_cam.head()
red_cam.LONGITUDE.mean()


mapbox_access_token = 'pk.eyJ1IjoibGVlZG9oeXVuIiwiYSI6ImNqbjl1Y2hmcTB6dTQzcnBiNDZ2cXcwbGEifQ.hcPVtUhnyzXDXZbQQH0nMw'

data = [go.Scattermapbox(

    lon = red_vlt['LONGITUDE'],

    lat = red_vlt['LATITUDE'],

    marker = dict(

        size = 10,

        color='red'

        

    ))]



layout = dict(

        title = 'Geo Locations based on Zip code',

        mapbox = dict(

            accesstoken = mapbox_access_token,

            center= dict(lat=41.9,lon=-87.7),

            bearing=5,

            pitch=5,

            zoom=8.5,

        )

    )

fig = dict( data=data, layout=layout )

iplot( fig, validate=False)
red_vlt['VIOLATIONS'].plot()
red_vlt['CAMERA ID'].plot(kind='hist')
red_vlt.head()
spd_cam.head()
spd_vlt.shape


mapbox_access_token = 'pk.eyJ1IjoibGVlZG9oeXVuIiwiYSI6ImNqbjl1Y2hmcTB6dTQzcnBiNDZ2cXcwbGEifQ.hcPVtUhnyzXDXZbQQH0nMw'

data1 = [go.Scattermapbox(

    lon = red_cam['LONGITUDE'],

    lat = red_cam['LATITUDE'],

    marker = dict(

        size = 10,

        color='red'

        

    )),

    go.Scattermapbox(

    lon = spd_cam['LONGITUDE'],

    lat = spd_cam['LATITUDE'],

    marker = dict(

        size = 10,

        color='green'

        

    ))    ]



layout = dict(

        title = 'red_cam locations and spd_cam locations',

        mapbox = dict(

            accesstoken = mapbox_access_token,

            center= dict(lat=41.9,lon=-87.7),

            bearing=5,

            pitch=5,

            zoom=8.5,

        )

    )

fig = dict( data=data1, layout=layout )

iplot( fig, validate=False)
spd_cam.head()
spd_vlt.head(20)
spd_vlt['CAMERA ID'].unique()
sns.distplot(spd_vlt.VIOLATIONS)