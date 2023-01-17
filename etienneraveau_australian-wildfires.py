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
# Read dataset
data = pd.read_csv('/kaggle/input/fires-from-space-australia-and-new-zeland/fire_archive_M6_96619.csv')

# Make sure data is sorted by date
data.sort_values(by='acq_time')

# Get some hints into the data
print(data.columns)
print(data.head)
# Rather try a scatter_geo plot
import plotly.express as px

geo_scat = px.scatter_geo(data_frame=data,lat='latitude',lon='longitude',color='brightness',center = {"lat": np.mean(data['latitude']), "lon": np.mean(data['longitude'])},opacity=0.5,animation_frame='acq_time',title="Evolution of Australian wildfires intensity over time")
geo_scat.show()
# Just try another visulaization
fig = px.scatter_3d(data, x='latitude', y='longitude', z='acq_time',color='brightness',opacity=0.2)
fig.show()