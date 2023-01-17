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
import plotly.express as px
import datetime

df = pd.read_csv('/kaggle/input/community-mobility-data-for-covid19/community_mobility_change.csv',
                 parse_dates=['date'])

parent_loc = 'world'
date = datetime.datetime(2020, 4, 4)

data = df[(df['parent_loc'] == parent_loc) & (df['date'] == date)]
    
geo_df = data.groupby(['location'])[['mobility_change']]\
.mean().reset_index()\
.rename(columns={'mobility_change': 'Average movility change'})



fig = px.choropleth(geo_df, locations='location',
                    locationmode='country names',
                    color="Average movility change",
                    hover_name="location",
                    range_color=[-0.8, 0.2],
                    color_continuous_scale=px.colors.sequential.Plasma,
                   )


fig.show()
