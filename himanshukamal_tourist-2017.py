# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/tourism'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/tourism/Tourism_In_India_Statistics_2018-Table_5.1.3(1).csv')

df
df = df.drop(df.index[36])
df.at[15,'State/UT']='Jammu & Kashmir'

df.at[16,'State/UT']='Karnataka'



df.at[31,'State/UT']='Telangana'
import plotly.express as px
fig = px.bar(df, x='State/UT', y='Domestic',title='Domestic Tourist',color_discrete_sequence=px.colors.sequential.Viridis)

fig
fig = px.pie(df,values = 'Domestic',names = 'State/UT',title ='Domestic tourist', color_discrete_sequence=px.colors.sequential.Viridis)

fig.show()
fig = px.bar(df, x='State/UT', y='Foreign',title = 'International tourist',color_discrete_sequence=px.colors.sequential.Viridis)

fig
fig = px.pie(df,values = 'Foreign',names = 'State/UT',title = 'International tourist',color_discrete_sequence=px.colors.sequential.Viridis)

fig.show()
df
import json


fig = px.choropleth(df, 

                    geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",

                    locations = 'State/UT',

                    color = 'Domestic',

                    featureidkey='properties.ST_NM',

                    color_continuous_scale='Viridis',

                    title = 'Domestic tourist'

                   )

fig.update_geos(fitbounds="locations", visible=False)

fig.show()
fig = px.choropleth(df, 

                    geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",

                    locations = 'State/UT',

                    color = 'Foreign',

                    featureidkey='properties.ST_NM',

                    color_continuous_scale='Viridis',

                    title = 'International tourist'

                   )

fig.update_geos(fitbounds="locations", visible=False)

fig.show()
df.at[15,'State/UT']='Jammu & Kashmir'

df
df.at[16,'State/UT']='Karnataka'

df
df