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

import plotly.express as px



df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

df = df.groupby(['ObservationDate','Country/Region']).sum().reset_index()



df['daily_existing']=df['Confirmed'].values-df['Deaths'].diff()-df['Recovered'].diff()



fig = px.choropleth(df,locations='Country/Region',locationmode='country names',color='Confirmed',hover_name='Country/Region',animation_frame='ObservationDate',color_continuous_scale='Jet',range_color=(0.1,200000.))

fig.update_layout(title_text='Remaining Confirmed Cases per Country of each day - Hiroki Sekiya',title_x=0.5)

fig.write_html('Remaining Confirmed Cases per Country of each day - Hiroki Sekiya.html')

fig.show()



df = pd.read_csv('../input/april-25-2015-earthquakes/query.csv',header=0)



df.index = pd.to_datetime(df['time'])

df['time'] = df.index.strftime('%Y-%m-%d %H:00:00')

fig = px.scatter_geo(df,lat='latitude',lon='longitude',color='mag',animation_frame='time',color_continuous_scale='Jet',range_color=(3.,7.))

fig.write_html('Earthquake April-25-2015 - Hiroki Sekiya.html')

fig.show()