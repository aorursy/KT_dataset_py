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
kaggle_pizza_df = pd.read_csv('../input/pizza-restaurants-and-the-pizza-they-sell/8358_1.csv')

kaggle_pizza_df
kaggle_pizza_df[['name','menus.name','menus.amountMax']][kaggle_pizza_df['menus.amountMax']==kaggle_pizza_df['menus.amountMax'].max()]
kaggle_pizza_df[['name','menus.name','menus.amountMin']][kaggle_pizza_df['menus.amountMin']==kaggle_pizza_df['menus.amountMin'][kaggle_pizza_df['menus.amountMin'].gt(0)].min()]
kaggle_pizza_df['menus.amountDiff'] = kaggle_pizza_df['menus.amountMax']- kaggle_pizza_df['menus.amountMin']
kaggle_pizza_df[['name','menus.name','menus.amountDiff']][kaggle_pizza_df['menus.amountDiff']==kaggle_pizza_df['menus.amountDiff'].max()]
kaggle_pizza_df[['name','menus.name','menus.amountDiff']][kaggle_pizza_df['menus.amountDiff']==kaggle_pizza_df['menus.amountDiff'].replace(0, np.nan).min()]
df1 = pd.DataFrame(kaggle_pizza_df['menus.dateSeen'].values, columns = ['date'])

df2=  df1['date'].str.split(',', expand=True).stack().reset_index(level=1,drop=True)

df2 = df2.to_frame('date').set_index(df2.groupby(df2.index).cumcount(), append=True)

df2['date']= df2['date'].str[:10]
df2['date']= pd.to_datetime(df2['date']) 

df2['day_of_week'] = df2['date'].dt.day_name()
day_count= df2.groupby(['day_of_week']).size().to_frame('day_count').reset_index()

day_count.sort_values(by='day_count', ascending=False)
day_count.sort_values(by='day_count').plot('day_of_week','day_count',kind='barh',color='C2')
pizza= kaggle_pizza_df.drop_duplicates(subset=['address','city','latitude','longitude','menus.name','name'], keep='first')

pizza_count= pizza.groupby(['menus.name']).size().to_frame('count').reset_index()

pizza_count_sort = pizza_count.sort_values(by='count', ascending=False)[:10]

pizza_count_sort
pizza_count_sort.plot('menus.name','count', kind='bar')
pizza_name_df_count= pizza.groupby(['name']).size().to_frame('count').reset_index()

pizza_name_df_count.sort_values(by='count', ascending=False)[:10]
pizza_name_df_count.sort_values(by='count', ascending=False)[:10].plot('name','count',kind='bar',color='C2')
kaggle_pizza_df_city= kaggle_pizza_df.groupby(['city']).size().to_frame('count').reset_index()

kaggle_pizza_df_city.sort_values(by='count', ascending=False)[:20]
kaggle_pizza_df_city.sort_values(by='count', ascending=False)[:10].plot('city','count',kind='bar',color='C6')
state_count= kaggle_pizza_df.groupby(['province']).size().to_frame('count').reset_index()

state_count.sort_values(by='count',ascending=False)
state_count.sort_values(by='count', ascending=False)[:20].plot('province','count',kind='bar')
kaggle_pizza_df['latitude'] = kaggle_pizza_df['latitude'].astype(str)

kaggle_pizza_df['longitude'] = kaggle_pizza_df['longitude'].astype(str)
kaggle_pizza_df['altitude'] = kaggle_pizza_df[['latitude', 'longitude']].apply(lambda x: ', '.join(x), axis=1)

kaggle_pizza_df_altitude= kaggle_pizza_df.groupby(['altitude']).size().to_frame('frequency').reset_index()

kaggle_pizza_df.drop_duplicates(subset= ['city','latitude','longitude','altitude'], keep='first', inplace=True)

kaggle_pizza_df_city= kaggle_pizza_df[['name','city','latitude','longitude','altitude']]

kaggle_pizza_df_arranged= pd.merge(kaggle_pizza_df_city, kaggle_pizza_df_altitude, on='altitude', how='left')

kaggle_pizza_df_arranged['longitude']= kaggle_pizza_df_arranged['longitude'].astype(float) 

kaggle_pizza_df_arranged['latitude']= kaggle_pizza_df_arranged['latitude'].astype(float) 
import matplotlib.pyplot as plt

kaggle_pizza_df_arranged.plot(kind="scatter", x="longitude", y ="latitude", s=10, alpha= 0.6)

plt.show()
import plotly.graph_objects as go

kaggle_pizza_df_arranged['text'] = kaggle_pizza_df_arranged['name']+ ', ' + kaggle_pizza_df_arranged['city'] + ', '+ 'total pizza types: ' + kaggle_pizza_df_arranged['frequency'].astype(str)

fig = go.Figure(data=go.Scattergeo(

        lon = kaggle_pizza_df_arranged['longitude'],

        lat = kaggle_pizza_df_arranged['latitude'],

        text = kaggle_pizza_df_arranged['text'],

        mode = 'markers',

        marker = dict(

            size = 8,

            opacity = 0.8,

            reversescale = False,

            autocolorscale = False,

            symbol = 'circle',

            line = dict(

                width=1,

                color='rgba(102, 102, 102)'

            ),

            colorscale = 'Blues',

            cmin = 0,

            color = kaggle_pizza_df_arranged['frequency'],

            cmax = kaggle_pizza_df_arranged['frequency'].max(),

            colorbar_title="Pizza Frequency"

        )))
fig.update_layout(

        title = 'Pizza distribution in USA<br>(Hover for pizza details)',

        geo = dict(

            scope='usa',

            projection_type='albers usa',

            showland = True,

            landcolor = "rgb(250, 250, 250)",

            subunitcolor = "rgb(217, 217, 217)",

            countrycolor = "rgb(217, 217, 217)",

            countrywidth = 0.5,

            subunitwidth = 0.5

        ),

    )

fig.show()
scl = [[0.0, 'rgb(248,255,206)'],[0.2, 'rgb(203,255,205)'],[0.4, 'rgb(155,255,164)'], [0.6, 'rgb(79,255,178)'],[0.8, 'rgb(15,183,132)'], [1, '#008059']]

data = [dict(

        type = 'choropleth',

        colorscale = scl,

        autocolorscale = False,

        locations = state_count.province,

        z= state_count['count'],

        locationmode= 'USA-states',

        marker = dict(

            line = dict(

                width=2,

                color='rgb(255, 255, 255)'

            )),

            colorbar = dict(

                title="Pizza Frequency")

        )]
layout = dict(

        title = 'Pizza distribution in USA<br>(Hover for pizza details)',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa'),

            showlakes = True,

            lakecolor = 'rgb(255,255,255)'

        ),

    )

py.init_notebook_mode(connected=True)

fig = dict(data=data, layout= layout)

py.iplot(fig, filename='d3-chloropleth-map')