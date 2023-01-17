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
data = pd.read_csv('/kaggle/input/asset-manager-2/Asset_Data.csv')

error = [81-1, 102-1, 186-1, 221-1, 241-1, 301-1, 332-1, 394-1]



data.dtypes
data = data.drop(error)

data['2019'] = data['2019'].astype(int)

data['2018'] = data['2018'].astype(float)



import ast

data['Asset Name'] = data['Asset Name'].apply(lambda x:' '.join(ast.literal_eval( x )))

data
america = ['US', 'Canada', 'Brazil', 'Colombia']

europe = ['UK', 'France', 'Germany', 'Switzerland', 'Netherlands', 'Spain', 'Denmark', 'Italy', 'Sweden', 'Austria', 'Finland', 'Norway', 'Ireland', 'Portugal', 'Russia', 'Iceland', 'Belgium', 'Luxembourg']

asia = ['Japan', 'Australia', 'China', 'Korea', 'Singapore', 'Kong'] #Hongkong

merge = ['US/UK', 'US/Germany', 'Africa/UK', 'UK/Switzerland']



map_ = {}

for i in america:

    map_[i] = 'America'

for i in europe:

    map_[i] = 'Europe'

for i in asia:

    map_[i] = 'Asia'

for i in merge:

    map_[i] = 'Merge'

map_
data['Profit/Loss'] = (data['2019'] - data['2018'])*1000000



data.loc[data['Profit/Loss'] < 0, 'Loss'] = 1

data.loc[data['Profit/Loss'] >= 0, 'Loss'] = 0



data['Region'] = data['Country Name'].map(map_)

data
data['Loss'].sum()
data.groupby('Country Name').agg(sum)
data.groupby('Region').agg(sum)
data_region = data.groupby('Region').agg(sum).join(data['Region'].value_counts())

data_region['Ratio'] = round(100*data_region['Loss']/data_region['Region'],2)

data_region['Name'] = data_region.index

data_region
data_country = data.groupby('Country Name').agg(sum).join(data['Country Name'].value_counts())

data_country['Ratio'] = round(100*data_country['Loss']/data_country['Country Name'],2)

data_country['Name'] = data_country.index

data_country
import plotly.express as px

fig = px.pie(data_region, values='Loss', names='Name', title='Region Loss')

fig.show()
fig = px.pie(data_country, values='Loss', names='Name', title='Country Loss')

fig.show()
data_region
fig = px.bar(data_region.sort_values(by=['Loss'], ascending=False), y='Loss', x='Name', text='Loss', title='Region Loss')

fig.update_traces(textposition='outside')

fig.show()
import plotly.graph_objects as go

animals=['giraffes', 'orangutans', 'monkeys']



fig = go.Figure(data=[

    go.Bar(name='SF Zoo', x=animals, y=[20, 14, 23]),

    go.Bar(name='LA Zoo', x=animals, y=[12, 18, 29])

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
data_region = data_region.sort_values(by=['Ratio'],ascending=False)

data_region
import plotly.graph_objects as go



fig = go.Figure(data=[

    go.Bar(name='Number of Loss', x=data_region['Name'], y=data_region['Loss'],text=data_region['Loss'],textposition='auto'),

    go.Bar(name='Ratio', x=data_region['Name'], y=data_region['Ratio'],text=data_region['Ratio'],textposition='auto',marker_color='lightsalmon')

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
fig = px.bar(data_country.sort_values(by=['Loss'], ascending=False), y='Loss', x='Name', text='Loss', title='Country Loss')

fig.update_traces(textposition='outside')

fig.show()
data[data['Country Name'] == 'Singapore']
data.sort_values(by=['Profit/Loss'])
data_plot = data[data['Profit/Loss'] <= 0].sort_values(by=['Profit/Loss'], ascending=False)

data_plot['Profit/Loss'] = data_plot['Profit/Loss'].abs()

data_plot
data_plot.sort_values(by=['Profit/Loss'], ascending=False)
data_showcase_3 = data_plot.tail(20)[['Index','Asset Name','Profit/Loss','Region']]

data_showcase_3['Profit/Loss'] = data_showcase_3['Profit/Loss']/1000000000

data_showcase_3.sort_values(by=['Profit/Loss'],ascending=False)
data_plot.tail(20)['Profit/Loss'].sum()/1000000000
fig = px.bar(data_plot.tail(20), x='Profit/Loss', y='Asset Name', text='Index', orientation='h')

fig.update_traces(textposition='outside')

fig.show()
fig = px.bar(data_plot.iloc[-50:-25], x='Profit/Loss', y='Asset Name', text='Index', orientation='h')

fig.update_traces(textposition='outside')

fig.show()
fig = px.bar(data_plot.iloc[-75:-50], x='Profit/Loss', y='Asset Name', text='Index', orientation='h')

fig.update_traces(textposition='outside')

fig.show()
fig = px.bar(data_plot.iloc[-100:-75], x='Profit/Loss', y='Asset Name', text='Index', orientation='h')

fig.update_traces(textposition='outside')

fig.show()
data_plot
data_plot
data_showcase_2 = data_plot.sort_values(by=['2019'],ascending=False).iloc[:11]

data_showcase_2['2019'] = round(data_showcase_2['2019']/1000000,3)

data_showcase_2['2018'] = round(data_showcase_2['2018']/1000000,3)

data_showcase_2['Profit/Loss'] = round(data_showcase_2['Profit/Loss']/1000000000,3)

data_showcase_2
data_showcase_2
data = data.sort_values(by=['Profit/Loss'])

data
data_plot
data_plot
data_plot.loc[244,'Index'] = '245'
data_plot.iloc[-1]
data_plot.isnull().sum()
data_plot['Index'] = data_plot['Index'].astype(int)

data_plot
data_output = data_plot.sort_values(by=['Index'])

data_output
data_output.to_excel("output.xlsx")
data_output.to_csv('Data_output.csv',index=False)
data = data.sort_values(by=['2019'],ascending=False)

data
import plotly.graph_objects as go





fig = go.Figure()

fig.add_trace(go.Bar(

    x=data['Asset Name'].iloc[:10],

    y=data['2018'].iloc[:10],

    name='2018',

    marker_color='blue'))

fig.add_trace(go.Bar(

    x=data['Asset Name'].iloc[:10],

    y=data['2019'].iloc[:10],

    name='2019',

    marker_color='red'))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.update_layout(barmode='group', xaxis_tickangle=-45)

fig.show()
data_showcase = data[['Asset Name','Country Name','2019','2018']].iloc[:10]

data_showcase['2019'] = round(data_showcase['2019']/1000000,3)

data_showcase['2018'] = round(data_showcase['2018']/1000000,3)

data_showcase
data_plot
import plotly.graph_objects as go





fig = go.Figure()

fig.add_trace(go.Bar(

    x=data_plot['Asset Name'].iloc[-10:],

    y=data_plot['2018'].iloc[-10:],

    name='2018',

    marker_color='blue'))

fig.add_trace(go.Bar(

    x=data_plot['Asset Name'].iloc[-10:],

    y=data_plot['2019'].iloc[-10:],

    name='2019',

    marker_color='red'))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.update_layout(barmode='group', xaxis_tickangle=-45)

fig.show()
import plotly.graph_objects as go



months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',

          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']



fig = go.Figure()

fig.add_trace(go.Bar(

    x=months,

    y=[20, 14, 25, 16, 18, 22, 19, 15, 12, 16, 14, 17],

    name='Primary Product',

    marker_color='indianred'))

fig.add_trace(go.Bar(

    x=months,

    y=[19, 14, 22, 14, 16, 19, 15, 14, 10, 12, 12, 16],

    name='Secondary Product',

    marker_color='lightsalmon'))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.update_layout(barmode='group', xaxis_tickangle=-45)

fig.show()
data
data[(data['2019']<500000/1.62)&(data['Loss']<500000/1.62)]