import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import datetime as datetime

import plotly.graph_objects as go

from plotly.subplots import make_subplots

%matplotlib inline
solar01 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

sensor01 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

solar02 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')

sensor02 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
solar01.head()
solar02.head()
print("Solar Generation Plant 1's info")

solar01.info()

print('\n')

print("Solar Generation Plant 2's info")

solar02.info()
print('Plant 1')

solar01.isnull().sum()
print('Plant 2')

solar02.isnull().sum()
print ('Plant 1 has '+ str(solar01['SOURCE_KEY'].nunique()) + ' inverters')

print ('Plant 2 has '+ str(solar02['SOURCE_KEY'].nunique()) + ' inverters')
print('Plant 1')

solar01.groupby('SOURCE_KEY').count()
print('Plant 2')

solar02.groupby('SOURCE_KEY').count()
sensor01.head()
sensor02.head()
print("Sensor 1's info")

sensor01.info()

print('\n')

print("Sensor 2's info")

sensor02.info()
print('Sensor 1 has '+str(sensor01['SOURCE_KEY'].nunique())+' source key')

print('Sensor 2 has '+str(sensor02['SOURCE_KEY'].nunique())+' source key')
print('Sensor 1')

sensor01.isnull().sum()
print('Sensor 2')

sensor01.isnull().sum()
print('Sensor 1')

sensor01.count()
print('Sensor 2')

sensor02.count()
print('the total should be '+ str(34*24*4)+' rows')
solar01.columns = solar01.columns.str.lower()

solar02.columns = solar02.columns.str.lower()

solar01.drop('plant_id', axis=1, inplace=True)

solar02.drop('plant_id', axis=1, inplace=True)
solar01['date_time'] = pd.to_datetime(solar01['date_time'],format ='%d-%m-%Y %H:%M')

solar02['date_time'] = pd.to_datetime(solar02['date_time'],format ='%Y-%m-%d %H:%M:%S')

solar01['date'] = solar01['date_time'].dt.date

solar01['time'] = solar01['date_time'].dt.time

solar02['date'] = solar02['date_time'].dt.date

solar02['time'] = solar02['date_time'].dt.time
solar01_inverter_id = solar01['source_key'].unique()

solar02_inverter_id = solar02['source_key'].unique()

solar01['source_key'] = solar01['source_key'].apply(lambda x :  int(np.where(solar01_inverter_id == x)[0]))

solar02['source_key'] = solar02['source_key'].apply(lambda x :  int(np.where(solar02_inverter_id == x)[0]))
solar01.head()
solar02.head()
solar01[(solar01['source_key']==0) & (solar01['date_time'].between('2020-05-15','2020-05-21'))]
data = solar01[(solar01['source_key']==0) & (solar01['date_time'].between('2020-05-15','2020-05-21'))]

data['time'] = data['time'].astype(str)

g = sns.relplot(

        data=data,

        x='time',

        y='dc_power',

        row='date',

        kind='line',

        height=2,

        aspect=6)



g.set(xlim=('00:00:00', '23:45:00'), xticks=['00:00:00','06:00:00','12:00:00','18:00:00','23:45:00'])
data = solar01[(solar01['source_key']==0) & (solar01['date_time'].between('2020-05-15','2020-05-21'))]

data['time'] = data['time'].astype(str)

g = sns.relplot(

        data=data,

        x='time',

        y='dc_power',

        row='date',

        kind='scatter',

        height=2,

        aspect=6

        )

    

g.set(xlim=('00:00:00', '23:45:00'), xticks=['00:00:00','06:00:00','12:00:00','18:00:00','23:45:00'])
data = solar01[(solar01['source_key']==0) & (solar01['date_time'].between('2020-05-15','2020-05-21'))]

data['time'] = data['time'].astype(str)

g = sns.relplot(

        data=data,

        x='time',

        y='daily_yield',

        row='date',

        kind='scatter',

        height=2,

        aspect=6

        )



g.set(xlim=('00:00:00', '23:45:00'), xticks=['00:00:00','06:00:00','12:00:00','18:00:00','23:45:00'])
data = solar01[(solar01['source_key']==0) & (solar01['date_time'].between('2020-05-15','2020-05-21'))]

data['date_time'] = data['date_time'].astype(str)

g = sns.relplot(

        data=data,

        x='date_time',

        y='total_yield',

        kind='scatter',

        height=6,

        aspect=2

        )



g.set(xlim=('2020-05-15 00:00:00', '2020-05-21 00:00:00'), xticks=['2020-05-15 00:00:00','2020-05-17 00:00:00','2020-05-19 00:00:00','2020-05-21 00:00:00'])
fulltime = pd.date_range(start='2020-05-15 00:00',end='2020-06-17 23:45' , freq='15T')

fulltime = pd.DataFrame({'date_time':fulltime})

fulltime
solar01_inv_0 = solar01[solar01['source_key']==0].reset_index(drop=True)
solar01_inv_0 = pd.merge(fulltime, solar01_inv_0, how='outer')

solar01_inv_0
solar01_inv_0.index = solar01_inv_0['date_time']

solar01_inv_0.drop('date_time', axis=1, inplace=True)
sns.heatmap(solar01_inv_0.isnull())
solar01_inv_0['date'] = solar01_inv_0.index.date

solar01_inv_0['time'] = solar01_inv_0.index.time

solar01_inv_0['source_key'] = 0
solar01_inv_0.isnull().sum()
early_morning = solar01_inv_0.between_time('00:00:00','05:45:00')

afternoon     = solar01_inv_0.between_time('06:00:00','18:30:00')

night         = solar01_inv_0.between_time('18:45:00','23:45:00')
early_morning['dc_power'].fillna(value=0, inplace=True)

early_morning['ac_power'].fillna(value=0, inplace=True)

early_morning['daily_yield'].fillna(value =0, inplace=True)
night['dc_power'].fillna(value=0, inplace=True)

night['ac_power'].fillna(value=0, inplace=True)

night['daily_yield'].fillna(method='ffill', inplace=True)
solar01_inv_0 = pd.concat([early_morning,afternoon, night])

solar01_inv_0 = solar01_inv_0.sort_index()
data = solar01_inv_0

data['time'] = data['time'].astype(str)

sns.set(font_scale =1.5)



g = sns.relplot(

        data=data,

        x='time',

        y='dc_power',

        col='date',

        kind='scatter',

        height=2,

        aspect=3,

        col_wrap=3

        )



g.set(xlim=('00:00:00', '23:45:00'), xticks=['00:00:00','06:00:00','12:00:00','18:00:00','23:45:00'])
data = solar01_inv_0

data['time'] = data['time'].astype(str)

sns.set(font_scale =1.5)



g = sns.relplot(

        data=data,

        x='time',

        y='daily_yield',

        col='date',

        kind='scatter',

        height=2,

        aspect=3,

        col_wrap=3

        )



g.set(xlim=('00:00:00', '23:45:00'), xticks=['00:00:00','06:00:00','12:00:00','18:00:00','23:45:00'])
solar01_inv_0.isnull().sum()
solar01_inv_0[['ac_power','dc_power','daily_yield']] = solar01_inv_0[['ac_power','dc_power','daily_yield']].interpolate(method='time')
data = solar01_inv_0

data['time'] = data['time'].astype(str)

sns.set(font_scale =1.5)



g = sns.relplot(

        data=data,

        x='time',

        y='dc_power',

        col='date',

        kind='scatter',

        height=2,

        aspect=3,

        col_wrap=3

        )



g.set(xlim=('00:00:00', '23:45:00'), xticks=['00:00:00','06:00:00','12:00:00','18:00:00','23:45:00'])
data = solar01_inv_0

data['time'] = data['time'].astype(str)

sns.set(font_scale =1.5)



g = sns.relplot(

        data=data,

        x='time',

        y='daily_yield',

        col='date',

        kind='scatter',

        height=2,

        aspect=3,

        col_wrap=3

        )



g.set(xlim=('00:00:00', '23:45:00'), xticks=['00:00:00','06:00:00','12:00:00','18:00:00','23:45:00'])
solar01_inv_0.isnull().sum()
solar01_inv_0['total_yield'] = solar01_inv_0['total_yield'].interpolate(method='time')
data = solar01_inv_0

data.index = data.index.astype(str)



f, ax =plt.subplots(figsize=(12,8))

ax = sns.lineplot(x=data.index, 

                  y='total_yield',

                  data = data

                 )



ax.set(xlim=('2020-05-15 00:00:00','2020-06-17 00:00:00'),xticks=['2020-05-15 00:00:00','2020-06-17 00:00:00'])
solar01_inv_0.isnull().sum()
solar01_inv = [0]*22



def data_filling(inverter_id):

    #create dataframe based on inverter id.

    solar01_inv[inverter_id] = solar01[solar01['source_key']==inverter_id].reset_index(drop=True)

    

    #add full timestamp to dataframe.

    solar01_inv[inverter_id] = pd.merge(fulltime, solar01_inv[inverter_id], how='outer')

    

    #fill na with fix values.

    solar01_inv[inverter_id]['date'] = solar01_inv[inverter_id]['date_time'].dt.date

    solar01_inv[inverter_id]['time'] = solar01_inv[inverter_id]['date_time'].dt.time

    solar01_inv[inverter_id]['source_key'] = inverter_id

    

    #convert column date time as index.

    solar01_inv[inverter_id].index = solar01_inv[inverter_id]['date_time']

    solar01_inv[inverter_id].drop('date_time', axis=1, inplace=True)

    

    #divide dateframe into 3 group.

    early_morning = solar01_inv[inverter_id].between_time('00:00:00','05:45:00')

    afternoon     = solar01_inv[inverter_id].between_time('06:00:00','18:30:00')

    night         = solar01_inv[inverter_id].between_time('18:45:00','23:45:00')

    

    #fill na values on early_morning group with zero.

    early_morning['dc_power'].fillna(value=0, inplace=True)

    early_morning['ac_power'].fillna(value=0, inplace=True)

    early_morning['daily_yield'].fillna(value =0, inplace=True)

    

    #fill na values on night group with zero and fflill method for daily_yield.

    night['dc_power'].fillna(value=0, inplace=True)

    night['ac_power'].fillna(value=0, inplace=True)

    night['daily_yield'].fillna(method='ffill', inplace=True)

    

    #join them together again and sort index, so we get sorted timeline.

    solar01_inv[inverter_id] = pd.concat([early_morning,afternoon, night])

    solar01_inv[inverter_id] = solar01_inv[inverter_id].sort_index()

    

    #fill others na with interpolate function that use method time

    solar01_inv[inverter_id]['dc_power'] = solar01_inv[inverter_id]['dc_power'].interpolate(method='time')

    solar01_inv[inverter_id]['ac_power'] = solar01_inv[inverter_id]['ac_power'].interpolate(method='time')

    solar01_inv[inverter_id]['daily_yield'] = solar01_inv[inverter_id]['daily_yield'].interpolate(method='time')

    solar01_inv[inverter_id]['total_yield'] = solar01_inv[inverter_id]['total_yield'].interpolate(method='time')

    

for i in range (22):

    data_filling(i)
solar01 = pd.concat(solar01_inv)
solar02_inv = [0]*22





def data_filling(inverter_id):

    #create dataframe based on inverter id.

    solar02_inv[inverter_id] = solar02[solar02['source_key']==inverter_id].reset_index(drop=True)

    

    #add full timestamp to dataframe.

    solar02_inv[inverter_id] = pd.merge(fulltime, solar02_inv[inverter_id], how='outer')

    

    #fill na with fix values.

    solar02_inv[inverter_id]['date'] = solar02_inv[inverter_id]['date_time'].dt.date

    solar02_inv[inverter_id]['time'] = solar02_inv[inverter_id]['date_time'].dt.time

    solar02_inv[inverter_id]['source_key'] = inverter_id

    

    #convert column date time as index.

    solar02_inv[inverter_id].index = solar02_inv[inverter_id]['date_time']

    solar02_inv[inverter_id].drop('date_time', axis=1, inplace=True)

    

    #divide dateframe into 3 group.

    early_morning = solar02_inv[inverter_id].between_time('00:00:00','05:45:00')

    afternoon     = solar02_inv[inverter_id].between_time('06:00:00','18:30:00')

    night         = solar02_inv[inverter_id].between_time('18:45:00','23:45:00')

    

    #fill na values on early_morning group with zero.

    early_morning['dc_power'].fillna(value=0, inplace=True)

    early_morning['ac_power'].fillna(value=0, inplace=True)

    early_morning['daily_yield'].fillna(value =0, inplace=True)

    

    #fill na values on night group with zero and fflill method for daily_yield.

    night['dc_power'].fillna(value=0, inplace=True)

    night['ac_power'].fillna(value=0, inplace=True)

    night['daily_yield'].fillna(method='ffill', inplace=True)

    

    #join them together again and sort index, so we get sorted timeline.

    solar02_inv[inverter_id] = pd.concat([early_morning,afternoon, night])

    solar02_inv[inverter_id] = solar02_inv[inverter_id].sort_index()

    

    #fill others na with interpolate function that use method time

    solar02_inv[inverter_id]['dc_power'] = solar02_inv[inverter_id]['dc_power'].interpolate(method='time')

    solar02_inv[inverter_id]['ac_power'] = solar02_inv[inverter_id]['ac_power'].interpolate(method='time')

    solar02_inv[inverter_id]['daily_yield'] = solar02_inv[inverter_id]['daily_yield'].interpolate(method='time')

    solar02_inv[inverter_id]['total_yield'] = solar02_inv[inverter_id]['total_yield'].interpolate(method='time')

    

for i in range (22):

    data_filling(i)
solar02 = pd.concat(solar02_inv)
sensor01.columns = sensor01.columns.str.lower()

sensor02.columns = sensor02.columns.str.lower()

sensor01.drop('plant_id', axis=1, inplace=True)

sensor02.drop('plant_id', axis=1, inplace=True)
sensor01['date_time'] = pd.to_datetime(sensor01['date_time'],format ='%Y-%m-%d %H:%M:%S')

sensor02['date_time'] = pd.to_datetime(sensor02['date_time'],format ='%Y-%m-%d %H:%M:%S')

sensor01['date'] = sensor01['date_time'].dt.date

sensor01['time'] = sensor01['date_time'].dt.time

sensor02['date'] = sensor02['date_time'].dt.date

sensor02['time'] = sensor02['date_time'].dt.time
sensor01.head()
sensor02.head()
data = sensor01

data['time'] = data['time'].astype(str)

sns.set(font_scale =1.5)

g = sns.relplot(data=data,

            x='time',

            y='ambient_temperature',

            col='date',

            kind='scatter',

            height=3,

            aspect=3,

            col_wrap=3

               )

g.set(xlim=('00:00:00', '23:45:00'), xticks=['00:00:00','06:00:00','12:00:00','18:00:00','23:45:00'])
data = sensor01

data['time'] = data['time'].astype(str)

sns.set(font_scale =1.5)

g = sns.relplot(data=data,

            x='time',

            y='module_temperature',

            col='date',

            kind='scatter',

            height=3,

            aspect=3,

            col_wrap=3)

g.set(xlim=('00:00:00', '23:45:00'), xticks=['00:00:00','06:00:00','12:00:00','18:00:00','23:45:00'])
data = sensor01

data['time'] = data['time'].astype(str)

sns.set(font_scale =1.5)

g = sns.relplot(data=data,

            x='time',

            y='irradiation',

            col='date',

            kind='scatter',

            height=3,

            aspect=3,

            col_wrap=3)

g.set(xlim=('00:00:00', '23:45:00'), xticks=['00:00:00','06:00:00','12:00:00','18:00:00','23:45:00'])
sensor01 = pd.merge(fulltime, sensor01, how='outer')

sensor02 = pd.merge(fulltime, sensor02, how='outer')

sensor01.index = sensor01['date_time']

sensor02.index = sensor02['date_time']
sns.heatmap(sensor01.isnull())
sensor01['date'] = sensor01.index.date

sensor01['time'] = sensor01.index.time

sensor02['date'] = sensor02.index.date

sensor02['time'] = sensor02.index.time
sensor01[['ambient_temperature','module_temperature', 'irradiation']] = sensor01[['ambient_temperature','module_temperature', 'irradiation']].interpolate(method='time')

sensor02[['ambient_temperature','module_temperature', 'irradiation']] = sensor02[['ambient_temperature','module_temperature', 'irradiation']].interpolate(method='time')
data = sensor01

data['time'] = data['time'].astype(str)

sns.set(font_scale =1.5)

g = sns.relplot(data=data,

            x='time',

            y='module_temperature',

            col='date',

            kind='scatter',

            height=3,

            aspect=3,

            col_wrap=3)

g.set(xlim=('00:00:00', '23:45:00'), xticks=['00:00:00','06:00:00','12:00:00','18:00:00','23:45:00'])
solar01_with_sensor01_inv= [0]*22

solar02_with_sensor02_inv= [0]*22



for i in range(22):

    solar01_with_sensor01_inv[i] = pd.concat([solar01_inv[i],sensor01.drop(['date','time'], axis=1)], axis=1)

    



for i in range(22):

    solar02_with_sensor02_inv[i] = pd.concat([solar02_inv[i],sensor02.drop(['date','time'], axis=1)], axis=1)
solar01_with_sensor01 = pd.concat(solar01_with_sensor01_inv)

solar02_with_sensor02 = pd.concat(solar02_with_sensor02_inv)
solar01_with_sensor01.head()
solar02_with_sensor02.head()
solar01_with_sensor01['plant_ID'] = '1'

solar02_with_sensor02['plant_ID'] = '2'
full_data = pd.concat([solar01_with_sensor01, solar02_with_sensor02], ignore_index=True)
full_data
data1=solar01.groupby(['source_key']).sum().reset_index()

data1['source_key'] = data1['source_key'].apply(lambda x: solar01_inverter_id[x])

data2=solar02.groupby(['source_key']).sum().reset_index()

data2['source_key'] = data2['source_key'].apply(lambda x: solar02_inverter_id[x])



specs = [[{'type':'domain'}, {'type':'domain'}]]

fig = make_subplots(rows=1, cols=2, specs=specs)



pull_factor = [0]*22

pull_factor[7] = 0.05



fig.add_trace(go.Pie(labels='P1 '+ data1['source_key'], 

                     values=data1['dc_power'], 

                     name='Plant 1', 

                     title='Plant 1',

                     titlefont=dict(

                                     size=25

                                   ),

                     hovertemplate="%{label} <br />generates %{value:,.0f} kW",

                     marker_colors = px.colors.qualitative.Dark24,

                     legendgroup = 'Plant 1',

                    ), 1, 1)



fig.add_trace(go.Pie(labels='P2 '+ data2['source_key'], 

                     values=data2['dc_power'], 

                     name='Plant 2', 

                     title='Plant 2',

                     titlefont=dict(

                                     size=25

                                   ),

                     hovertemplate="%{label} <br />generates %{value:,.0f} kW",

                     marker_colors = px.colors.qualitative.Light24,   

                     legendgroup = 'Plant 2',

                     pull =pull_factor,

                    ), 1, 2)



fig.update_traces(hole=.4)



fig.update_layout(

    title_text="DC Power Generation of each Inverter"

)



fig.show()
data=solar01.groupby(['source_key','date']).sum().reset_index()

data['source_key'] = data['source_key'].apply(lambda x: solar01_inverter_id[x])



fig=px.bar( 

    data_frame = data,

    x = data['date'],

    y = data['dc_power'],

    color = 'source_key',

    color_discrete_sequence = px.colors.qualitative.Dark24,

    hover_data = {'date':True,

                  'source_key':True,

                  'dc_power':':,.0f',

                 },

    opacity = 0.8,

    labels={'date':'date',

            'dc_power':'DC Power Generated (kW)',

            'source_key':'Inverter ID'

           },

    title='DC Power Generated in Plant 1 based on date',

    height = 650

)



fig.show()
data=solar02.groupby(['source_key','date']).sum().reset_index()

data['source_key'] = data['source_key'].apply(lambda x: solar02_inverter_id[x])



fig=px.bar( 

    data_frame = data,

    x = data['date'],

    y = data['dc_power'],

    color = 'source_key',

    color_discrete_sequence = px.colors.qualitative.Light24,

    hover_data = {'date':True,

                  'source_key':True,

                  'dc_power':':,.0f',

                 },

    opacity = 0.8,

    labels={'date':'date',

            'dc_power':'DC Power Generated (kW)',

            'source_key':'Inverter ID'},

    title='DC Power Generated in Plant 2 based on date',

    height = 650

)





fig.show()
data1=solar01.groupby(['source_key']).sum().reset_index()

data1['source_key'] = data1['source_key'].apply(lambda x: solar01_inverter_id[x])

data2=solar02.groupby(['source_key']).sum().reset_index()

data2['source_key'] = data2['source_key'].apply(lambda x: solar02_inverter_id[x])



specs = [[{'type':'domain'}, {'type':'domain'}]]

fig = make_subplots(rows=1, cols=2, specs=specs)



pull_factor = [0]*22

pull_factor[7] = 0.05



fig.add_trace(go.Pie(labels='P1 '+ data1['source_key'], 

                     values=data1['ac_power'], 

                     name='Plant 1', 

                     title='Plant 1',

                     titlefont=dict(

                                     size=25

                                   ),

                     hovertemplate="%{label} <br />generates %{value:,.0f} kW",

                     marker_colors = px.colors.qualitative.Dark24,

                     legendgroup = 'Plant 1',

                    ), 1, 1)



fig.add_trace(go.Pie(labels='P2 '+ data2['source_key'], 

                     values=data2['ac_power'], 

                     name='Plant 2', 

                     title='Plant 2',

                     titlefont=dict(

                                     size=25

                                   ),

                     hovertemplate="%{label} <br />generates %{value:,.0f} kW",

                     marker_colors = px.colors.qualitative.Light24,   

                     legendgroup = 'Plant 2',

                     pull =pull_factor,

                    ), 1, 2)



fig.update_traces(hole=.4)



fig.update_layout(

    title_text="AC Power Generation of each Inverter"

)





fig.show()
data=solar01.groupby(['source_key','date']).sum().reset_index()

data['source_key'] = data['source_key'].apply(lambda x: solar01_inverter_id[x])



fig=px.bar( 

    data_frame = data,

    x = data['date'],

    y = data['ac_power'],

    color = 'source_key',

    color_discrete_sequence = px.colors.qualitative.Dark24,

    hover_data = {'date':True,

                  'source_key':True,

                  'ac_power':':,.0f',

                 },

    opacity = 0.8,

    labels={'date':'date',

            'ac_power':'AC Power Generated (kW)',

            'source_key':'Inverter ID'},

    title='AC Power Generated in Plant 1 based on date',

    height = 650

)





fig.show()
data=solar02.groupby(['source_key','date']).sum().reset_index()

data['source_key'] = data['source_key'].apply(lambda x: solar02_inverter_id[x])



fig=px.bar( 

    data_frame = data,

    x = data['date'],

    y = data['ac_power'],

    color = 'source_key',

    color_discrete_sequence = px.colors.qualitative.Light24,

    hover_data = {'date':True,

                  'source_key':True,

                  'ac_power':':,.0f',

                 },

    opacity = 0.8,

    labels={'date':'date',

            'ac_power':'AC Power Generated (kW)',

            'source_key':'Inverter ID'},

    title='AC Power Generated in Plant 2 based on date',

    height = 650

)





fig.show()
data1=solar01

data2=solar02



fig = go.Figure()



fig.add_trace(go.Scattergl(x=data1['time'], 

                         y=data1['dc_power'],

                         mode='markers',

                         marker=dict(

                             size=4,

                             color= data1['dc_power'],

                             cauto=True,

                             colorscale ='Oryel',

                             opacity=0.3

                         ),

                         name='Plant 1 DC power'))



fig.add_trace(go.Scatter(x=data1['time'], 

                         y=data1.groupby('time').mean()['dc_power'],

                         mode='lines',

                             line=dict(

                             color='DarkGray',

                             width=3

                         ),

                         name='Plant 1 Mean'))



fig.add_trace(go.Scattergl(x=data2['time'], 

                         y=data2['dc_power'],

                         mode='markers',

                         marker=dict(

                             size=4,

                             color= data2['dc_power'],

                             cauto=True,

                             colorscale ='Blugrn',

                             opacity=0.3

                         ),

                         name='Plant 2 DC power'))



fig.add_trace(go.Scatter(x=data2['time'], 

                         y=data2.groupby('time').mean()['dc_power'],

                         mode='lines',

                             line=dict(

                             color='DarkOliveGreen',

                             width=3

                         ),

                         name='Plant 2 Mean'))



fig.update_layout(title= 'DC Power Generation by time',

                  height = 600)

fig.show()
data1=solar01

data2=solar02



fig = go.Figure()



fig.add_trace(go.Scattergl(x=data1['time'], 

                         y=data1['ac_power'],

                         mode='markers',

                         marker=dict(

                             size=4,

                             color= data1['ac_power'],

                             cauto=True,

                             colorscale ='Oryel',

                             opacity=0.3

                         ),

                         name='Plant 1 AC power'))



fig.add_trace(go.Scatter(x=data1['time'], 

                         y=data1.groupby('time').mean()['ac_power'],

                         mode='lines',

                             line=dict(

                             color='DarkGray',

                             width=3

                         ),

                         name='Plant 1 Mean'))



fig.add_trace(go.Scattergl(x=data2['time'], 

                         y=data2['ac_power'],

                         mode='markers',

                         marker=dict(

                             size=4,

                             color= data2['ac_power'],

                             cauto=True,

                             colorscale ='Blugrn',

                             opacity=0.3

                         ),

                         name='Plant 2 AC power'))



fig.add_trace(go.Scatter(x=data2['time'], 

                         y=data2.groupby('time').mean()['ac_power'],

                         mode='lines',

                             line=dict(

                             color='DarkOliveGreen',

                             width=3

                         ),

                         name='Plant 2 Mean'))



fig.update_layout(title= 'AC Power Generation by time',

                  height = 600)

fig.show()
plt.figure(figsize=(10,8))

sns.heatmap(solar01_with_sensor01.corr(), annot=True)
plt.figure(figsize=(10,8))

sns.heatmap(solar02_with_sensor02.corr(), annot=True)
data = full_data.groupby(['plant_ID','date','time']).mean().reset_index()

data = data.drop(['date','time'], axis=1)

sns.relplot(data=data, 

            x="ambient_temperature", 

            y="module_temperature", 

            hue='irradiation',

            size='irradiation',

            sizes=(50,200),

            palette='gist_heat',

            height=12,

            col='plant_ID'

           )

data = full_data.groupby(['plant_ID','date','time']).mean().reset_index()

data = data.drop(['date','time'], axis=1)

sns.relplot(data=data, 

            x="irradiation", 

            y="dc_power", 

            hue='ac_power',

            size='irradiation',

            sizes=(50,200),

            palette='gist_heat',

            height=12,

            col='plant_ID'

           )