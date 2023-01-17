import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
from IPython.display import HTML
import plotly.offline as py

df = pd.read_csv('../input/VietnamConflict.csv')

state_pop = {'ME':0.982,'NH':0.691,'VT':0.42,'MA':5.434,'RI':0.901,'CT':2.918,'NY':18.023,'NJ':6.981,'PA':11.672,'OH':10.488,
'IN':5.012,'IL':10.887,'MI':8.608,'WI':4.914,'MN':3.625,'IA':2.772,'MS':4.587,'ND':0.632,'SD':0.668,'NE':1.443,'KS':2.281,'DE':0.524,
'MD':3.68,'DC':0.808,'VA':4.541,'WV':1.807,'NC':5.509,'SC':2.638,'GA':4.49,'FL':6.035,'KY':3.201,'TN':3.926,'AL':3.533,'MO':2.344,
'AR':1.972,'LA':3.633,'OK':2.516,'TX':10.858,'MT':0.699,'ID':0.703,'WY':0.319,'CO':2.012,'NM':1.002,'AZ':1.637,'UT':1.022,'NV':0.436,
'WA':3.208,'OR':1.981,'CA':18.992,'AK':0.271,'HI':0.76}

df_states = pd.DataFrame(list(state_pop.items()))
df_states.columns = ['STATE_CODE', 'POPULATION']
print(df_states.head(5))
death_by_state = df['STATE_CODE'].value_counts().reset_index().rename(columns={'index': 'STATE_CODE', 'STATE_CODE': 'COUNT'})

df_normalised = pd.merge(df_states,death_by_state,on='STATE_CODE',how='left')
df_normalised['NORMALISED_COUNT'] = df_normalised['COUNT']/df_normalised['POPULATION']
df_normalised['NORMALISED_COUNT'] = df_normalised['NORMALISED_COUNT'].round()

for col in df_normalised.columns:
    df_normalised[col] = df_normalised[col].astype(str)
py.init_notebook_mode(connected=True)
scl = [[0.0, 'rgb(204, 255, 204)'],[0.2, 'rgb(102, 255, 102)'],[0.4, 'rgb(0, 230, 0)'], [0.6, 'rgb(0, 153, 0)'],[0.8, 'rgb(0, 77, 0)'],[1.0, 'rgb(0, 26, 0)']]

labels = df_normalised['STATE_CODE']
values = df_normalised['NORMALISED_COUNT']


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = labels,
        z = np.array(values).astype(float),
        locationmode = 'USA-states',
        text = labels,
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "US casualties")
        ) ]

layout = dict(
        title = 'US casualties in Vietnam war<br>(Normalised by approximate 1967 state pop)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='US_Vietnam_war_casualties.html' )
df_mo = df[df['STATE_CODE']=='MO']
print('number of fatalities from Missouri = ' + str(len(df_mo)))
df_ms = df[df['STATE_CODE']=='MS']
print('number of fatalities from Mississippi = ' + str(len(df_ms)))
df_mc = df[df['BRANCH']=='MARINE CORPS']

death_by_state = df_mc['STATE_CODE'].value_counts().reset_index().rename(columns={'index': 'STATE_CODE', 'STATE_CODE': 'COUNT'})

df_normalised = pd.merge(df_states,death_by_state,on='STATE_CODE',how='left')
df_normalised['NORMALISED_COUNT'] = df_normalised['COUNT']/df_normalised['POPULATION']
df_normalised['NORMALISED_COUNT'] = df_normalised['NORMALISED_COUNT'].round()

for col in df_normalised.columns:
    df_normalised[col] = df_normalised[col].astype(str)
    
import plotly.offline as py
py.init_notebook_mode(connected=True)
scl = [[0.0, 'rgb(204, 255, 204)'],[0.2, 'rgb(102, 255, 102)'],[0.4, 'rgb(0, 230, 0)'], [0.6, 'rgb(0, 153, 0)'],[0.8, 'rgb(0, 77, 0)'],[1.0, 'rgb(0, 26, 0)']]

labels = df_normalised['STATE_CODE']
values = df_normalised['NORMALISED_COUNT']


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = labels,
        z = np.array(values).astype(float),
        locationmode = 'USA-states',
        text = labels,
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "US casualties")
        ) ]

layout = dict(
        title = 'US Marine Corps casualties in Vietnam war<br>(Normalised by approximate 1967 state pop)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='US_Vietnam_war_casualties.html' )
df_army = df[df['BRANCH']=='ARMY']

death_by_state = df_army['STATE_CODE'].value_counts().reset_index().rename(columns={'index': 'STATE_CODE', 'STATE_CODE': 'COUNT'})

df_normalised = pd.merge(df_states,death_by_state,on='STATE_CODE',how='left')
df_normalised['NORMALISED_COUNT'] = df_normalised['COUNT']/df_normalised['POPULATION']
df_normalised['NORMALISED_COUNT'] = df_normalised['NORMALISED_COUNT'].round()

for col in df_normalised.columns:
    df_normalised[col] = df_normalised[col].astype(str)


py.init_notebook_mode(connected=True)
scl = [[0.0, 'rgb(204, 255, 204)'],[0.2, 'rgb(102, 255, 102)'],[0.4, 'rgb(0, 230, 0)'], [0.6, 'rgb(0, 153, 0)'],[0.8, 'rgb(0, 77, 0)'],[1.0, 'rgb(0, 26, 0)']]

labels = df_normalised['STATE_CODE']
values = df_normalised['NORMALISED_COUNT']


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = labels,
        z = np.array(values).astype(float),
        locationmode = 'USA-states',
        text = labels,
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "US casualties")
        ) ]

layout = dict(
        title = 'US Army casualties in Vietnam war<br>(Normalised by approximate 1967 state pop)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='US_Vietnam_war_casualties.html' )
df_af = df[df['BRANCH']=='AIR FORCE']

death_by_state = df_af['STATE_CODE'].value_counts().reset_index().rename(columns={'index': 'STATE_CODE', 'STATE_CODE': 'COUNT'})

df_normalised = pd.merge(df_states,death_by_state,on='STATE_CODE',how='left')
df_normalised['NORMALISED_COUNT'] = df_normalised['COUNT']/df_normalised['POPULATION']
df_normalised['NORMALISED_COUNT'] = df_normalised['NORMALISED_COUNT'].round()

for col in df_normalised.columns:
    df_normalised[col] = df_normalised[col].astype(str)

py.init_notebook_mode(connected=True)
scl = [[0.0, 'rgb(204, 255, 204)'],[0.2, 'rgb(102, 255, 102)'],[0.4, 'rgb(0, 230, 0)'], [0.6, 'rgb(0, 153, 0)'],[0.8, 'rgb(0, 77, 0)'],[1.0, 'rgb(0, 26, 0)']]

labels = df_normalised['STATE_CODE']
values = df_normalised['NORMALISED_COUNT']


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = labels,
        z = np.array(values).astype(float),
        locationmode = 'USA-states',
        text = labels,
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "US casualties")
        ) ]

layout = dict(
        title = 'Air Force casualties in Vietnam war<br>(Normalised by approximate 1967 state pop)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='US_Vietnam_war_casualties.html' )
df_n = df[df['BRANCH']=='NAVY']

death_by_state = df_n['STATE_CODE'].value_counts().reset_index().rename(columns={'index': 'STATE_CODE', 'STATE_CODE': 'COUNT'})

df_normalised = pd.merge(df_states,death_by_state,on='STATE_CODE',how='left')
df_normalised['NORMALISED_COUNT'] = df_normalised['COUNT']/df_normalised['POPULATION']
df_normalised['NORMALISED_COUNT'] = df_normalised['NORMALISED_COUNT'].round()

for col in df_normalised.columns:
    df_normalised[col] = df_normalised[col].astype(str)

py.init_notebook_mode(connected=True)
scl = [[0.0, 'rgb(204, 255, 204)'],[0.2, 'rgb(102, 255, 102)'],[0.4, 'rgb(0, 230, 0)'], [0.6, 'rgb(0, 153, 0)'],[0.8, 'rgb(0, 77, 0)'],[1.0, 'rgb(0, 26, 0)']]

labels = df_normalised['STATE_CODE']
values = df_normalised['NORMALISED_COUNT']


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = labels,
        z = np.array(values).astype(float),
        locationmode = 'USA-states',
        text = labels,
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "US casualties")
        ) ]

layout = dict(
        title = 'Navy casualties in Vietnam war<br>(Normalised by approximate 1967 state pop)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='US_Vietnam_war_casualties.html' )