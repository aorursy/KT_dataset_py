import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import matplotlib.ticker as ticker

import matplotlib.animation as animation

from IPython.display import HTML

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode(connected=True)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
state = pd.read_csv('/kaggle/input/crimeanalysis/crime_by_state.csv')

district = pd.read_csv('/kaggle/input/crimeanalysis/crime_by_district.csv')

print('Shape of state is {} \nShape of district is {}'.format(state.shape,district.shape))
state.sample(5)
state['STATE/UT'].unique()
state = state[state['STATE/UT'] != 'TOTAL (STATES)']

state = state[state['STATE/UT'] != 'TOTAL (ALL-INDIA)']

state = state[state['STATE/UT'] != 'TOTAL (UTs)']
def animation_graphs(df):

    fig = px.scatter(state, x="STATE/UT", y="Murder",animation_frame="Year",size='Murder',color="Murder",range_y=[0,450])

    fig.show()

    

    fig = px.scatter(state, x="STATE/UT", y="Assault on women",animation_frame="Year",size='Assault on women',color="Assault on women",range_y=[0,430])

    fig.show()

    

    fig = px.scatter(state, x="STATE/UT", y="Kidnapping and Abduction",animation_frame="Year",size='Kidnapping and Abduction',color="Kidnapping and Abduction",range_y=[0,370])

    fig.show()

    

    fig = px.scatter(state, x="STATE/UT", y="Dacoity",animation_frame="Year",size='Dacoity',color="Dacoity",range_y=[0,25])

    fig.show()

    

    fig = px.scatter(state, x="STATE/UT", y="Robbery",animation_frame="Year",size='Robbery',color="Robbery",range_y=[0,90])

    fig.show()

    

    fig = px.scatter(state, x="STATE/UT", y="Arson",animation_frame="Year",size='Arson',color="Arson",range_y=[0,190])

    fig.show()

    

    fig = px.scatter(state, x="STATE/UT", y="Hurt",animation_frame="Year",size='Hurt',color="Hurt",range_y=[0,1300])

    fig.show()

    

    fig = px.scatter(state, x="STATE/UT", y="Prevention of atrocities (POA) Act",

                     animation_frame="Year",size='Prevention of atrocities (POA) Act',

                     color="Prevention of atrocities (POA) Act",range_y=[0,5000])

    fig.show()

    

    fig = px.scatter(state, x="STATE/UT", y="Protection of Civil Rights (PCR) Act",

                     animation_frame="Year",size='Protection of Civil Rights (PCR) Act',

                     color="Protection of Civil Rights (PCR) Act",range_y=[0,470])

    fig.show()

    

    fig = px.scatter(state, x="STATE/UT", y="Other Crimes Against SCs",animation_frame="Year",

                     size='Other Crimes Against SCs',color="Other Crimes Against SCs",range_y=[0,4800])

    fig.show()
animation_graphs(state)
def line_graphs():

    fig = px.line(state,x="Year", y="Murder", color="STATE/UT")

    fig.show()

    

    fig = px.line(state,x="Year", y="Assault on women", color="STATE/UT")

    fig.show()

    

    fig = px.line(state,x="Year", y="Kidnapping and Abduction", color="STATE/UT")

    fig.show()

    

    fig = px.line(state,x="Year", y="Dacoity", color="STATE/UT")

    fig.show()

    

    fig = px.line(state,x="Year", y="Robbery", color="STATE/UT")

    fig.show()

    

    fig = px.line(state,x="Year", y="Arson", color="STATE/UT")

    fig.show()

    

    fig = px.line(state,x="Year", y="Hurt", color="STATE/UT")

    fig.show()

    

    fig = px.line(state,x="Year", y="Prevention of atrocities (POA) Act", color="STATE/UT")

    fig.show()

    

    fig = px.line(state,x="Year", y="Protection of Civil Rights (PCR) Act", color="STATE/UT")

    fig.show()

    

    fig = px.line(state,x="Year", y="Other Crimes Against SCs", color="STATE/UT")

    fig.show()

    
line_graphs()
def plot_districts(df):

    #Dataframe for murder

    df = df[df['DISTRICT'] != 'TOTAL']

    district_murder = pd.DataFrame(df.groupby('DISTRICT')['Murder'].sum())

    district_murder = district_murder.sort_values('Murder', ascending=False)

    district_murder.reset_index(inplace=True)

    

    district_assault = pd.DataFrame(df.groupby('DISTRICT')['Assault on women'].sum())

    district_assault = district_assault.sort_values('Assault on women', ascending=False)

    district_assault.reset_index(inplace=True)

    

    district_kidnapping = pd.DataFrame(df.groupby('DISTRICT')['Kidnapping and Abduction'].sum())

    district_kidnapping = district_kidnapping.sort_values('Kidnapping and Abduction', ascending=False)

    district_kidnapping.reset_index(inplace=True)

        

    district_dacoity = pd.DataFrame(df.groupby('DISTRICT')['Dacoity'].sum())

    district_dacoity = district_dacoity.sort_values('Dacoity', ascending=False)

    district_dacoity.reset_index(inplace=True)

    

    district_robbery = pd.DataFrame(df.groupby('DISTRICT')['Robbery'].sum())

    district_robbery = district_robbery.sort_values('Robbery', ascending=False)

    district_robbery.reset_index(inplace=True)

    

    district_arson = pd.DataFrame(df.groupby('DISTRICT')['Arson'].sum())

    district_arson = district_arson.sort_values('Arson', ascending=False)

    district_arson.reset_index(inplace=True)

    

    district_hurt = pd.DataFrame(df.groupby('DISTRICT')['Hurt'].sum())

    district_hurt = district_hurt.sort_values('Hurt', ascending=False)

    district_hurt.reset_index(inplace=True)

    

    district_poa = pd.DataFrame(df.groupby('DISTRICT')['Prevention of atrocities (POA) Act'].sum())

    district_poa = district_poa.sort_values('Prevention of atrocities (POA) Act', ascending=False)

    district_poa.reset_index(inplace=True)

    

    district_pcr = pd.DataFrame(df.groupby('DISTRICT')['Protection of Civil Rights (PCR) Act'].sum())

    district_pcr = district_pcr.sort_values('Protection of Civil Rights (PCR) Act', ascending=False)

    district_pcr.reset_index(inplace=True)

    

    district_scst = pd.DataFrame(df.groupby('DISTRICT')['Other Crimes Against SCs'].sum())

    district_scst = district_scst.sort_values('Other Crimes Against SCs', ascending=False)

    district_scst.reset_index(inplace=True)

    

    values = district_murder['Murder'].head(5)

    labels = district_murder['DISTRICT'].head(5)

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

    fig.update_layout(title_text='Murder')

    fig.show()

    

    values = district_assault['Assault on women'].head(5)

    labels = district_assault['DISTRICT'].head(5)

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

    fig.update_layout(title_text='Assault on women')

    fig.show()

    

    values = district_kidnapping['Kidnapping and Abduction'].head(5)

    labels = district_kidnapping['DISTRICT'].head(5)

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

    fig.update_layout(title_text='Kidnapping and Abduction')

    fig.show()

    

    values = district_dacoity['Dacoity'].head(5)

    labels = district_dacoity['DISTRICT'].head(5)

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

    fig.update_layout(title_text='Dacoity')

    fig.show()

    

    values = district_robbery['Robbery'].head(5)

    labels = district_robbery['DISTRICT'].head(5)

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

    fig.update_layout(title_text='Robbery')

    fig.show()

    

    values = district_arson['Arson'].head(5)

    labels = district_arson['DISTRICT'].head(5)

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

    fig.update_layout(title_text='Arson')

    fig.show()

    

    values = district_hurt['Hurt'].head(5)

    labels = district_hurt['DISTRICT'].head(5)

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

    fig.update_layout(title_text='Hurt')

    fig.show()

    

    values = district_poa['Prevention of atrocities (POA) Act'].head(5)

    labels = district_poa['DISTRICT'].head(5)

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

    fig.update_layout(title_text='Prevention of atrocities (POA) Act')

    fig.show()

    

    values = district_pcr['Protection of Civil Rights (PCR) Act'].head(5)

    labels = district_pcr['DISTRICT'].head(5)

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

    fig.update_layout(title_text='Protection of Civil Rights (PCR) Act')

    fig.show()

    

    values = district_scst['Other Crimes Against SCs'].head(5)

    labels = district_scst['DISTRICT'].head(5)

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

    fig.update_layout(title_text='Other Crimes Against SCs')

    fig.show()
plot_districts(district)
district = district[district['DISTRICT'] != 'TOTAL']

district_serious = pd.DataFrame(district.groupby('DISTRICT')['Murder','Assault on women',

                                                       'Kidnapping and Abduction'].sum())

district_serious['Total'] = district_serious.sum(axis = 1, skipna = True) 

district_serious = district_serious.sort_values('Total', ascending=False)

district_serious.reset_index(inplace=True)
district_serious = district_serious.head(10)
fig = px.bar(district_serious, x="Total", y="DISTRICT", color='DISTRICT', orientation='h')

fig.show()