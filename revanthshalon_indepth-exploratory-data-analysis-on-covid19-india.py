# Basic Libraries

import numpy as np

import pandas as pd



# Plotting and Visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots



# Datetime 

import warnings

warnings.filterwarnings('ignore')
# Importing data

data = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')
# Checking the shape of the data

data.shape
# Checking for total number of null values

data.isnull().sum()
# Checking datatypes of each column

data.dtypes
# Changing the datatype of Date from Object to Datetime

data.Date = pd.to_datetime(data.Date,dayfirst=True,errors='coerce')

# data.ConfirmedForeignNational = data.ConfirmedForeignNational.astype(np.int64)

# data.ConfirmedIndianNational = data.ConfirmedIndianNational.astype(np.int64)
# Checking the dtype of columns

data.dtypes
# Dropping the columns

data.drop(['Sno','Time'],axis=1,inplace=True)
data['State/UnionTerritory'].unique()
data[data['State/UnionTerritory']=='Unassigned']
data.drop([500,528,617],inplace=True) # Dropped the unassigned values
# Checking the Data head

data.head(3)
ind = data.groupby('Date').agg({'Confirmed':'sum','Cured':'sum','Deaths':'sum'})

ind['SpreadRate'] = 0

for i in ind.index:

  ind.loc[i,'SpreadRate'] = (ind.loc[ind.index.min(),'Confirmed'] if i ==  ind.index.min() else ind.loc[i,'Confirmed'] - ind.loc[i-np.timedelta64(1,'D'),'Confirmed'])

ind['ActiveCases'] = ind['Confirmed'] - ind['Cured']

fig = make_subplots(rows=3,cols=2,specs=[[{},{}],[{'rowspan':2,'colspan':2},None],[None,None]],subplot_titles=['India Spread Rate','Active Cases','Confirmed vs Deaths vs Cured'])

# Spread Rate in India

fig.add_trace(go.Scatter(name='Spread Rate in India',x=ind.index,y=ind.SpreadRate,mode='markers+lines'),row=1,col=1)

fig.update_xaxes(title_text='Date',row=1,col=1)

fig.update_yaxes(title_text='Spread Rate',row=1,col=1)

# Active Cases in India

fig.add_trace(go.Scatter(name='Active Cases in India',x=ind.index,y=ind.ActiveCases,mode='markers+lines'),row=1,col=2)

fig.update_xaxes(title_text='Date',row=1,col=2)

fig.update_yaxes(title_text='Active Cases in India',row=1,col=2)

# Bar plot Comparison

ht = [f'Confirmed Cases:{i}' for i in ind.Confirmed]

fig.add_trace(go.Bar(x=ind.index,y=ind.Confirmed,name='Confirmed Cases',hovertext=ht),row=2,col=1)

ht = [f'Cured Cases:{i}' for i in ind.Cured]

fig.add_trace(go.Bar(x=ind.index,y=ind.Cured,name='Cured Cases',hovertext=ht),row=2,col=1)

ht = [f'Deceased Cases:{i}' for i in ind.Deaths]

fig.add_trace(go.Bar(x=ind.index,y=ind.Deaths,name='Deceased Cases',hovertext=ht),row=2,col=1)

fig.update_xaxes(title_text='Date')

fig.update_yaxes(title_text='Number of cases')

# Layout

fig.update_layout(title_text='Spread Rate vs Active Cases in India',barmode='stack',height=950)

fig.show()
latest_data = data.groupby('Date').get_group(data.Date.max())[['Date','State/UnionTerritory','Cured','Deaths','Confirmed']]

latest_data['ActiveCases'] = latest_data['Confirmed'] - latest_data['Cured'] - latest_data['Deaths']

latest_data.sort_values(by='ActiveCases',ascending=False,inplace=True)

top3 = list(latest_data.head(3)['State/UnionTerritory'])

display(latest_data.style.background_gradient(cmap='summer',subset=['ActiveCases']))

print()

print(f'States with Highest Active cases are {top3}')
# Storing group data in state_wise_groups

state_wise_groups = data.groupby('State/UnionTerritory')
'''Defining a function to give the stats of each State'''

def state_details(state_name,start_date=data.Date.min(),end_date=data.Date.max()):



  '''Single State'''

  if isinstance(state_name,str):

    state = state_wise_groups.get_group(state_name).reset_index(drop=True)

    state = state[(state.Date>=np.datetime64(start_date))&(state.Date<=np.datetime64(end_date))]

    state['SpreadRate'] = 0

    for i in state.index:

      state.loc[i,'SpreadRate'] = (state.loc[i,'Confirmed'] if i == state.index.min() else state.loc[i,'Confirmed']-state.loc[i-1,'Confirmed'])

    state['ActiveCases'] = state['Confirmed'] - state['Cured'] - state['Deaths']



    fig = make_subplots(rows=2,cols=2,specs=[[{},{}],[{'colspan':2},None]],subplot_titles=[f'Spread Rate over a Period of {state.Date.max() - state.Date.min()} in {state_name}',f'Active Cases in {state_name}',f'Confirmed vs Deceased and Recovered Cases in {state_name}'])



    # Spread Rate in the Given State

    fig.add_trace(go.Scatter(name='Spread Cases per Day',x = state.Date,y = state.SpreadRate,mode='markers+lines'),row=1,col=1)

    fig.update_xaxes(title_text='Dates',row=1,col=1)

    fig.update_yaxes(title_text='Spread Rate',row=1,col=1)



    # Active Cases in Given State

    fig.add_trace(go.Scatter(name='Active Cases',x = state.Date,y = state.ActiveCases,mode='markers+lines'),row=1,col=2)

    fig.update_xaxes(title_text='Dates',row=1,col=2)

    fig.update_yaxes(title_text='Active Cases',row=1,col=2)



    # Bar Plot

    ht = [f'Confirmed Cases:{i}' for i in state.Confirmed]

    fig.add_trace(go.Bar(x=state.Date,y=state.Confirmed,name='Confirmed Cases',hovertext=ht),row=2,col=1)

    ht = [f'Cured Cases:{i}' for i in state.Cured]

    fig.add_trace(go.Bar(x=state.Date,y=state.Cured,name='Cured Cases',hovertext=ht),row=2,col=1)

    ht = [f'Deceased Cases:{i}' for i in state.Deaths]

    fig.add_trace(go.Bar(x=state.Date,y=state.Deaths,name='Deceased Cases',hovertext=ht),row=2,col=1)



    fig.update_layout(title_text=f'Spread Rate and Comparison of Cured vs Deceased vs Confirmed in {state_name}',height=950,barmode='stack')

    fig.show()



    # Showing Average Spread Rate in the Given Time Period

    print(f'Average Spread Rate in {state_name} is {state.SpreadRate.mean()}')

  

  elif isinstance(state_name,list):

    fig = make_subplots(rows=2,cols=2,subplot_titles=['Active Cases','Spread Rate','Deceased Cases','Cured Cases'])

    state_color=['purple', 'red', 'rosybrown','royalblue', 'saddlebrown', 'salmon', 'sandybrown','seagreen', 'seashell', 'sienna', 'silver', 'skyblue']

    count=0

    # State-wise plotting

    for s in set(state_name):

      state = state_wise_groups.get_group(s).reset_index(drop=True)

      state = state[(state.Date>=np.datetime64(start_date))&(state.Date<=np.datetime64(end_date))]

      state['SpreadRate'] = 0

      for i in state.index:

        state.loc[i,'SpreadRate'] = (state.loc[i,'Confirmed'] if i == state.index.min() else state.loc[i,'Confirmed']-state.loc[i-1,'Confirmed'])

      state['ActiveCases'] = state['Confirmed'] - state['Cured']

      c=state_color[count] 

      count+=1

      # Active Cases

      ht = [f'{s}:{i}' for i in state.ActiveCases]

      fig.add_trace(go.Scatter(name=f'{s}',x=state.Date,y=state.ActiveCases,hovertext=ht,mode='markers+lines',legendgroup='state',showlegend=True,line=dict(color=c)),row=1,col=1)



      # Spread Rate

      ht = [f'{s}:{i}' for i in state.SpreadRate]

      fig.add_trace(go.Scatter(name=f'{s}',x=state.Date,y=state.SpreadRate,hovertext=ht,mode='markers+lines',legendgroup='state',showlegend=False,line=dict(color=c)),row=1,col=2)



      # Deceased Cases

      ht = [f'{s}:{i}' for i in state.Deaths]

      fig.add_trace(go.Scatter(name=f'{s}',x=state.Date,y=state.Deaths,hovertext=ht,mode='markers+lines',legendgroup='state',showlegend=False,line=dict(color=c)),row=2,col=1)



      # Cured Cases

      ht = [f'{s}:{i}' for i in state.Cured]

      fig.add_trace(go.Scatter(name=f'{s}',x=state.Date,y=state.Cured,hovertext=ht,mode='markers+lines',legendgroup='state',showlegend=False,line=dict(color=c)),row=2,col=2)

    

    # X and Y axis labels for Each Subplots

    fig.update_xaxes(title_text='Date',row=1,col=1)

    fig.update_yaxes(title_text='Active Cases',row=1,col=1)

    fig.update_xaxes(title_text='Date',row=1,col=2)

    fig.update_yaxes(title_text='Spread Rate',row=1,col=2)

    fig.update_xaxes(title_text='Date',row=2,col=1)

    fig.update_yaxes(title_text='Deceased Cases',row=2,col=1)

    fig.update_xaxes(title_text='Date',row=2,col=2)

    fig.update_yaxes(title_text='Cured Cases',row=2,col=2)



    # Layout properties

    fig.update_layout(title_text=f'Comparison between {state_name}',height=950)

    fig.show()
# Spread Rate from the day of first detection

state_details('Kerala')
# Spread Rate in Kerala for past 7 days

state_details('Kerala','2020-03-29')
# Spread Rate in Tamil Nadu since First Detection

state_details('Tamil Nadu')
# Spread Rate in Tamil Nadu Over past 7 Days

state_details('Tamil Nadu','2020-03-29')
state_details('Maharashtra')
# Spread Rate in Maharashtra Over past 7 Days

state_details('Maharashtra','2020-03-29')
state_details('Delhi')
state_details('Delhi',data.Date.max()-pd.Timedelta(7,'D'))
state_details(top3)
state_details(top3,start_date=data.Date.max()-pd.Timedelta(7,'D'))
data['SpreadRate'] = 0

for s in data['State/UnionTerritory'].unique():

  state = state_wise_groups.get_group(s)

  state['SpreadRate'] = 0

  state['CureRate'] = 0

  state['DeathRate'] = 0

  for i in state.index:

    data.loc[i,'SpreadRate']=state[state.Date == state.Date.min()]['Confirmed'].values if state.loc[i,'Date']==state.Date.min() else state.loc[i,'Confirmed']-state.loc[state[state.Date==state.loc[i,'Date']-pd.Timedelta(1,'D')].index.values,'Confirmed'].values

    data.loc[i,'CureRate']=state[state.Date == state.Date.min()]['Cured'].values if state.loc[i,'Date']==state.Date.min() else state.loc[i,'Cured']-state.loc[state[state.Date==state.loc[i,'Date']-pd.Timedelta(1,'D')].index.values,'Cured'].values

    data.loc[i,'DeathRate']=state[state.Date == state.Date.min()]['Deaths'].values if state.loc[i,'Date']==state.Date.min() else state.loc[i,'Deaths']-state.loc[state[state.Date==state.loc[i,'Date']-pd.Timedelta(1,'D')].index.values,'Deaths'].values
# Average Spread rate,Cure Rate,Death Rate for each state

arps = []

for s in data['State/UnionTerritory'].unique():

  state = state_wise_groups.get_group(s)

  arps.append({'State':s,'Average Spread Rate':state['SpreadRate'].mean(),'Average Cure Rate':state['CureRate'].mean(),'Average Death Rate':state['DeathRate'].mean()})

averages_statewise = pd.DataFrame(arps)
# Averages for the Entirity 

print('Total Averages for all the States')

averages_statewise.sort_values(by='Average Spread Rate',ascending=False).style.background_gradient(cmap='summer')
# Averages for the past 7 Days

arps = []

for s in data['State/UnionTerritory'].unique():

  state = state_wise_groups.get_group(s)

  state = state[(state.Date.max()>state.Date)&(state.Date>=state.Date.max()-pd.Timedelta(7,'D'))]

  arps.append({'State':s,'Average Spread Rate':state['SpreadRate'].mean(),'Average Cure Rate':state['CureRate'].mean(),'Average Death Rate':state['DeathRate'].mean()})

weekly_average = pd.DataFrame(arps)
print('Weekly Average for All States')

weekly_average.sort_values(by='Average Spread Rate',ascending=False).style.background_gradient(cmap='summer')