# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls
import squarify
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_census = pd.read_csv("../input/acs2015_census_tract_data.csv")
df_census.head()
df_census.shape
df_census.describe()
df_census.info()
df_census.isnull().sum()
state_code = {'District of Columbia' : 'dc','Mississippi': 'MS', 'Oklahoma': 'OK', 'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 
                 'Arkansas': 'AR', 'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 'Idaho': 'ID', 'Wyoming': 'WY', 
                 'Tennessee': 'TN', 'Arizona': 'AZ', 'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 'Virginia': 'VA',
                 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV',
                 'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI', 'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 
                 'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO',
                 'Ohio': 'OH', 'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 'Colorado': 'CO', 'New Jersey': 'NJ', 
                 'Washington': 'WA', 'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 'Nevada': 'NV', 'Maine': 'ME', 'Puerto Rico':'PRI'}
df_census['StateCode'] = df_census['State'].apply(lambda x : state_code[x])
df_census.head()

states_list = df_census['State'].nunique()
states_list

df_state = df_census[['State','StateCode','TotalPop']].groupby(['State','StateCode']).sum().sort_values('TotalPop', ascending=0).reset_index()
print('Total Population of the United States =', df_state['TotalPop'].sum())
df_state
#states = df_state['State']
colorscl = [[0, 'rgb(166,206,227)'], [0.25, 'rgb(31,120,180)'], [0.45, 'rgb(178,223,138)'], [0.65, 'rgb(51,160,44)'],
            [0.85, 'rgb(251,154,153)'], [1, 'rgb(227,26,28)']]
data = [ dict(
        type='choropleth',
        colorscale = colorscl,
        autocolorscale = False,
        locations = df_state['StateCode'],
        z = df_state['TotalPop'],
        locationmode = 'USA-states',
        text = df_state['State'],
        marker = dict(
                      line = dict (
                      color = 'rgb(250,250,250)',
                      width = 2)),
                colorbar = dict(title = "Population(in Millions)"))]

layout = dict( title = 'State wise Population Distribution ',
               geo = dict(
              scope='usa',
              projection=dict( type='albers usa' ),
              showlakes = True,
              lakecolor = 'rgb(255,255,255)'))
    
fig = dict(data=data, layout=layout)
iplot(fig, filename='d1-cloropleth-map')
plt.figure(figsize=(18,16))
sns.set_style("darkgrid")
sns.barplot(y=df_state['State'], x=df_state['TotalPop'], linewidth=1, edgecolor="k"*len(df_state), palette='YlGnBu_r') 
plt.grid(True)
plt.xticks(rotation=90)
plt.xlabel('Population', fontsize=15, color='#191970')
plt.ylabel('States', fontsize=15, color='#191970')
plt.title('State Distribution', fontsize=15, color='#191970')
plt.show()
df_pop = df_census[['CensusTract','State','County', 'TotalPop','Hispanic','White','Black', 'Native','Asian','Pacific']]

columns_in_percent = ['Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific']

for i in columns_in_percent:
    df_pop[i] = round((df_pop[i]*df_pop['TotalPop'])/100)

df_pop.head()
df_white1 = df_pop[['TotalPop', 'White','Black','Hispanic','Native','Asian','Pacific']].sum()
df_white1

df_hispanic1 = df_pop[['State', 'White']].groupby(['State']).sum().sort_values('White', ascending=0).reset_index()
df_hispanic1
state_to_code = {'District of Columbia' : 'dc','Mississippi': 'MS', 'Oklahoma': 'OK', 'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 
                 'Arkansas': 'AR', 'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 'Idaho': 'ID', 'Wyoming': 'WY', 
                 'Tennessee': 'TN', 'Arizona': 'AZ', 'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 'Virginia': 'VA',
                 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV',
                 'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI', 'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 
                 'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO',
                 'Ohio': 'OH', 'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 'Colorado': 'CO', 'New Jersey': 'NJ', 
                 'Washington': 'WA', 'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 'Nevada': 'NV', 'Maine': 'ME', 'Puerto Rico':'PRI'}
df_hispanic1['state_code'] = df_hispanic1['State'].apply(lambda x : state_to_code[x])
df_hispanic1
states = df_hispanic1['State']

data = [ dict(
        type='choropleth',
        colorscale = 'Reds',
        autocolorscale = False,
        locations = df_hispanic1['state_code'],
        z = df_hispanic1['White'],
        locationmode = 'USA-states',
        text = df_hispanic1['State'],
        marker = dict(
                      line = dict (
                      color = 'rgb(250,250,250)',
                      width = 2)),
                colorbar = dict(title = "White Population Distribution"))]

layout = dict( title = 'State wise ',
               geo = dict(
              scope='usa',
              projection=dict( type='albers usa' ),
              showlakes = True,
              lakecolor = 'rgb(255,255,255)'))
    
fig = dict(data=data, layout=layout)
iplot(fig, filename='d3-cloropleth-map')

layout

df_white = df_pop[['State','TotalPop', 'White']].groupby(['State']).sum().sort_values('White', ascending=0).reset_index()
df_black = df_pop[['State','TotalPop', 'Black']].groupby(['State']).sum().sort_values('Black', ascending=0).reset_index()
df_hispanic = df_pop[['State','TotalPop', 'Hispanic']].groupby(['State']).sum().sort_values('Hispanic', ascending=0).reset_index()
df_native = df_pop[['State','TotalPop', 'Native']].groupby(['State']).sum().sort_values('Native', ascending=0).reset_index()

plt.figure(figsize=(16,30))
sns.set_style("darkgrid")

plt.subplot(4,1,1)
white_graph = sns.barplot(y=df_white['State'].head(25), x=df_white['White'], linewidth=1, edgecolor="k"*len(df_white), palette='Blues_r')
white_graph.set_xlabel('White Population', fontsize=15, color='#191970')
white_graph.set_ylabel('States', fontsize=15, color='#191970')
white_graph.set_title('White State Distribution', fontsize=15, color='#191970')

plt.subplot(4,1,2)
hispanic_graph = sns.barplot(y=df_hispanic['State'].head(25), x=df_hispanic['Hispanic'], linewidth=1, edgecolor="k"*len(df_hispanic), palette='OrRd_r')
hispanic_graph.set_xlabel('Hispanic Population', fontsize=15, color='#B20000')
hispanic_graph.set_ylabel('States', fontsize=15, color='#B20000')
hispanic_graph.set_title('Hispanic State Distribution', fontsize=15, color='#B20000')

plt.subplot(4,1,3)
black_graph = sns.barplot(y=df_black['State'].head(25), x=df_black['Black'], linewidth=1, edgecolor="k"*len(df_black), palette='BuPu_r')
black_graph.set_xlabel('Black Population', fontsize=15, color='#2f4f4f')
black_graph.set_ylabel('States', fontsize=15, color='#2f4f4f')
black_graph.set_title('Black State Distribution', fontsize=15, color='#2f4f4f')

plt.subplot(4,1,4)
native_graph = sns.barplot(y=df_native['State'].head(25), x=df_native['Native'], linewidth=1, edgecolor="k"*len(df_native), palette='Spectral')
native_graph.set_xlabel('Native Population', fontsize=15, color='#4A6381')
native_graph.set_ylabel('States', fontsize=15, color='#4A6381')
native_graph.set_title('Native State Distribution', fontsize=15, color='#4A6381')


plt.subplots_adjust(hspace = 0.5,top = 0.9)
plt.show()
df_m_w = df_census[['State','County', 'TotalPop', 'Men','Women']]
df1 = df_m_w[['State', 'Men','Women']].groupby(['State']).sum().reset_index()
df1.head()
df_income = df_census[['State','StateCode','County','TotalPop','Income','IncomeErr','IncomePerCap','IncomePerCapErr']]
df_income.head(20)
df_county = df_income[['State','StateCode', 'County','Income']].groupby(['State','StateCode', 'County']).median().sort_values('Income', ascending=0).reset_index()
df_county
newyorkcity = ['New York', 'Bronx', 'Queens', 'Kings', 'Richmond']
df_nyc = df_county.loc[(df_county['State'] == 'New York') & (df_county['County'].isin(newyorkcity))]
df_nyc

plt.figure(figsize=(8,4))
sns.set_style("darkgrid")
sns.barplot(y=df_nyc['County'], x=df_nyc['Income'], linewidth=1, edgecolor="k"*len(df_state), palette='Blues_r') #YlGnBu_r
plt.grid(True)
plt.xticks(rotation=20)
plt.xlabel('Income($)', fontsize=15, color='#191970')
plt.ylabel('New York City Counties', fontsize=15, color='#191970')
plt.title('Income Distribution', fontsize=15, color='#191970')
plt.show()
df_state  = df_county[['State','StateCode','Income']].groupby(['State','StateCode']).median().sort_values('Income', ascending=0).reset_index()
df_state
df_national = df_state.median()
df_national