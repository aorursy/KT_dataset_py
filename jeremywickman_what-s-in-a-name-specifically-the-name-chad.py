#Import packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper
from bq_helper import BigQueryHelper

#Read the datset from BigQuery file
dataset = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="usa_names")
#What tables does this dataset contain?
dataset.list_tables()
#Take a look at one of the tables
dataset.head('usa_1910_current')
#How many unique names are in this datset every year?
query1 = """
SELECT 
COUNT(DISTINCT name) as unique_names,
year
FROM `bigquery-public-data.usa_names.usa_1910_current`
GROUP BY 2
ORDER BY year DESC
"""
output = dataset.query_to_pandas_safe(query1)

#Let's look at the last 5 years and the most recent 5 years of names contained in the table
print(output[:5])
print(output[-5:])
#Query Chad data into dataset
query = """
SELECT 
state,
gender,
year,
name,
number
FROM `bigquery-public-data.usa_names.usa_1910_current`
WHERE name="Chad"
"""
chad_df = dataset.query_to_pandas_safe(query)
#How many people in the US have ever named Chad?
answer = chad_df.number.sum()
print(answer, "...awesome")
#When was the first Chad?
answer = chad_df['year'].min()
print(answer, "...according to our dataset")
#Has there ever been a female Chad?
answer = 'F' in chad_df.gender.values
print(answer, "...surprisingly")
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
%matplotlib inline
import seaborn as sns

chad_time = chad_df.groupby('year')['number'].sum()
chad_time = chad_time.reset_index()

#Note some important dates
first_chad = chad_time['year'].min()
peak_chad = chad_time[chad_time['number'] == chad_time['number'].max()]
chad_dydx = chad_time[chad_time['number'].diff() == chad_time['number'].diff().max()]
chad_fall = chad_time[chad_time['number'].diff() == chad_time['number'].diff().min()]
last_chad = chad_time['year'].max()
#Plot the graph
ax = chad_time.plot(x='year',y='number',figsize=(16,10))
ax.set_title("Chad Over Time")
ax.set_ylabel('Number of Chads Born')

#Adding important commentary
ax.annotate('The first recorded Chad', 
            xy=(1917, 8), 
            xytext=(1920, 2000),
            arrowprops=dict(facecolor='black', 
                            shrink=0.05)
            )
ax.annotate('The second recorded Chad', 
            xy=(first_chad, 8), 
            xytext=(1930, 1000),
            arrowprops=dict(facecolor='black', 
                            shrink=0.05)
            )
ax.annotate('A lot of people with poor taste start having sex', 
            xy=(1965, 1000), 
            xytext=(1950, 3000),
            arrowprops=dict(facecolor='black', 
                            shrink=0.05)
            )
ax.annotate('Peak Chad', 
            xy=(peak_chad['year'], peak_chad['number']), 
            xytext=(peak_chad['year'] - 20, peak_chad['number']),
            arrowprops=dict(facecolor='black', 
                            shrink=0.05)
            )
ax.annotate('Maximum Rate of Chad', 
            xy=(chad_dydx['year'], chad_dydx['number']), 
            xytext=(chad_dydx['year'] - 20, chad_dydx['number']),
            arrowprops=dict(facecolor='black', 
                            shrink=0.05)
            )
ax.annotate('Asteroid hits, presumably', 
            xy=(chad_fall['year'], chad_fall['number']), 
            xytext=(chad_fall['year'] + 20, chad_fall['number']),
            arrowprops=dict(facecolor='black', 
                            shrink=0.05)
            )
chad_state = chad_df.groupby('state')['number'].sum()
chad_state = chad_state.reset_index()
chad_state = chad_state.sort_values('number', ascending=False)
#Create plot of top 10, bottom 10 states by Chad count
fig, (ax1, ax2) = plt.subplots(nrows=2, sharey=True, sharex=False, figsize=(10,10))
sns.barplot(x='state', y='number', data=chad_state[:10], ax=ax1)
sns.barplot(x='state', y='number', data=chad_state[-10:], ax=ax2)
#Let's adjust these numbers to account for total count of the name population, because that's the right thing to do
query = """
SELECT 
state,
SUM(number) as total_people
FROM `bigquery-public-data.usa_names.usa_1910_current`
GROUP BY 1
ORDER BY state DESC
"""
total_people_state = dataset.query_to_pandas_safe(query)
merge2 = total_people_state.merge(chad_state, left_on='state', right_on='state', how='outer')
merge2['number'].fillna(0,inplace=True)
merge2['percent_chad']=merge2['number']/merge2['total_people']*100
merge2 = merge2.sort_values(by='percent_chad', ascending=False)
#Let's see this again, this time with the corrected proportion of Chads
fig, (ax1, ax2) = plt.subplots(nrows=2, sharey=True, sharex=False, figsize=(10,10))
sns.barplot(x='state', y='percent_chad', data=merge2[:10], ax=ax1)
sns.barplot(x='state', y='percent_chad', data=merge2[-10:], ax=ax2)
#Import from plotly
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()
#Set color scale
scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]
#Format columns to text
for col in merge2.columns:
    merge2[col] = merge2[col].astype(str)
#Add label column
merge2['text'] = merge2['state'] + '<br>' +\
    'Percent Chad ' + merge2['percent_chad']
#Set data and layout
data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = merge2['state'],
        z = merge2['percent_chad'].astype(float),
        locationmode = 'USA-states',
        text = merge2['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Percent Chad")
        ) ]

layout = dict(
        title = 'Percent of Babies Named Chad ACROSS ALL OF RECORDED TIME',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )

fig = dict( data=data, layout=layout )
iplot( fig, filename='d3-cloropleth-map' )
#Get number of Chads per year per state
query = """
SELECT 
year,
state,
SUM(number) as total_chads
FROM `bigquery-public-data.usa_names.usa_1910_current`
WHERE name="Chad"
GROUP BY 1,2
ORDER BY year ASC
"""
chad_year_state = dataset.query_to_pandas_safe(query)
#Get total number of people per year per state
query = """
SELECT 
year,
state,
SUM(number) as total_people
FROM `bigquery-public-data.usa_names.usa_1910_current`
GROUP BY 1,2
ORDER BY year ASC
"""
people_year_state = dataset.query_to_pandas_safe(query)
merge3 = people_year_state.merge(chad_year_state, left_on=['state','year'], right_on=['state','year'], how='outer')
merge3['total_chads'].fillna(0,inplace=True)
merge3['percent_chad']=merge3['total_chads']/merge3['total_people']*100
states = merge3['state'].value_counts().index.tolist()
states = sorted(states)
#create a list of positions for the chart
position = []
for i in range(10):
    for j in range(5):
        b = i,j
        position.append(b)
        
#create base of subplot chart.. rows x columbs = graphs
fig, axes = plt.subplots(nrows=10, ncols=5, sharey=True, sharex=False, figsize=(20,20))
fig.subplots_adjust(hspace=.5)

#Fill in base with graphs based off of position
for i in range(50):
    merge3[merge3['state']==states[i]].plot(x='year', y='percent_chad', 
                                                           ax=axes[position[i]], legend=False)    

#Set the formatting elements of the axes for each graph
for i in range(50):
    axes[position[i]].set_title(states[i], size = 12)
    axes[position[i]].tick_params(labelsize=5)
    axes[position[i]].set_xlabel("", size = 5)
