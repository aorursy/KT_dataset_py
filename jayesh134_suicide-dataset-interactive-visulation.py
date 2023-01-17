# Data Cleaning & Manipulation Libraries

import numpy as np

import pandas as pd
# Data Visulation Libraries

import matplotlib.pyplot as plt

import plotly as py

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()
# Reading master.csv file using Pandas

Suicide = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')
# Suicide Dataset info.

Suicide.info()     # Eagle Eye View
# Suicide Dateset 

Suicide.head()   # In-depth View
# Data Set    

total_ppl = Suicide.groupby(by='country').population.sum().sort_values(ascending=False)

total_ppl     # 'United States with highest population' & 'Dominica with lowest population'
# Data Visulation - Bar Plot

total_ppl.iplot(kind='bar',title='Population across the Globe',xaxis_title='Countries',yaxis_title='Population in Billion')
# Geographical Visulation - choropleth map



# Data Object

data = dict(type='choropleth',

           locations = total_ppl.index,

           locationmode = 'country names',

           z = total_ppl[:],

           colorscale = 'oranges',

           colorbar = {'title':'Population in Billion'},

           text = total_ppl.index)

# Layout Object

layout = dict(geo = dict(projection={'type':'orthographic'},showframe=False),title='Population across the Globe')

# Plotting

choromap = go.Figure(data = [data],layout = layout)

iplot(choromap,validate=False)
# Minor Change in age column     # for personal preference only

Suicide['age'].replace(to_replace='5-14 years',value='05-14 years',inplace=True)
# Data Set

Suicide.groupby('age').population.sum()
# Data Visulation - Bar Plot

Suicide.groupby('age').population.sum().iplot(kind='bar',title='World\'s Population as per Age Category',

                                              xTitle='Age Category',yTitle='Population in Billions')
# Data Visulation - Pie Chart

go.Figure(data=go.Pie(labels=Suicide.groupby('age').population.sum().index, values=Suicide.groupby('age').population.sum()[:],

                      title=' World\'s Population as per Age Category'))
# Data Set

total_suicide = Suicide.groupby('country').suicides_no.sum().sort_values(ascending=False)

total_suicide    # 'Russia with highest suicides' & 'Dominica with lowest suicides'
# Data Visulation - Bar Plot

total_suicide.iplot(kind='bar',color='red',title='Suicides per Country',xTitle='Countries',yTitle='Suicide Counts')
# Data Visulation - Line Plot

total_suicide.iplot(kind='line',color='red',title='Suicides per Country',xTitle='Countries',yTitle='Suicide Counts')
# Geographical Visulation - choropleth map



# Data Object

data = dict(type='choropleth',

           locations = total_suicide.index,

           locationmode = 'country names',

           z = total_suicide[:],

           colorscale = 'viridis',

           colorbar = {'title':'Suicides in Million'},

           text = total_suicide.index)

# Layout Object

layout = dict(geo = dict(projection={'type':'orthographic'},showframe=False),title='Suicide around the World')

# Plotting

choromap = go.Figure(data = [data],layout = layout)

iplot(choromap,validate=False)
# dataset

Suicide.groupby('year').suicides_no.sum()
# Maximum no. of suicide & year of occurence

print('Maximum Suicide Number :',Suicide.groupby('year').suicides_no.sum().max())

print('Maximum Suicide Year :',Suicide.groupby('year').suicides_no.sum().idxmax())



# Minimum no. of suicide & year of occurence

print('\nMinimum Suicide Number :',Suicide.groupby('year').suicides_no.sum().min())

print('Minimum Suicide Year :',Suicide.groupby('year').suicides_no.sum().idxmin())
# Data Visulation - Bar Plot

Suicide.groupby('year').suicides_no.sum().iplot(kind='bar',title='Suicides per Year 1985 to 2016',

                                                xTitle='Year',yTitle='Suicide Counts')
# Data Visulation - Line Graph

Suicide.groupby('year').suicides_no.sum().iplot(kind='line',title='Suicides per Year 1985 to 2016',

                                                xTitle='Year',yTitle='Suicide Counts')
# Dataset

df = Suicide.groupby(['generation','sex']).suicides_no.sum().unstack()

df
# Data Visulation - Bar Plot

df.iplot(kind='bar',title='Suicides per Generation',xTitle='Generation',yTitle='Total Suicides')
# Data Set

Suicide.groupby(['year','sex']).mean()['suicides/100k pop'].unstack()
# Data Visulation - Line Plot

Suicide.groupby(['year','sex']).mean()['suicides/100k pop'].unstack().iplot(kind='line',

                                title='Suicide Rate per 100K People',xTitle='Year',yTitle='Suicide Rate')
# Data Set

Suicide.groupby('age').mean()['suicides/100k pop']
# Data Visulation - Pie Chart

go.Figure(data=[go.Pie(labels=Suicide['age'],values=Suicide['suicides/100k pop'],

                       title='Suicide Rate per Age Category')])