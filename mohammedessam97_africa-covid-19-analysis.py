import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns 

import datetime 

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import matplotlib.animation as animation

from IPython.display import HTML
path = '../input/africa-covid19-daily-cases/Africa Covid-19 Daily Cases .csv'

data = pd.read_csv(path)
data.head()
for col in data.columns: 

    print(col)

    print(data[col].isnull().sum())
data.fillna(0,inplace=True)
data['Daily_Cases']=data['Daily_Cases'].apply(lambda x : 0 if x<0 else x)
data.dtypes
data['Daily_Cases']=data['Daily_Cases'].astype('int64')

data['Daily_Deaths']=data['Daily_Deaths'].astype('int64')

data['Total_Deaths']=data['Total_Deaths'].astype('int64')
data.describe()
# Group data by Country  

total_cases = data.groupby('Country')['Daily_Cases'].sum()
total_cases = total_cases.reset_index()

total_cases.head()
# Select Top 10 countries in Daily Cases sum 

total_cases = total_cases.nlargest(10, ['Daily_Cases']) 
plt.figure(figsize=(15,10))

sns.barplot(x='Country',y='Daily_Cases',data=total_cases)
total_deaths = data.groupby('Country')['Daily_Deaths'].sum()
total_deaths = total_deaths.reset_index()

total_deaths.head()
# Select Top 10 countries in Daily Cases sum 

total_deaths = total_deaths.nlargest(10, ['Daily_Deaths']) 
plt.figure(figsize=(15,10))

sns.barplot(x='Country',y='Daily_Deaths',data=total_deaths)
colors = dict(zip(data['Country'].unique(),sns.color_palette(None, data['Country'].nunique())

))

data['group']=data['Country']

group_lk = data.set_index('Country')['group'].to_dict()
fig, ax = plt.subplots(figsize=(15, 8))

def draw_barchart(Date):

    dff = (data[data['Date'].eq(Date)].sort_values(by='Daily_Cases', ascending=True)).tail(10)

    ax.clear()

    ax.barh(dff['Country'], dff['Daily_Cases'],color=[colors[group_lk[x]] for x in dff['Country']])

    dx = dff['Daily_Cases'].max() / 200

    for i, (value, name) in enumerate(zip(dff['Daily_Cases'], dff['Country'])):

        ax.text(value-dx, i,     name,           size=14, weight=600, ha='right', va='bottom')

        ax.text(value+dx, i,     f'{value:,.0f}',  size=14, ha='left',  va='center')



    ax.text(1, 0.4, Date, transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)

    ax.text(0, 1.06, 'Covid-19 Cases', transform=ax.transAxes, size=12, color='#777777')

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    ax.xaxis.set_ticks_position('top')

    ax.tick_params(axis='x', colors='#777777', labelsize=12)

    ax.set_yticks([])

    ax.margins(0, 0.01)

    ax.grid(which='major', axis='x', linestyle='-')

    ax.set_axisbelow(True)

    ax.text(0, 1.12, 'The Countries with most Daily Cases in Africa from Feb 15 to Sep 24',

            transform=ax.transAxes, size=24, weight=600, ha='left')

    plt.box(False)

    

draw_barchart('Mar 15')
import matplotlib.animation as animation

from IPython.display import HTML

fig, ax = plt.subplots(figsize=(15, 8))

animator = animation.FuncAnimation(fig, draw_barchart, frames=data['Date'].unique())

HTML(animator.to_jshtml()) 