# Import Libraries

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import matplotlib.animation as animation

from IPython.display import HTML
# read the file

fullData = pd.read_csv("../input/covid19-dataset/full_data.csv")
fullData.head()
df = pd.read_csv("../input/covid19-dataset/full_data.csv",

                 usecols=['date','location','total_cases'])
df.head()
df.drop(df[df['location'] == 'World' ].index, inplace=True)

df.fillna(0)

df
end = df['date'].max()

dff = (df[df['date'].eq(end)]

       .sort_values(by='total_cases', ascending=False).head(10))

dff
array = dff['location'].tolist()

df = df.loc[df['location'].isin(array)]

df
fig, ax = plt.subplots(figsize=(15, 8))

ax.barh(dff['location'], dff['total_cases'])
colors = dict(zip(

    dff['location'].tolist(),

    ['#adb0ff', '#ffb3ff', '#90d595', '#e48381','#9e9395','#f2eea7',

     '#aafbff', '#f7bb5f', '#eafb50', '#f26d85']

))
fig, ax = plt.subplots(figsize=(15, 8))

dff = dff[::-1] 

ax.barh(dff['location'], dff['total_cases'], color=[colors[x] for x in dff['location']])

for i, (total_cases, location) in enumerate(zip(dff['total_cases'], dff['location'])):

    ax.text(total_cases, i,     location,            ha='right') 

    ax.text(total_cases, i,     total_cases,           ha='left')

ax.text(1, 0.4, end, transform=ax.transAxes, size=46, ha='right')
def draw_barchart(date):

        

    dff = (df[df['date'].eq(date)]

           .sort_values(by='total_cases', ascending=False)

           .head(10))

    

    ax.clear()

    dff = dff[::-1]

    ax.barh(dff['location'], dff['total_cases'], color=[colors[x] for x in dff['location']])

    

    dx = dff['total_cases'].max() / 200

    for i, (total_cases, location) in enumerate(zip(dff['total_cases'], dff['location'])):

        ax.text(total_cases-dx, i,     location,           size=14, weight=600, ha='right', va='bottom')

        ax.text(total_cases+dx, i,     f'{total_cases:,.0f}',  size=14, ha='left',  va='center')

        

    ax.text(1, 0.4, date, transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)

    ax.text(0, 1.06, 'Confirmed Cases', transform=ax.transAxes, size=12, color='#777777')

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    ax.xaxis.set_ticks_position('top')

    ax.tick_params(axis='x', colors='#777777', labelsize=12)

    ax.set_yticks([])

    ax.margins(0, 0.01)

    ax.grid(which='major', axis='x', linestyle='-')

    ax.set_axisbelow(True)

    ax.text(0, 1.12, 'Confirmed Coronavirus cases in the world',

            transform=ax.transAxes, size=24, weight=600, ha='left')

    ax.text(1, 0, 'by @minaliu', transform=ax.transAxes, ha='right',

            color='#777777', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))

    plt.box(False)

fig, ax = plt.subplots(figsize=(15, 8))

draw_barchart(df['date'].max())
start = df['date'].min()

fig, ax = plt.subplots(figsize=(15, 8))

draw_barchart(start)
fig, ax = plt.subplots(figsize=(15, 8))

draw_barchart('2020-02-14')
dates = df.date.unique()

fig, ax = plt.subplots(figsize=(15, 8))

animator = animation.FuncAnimation(fig, draw_barchart, frames=dates)

HTML(animator.to_jshtml()) 