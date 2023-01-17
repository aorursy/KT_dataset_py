import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv("/kaggle/input/covid19-in-india/covid_19_india.csv")

df.rename(columns={'State/UnionTerritory':'state','Confirmed':'count'}, inplace = 'True')
df = df[['Date','state','count']]

df.head()
#Importing Libraries



import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import matplotlib.animation as animation

import matplotlib.cm as cm





from IPython.display import HTML

from datetime import timedelta, date

from datetime import datetime as dt
#Function to generate bar plot as of date



def draw_barchart(Date):

    dff = df[df['Date'].eq(Date)].sort_values(by='count', ascending=True)

    total = dff['count'].sum()

    dff = dff.tail(10)

    ax.clear()

    ax.barh(dff['state'], dff['count'], color= 'c')

    dx = dff['count'].max() / 200

    for i, (count, state) in enumerate(zip(dff['count'], dff['state'])):

        ax.text(count-dx, i,     state,           size=14, weight=600, ha='right', va='bottom')

        ax.text(count+dx, i,     f'{count:,.0f}',  size=14, ha='left',  va='center')

    # ... polished styles

    ax.text(1, 0.4, Date, transform=ax.transAxes, color='#777777', size=26, ha='right', weight=800)

    ax.text(0.8, 0.3, 'Total Cases', transform=ax.transAxes, color='#999999', size=26, ha='right', weight=800)

    ax.text(1, 0.3, total, transform=ax.transAxes, color='#999999', size=26, ha='right', weight=800)

    ax.text(0, 1.06, 'Confirmed Cases', transform=ax.transAxes, size=15, color='#777777')

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    ax.xaxis.set_ticks_position('top')

    ax.tick_params(axis='x', colors='#777777', labelsize=12)

    ax.set_yticks([])

    ax.margins(0, 0.01)

    ax.grid(which='major', axis='x', linestyle='-')

    ax.set_axisbelow(True)

    ax.text(0, 1.10, 'Statewise Covid-19 Trend in India - Till Today',

            transform=ax.transAxes, size=24, weight=600, ha='left')

    ax.text(1, 0, 'by @KrishnaKashid', transform=ax.transAxes, ha='right',

            color='#777777', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))

    plt.box(False)
#Date range generation for animation

dtm = lambda x: dt.strptime(str(x), "%d/%m/%y")

df["Date"] = df["Date"].apply(dtm)

date_min = df['Date'].min()

date_max = df['Date'].max()



def daterange(date1, date2):

    for n in range(int ((date2 - date1).days)+1):

        yield date1 + timedelta(n)
#Animation generation

fig, ax = plt.subplots(figsize=(15, 8))

animator = animation.FuncAnimation(fig, draw_barchart,interval = 700, repeat = False, frames=daterange(date_min,date_max))

# HTML(animator.to_jshtml()) 

# animator.to_html5_video()

animator.save('covid19_india.gif', writer = 'imagemagick')