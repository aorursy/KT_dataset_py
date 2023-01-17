import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# we will be using only the following columns 

# Date ,State/Union Territory & Confirmed Count



use_columns = ['Date' , 'State/UnionTerritory', 'Confirmed']

import pandas as pd

df = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv' , usecols = use_columns)

# Let us convert the date from String into YYYY-MM-DD format using pandas to_datetime function

df['Date'] =pd.to_datetime(df['Date'] , format='%d/%m/%y')

df.head(5)

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import matplotlib.animation as animation

from IPython.display import HTML

current_date = '2020-04-30'

df_10 = (df[df['Date']==current_date]

       .sort_values(by='Confirmed', ascending=False)

       .head(10))

print(df_10)



fig, ax = plt.subplots(figsize=(16, 8))

ax.barh(df_10['State/UnionTerritory'], df_10['Confirmed'])
fig, ax = plt.subplots(figsize=(16, 8))

df_10= df_10[::-1]   # States with highest count will be pushed down - so that it shows up in the graph



ax.barh(df_10['State/UnionTerritory'], df_10['Confirmed'] , color ='plum')



for i, (count, region) in enumerate(zip(df_10['Confirmed'], df_10['State/UnionTerritory'])):

    ax.text(count, i,     region,      size=12,      ha='right' )  # Maharashtra

    ax.text(count, i,     count,       size=12  ,  ha='left')   # 9915



ax.text(1, 0.4, current_date, transform=ax.transAxes, size=40, ha='right') # add the date 
fig, ax = plt.subplots(figsize=(16, 8))

def draw_horizontal_bar (date):

    ax.clear()  # this is important - as each function execution should run on fresh axis 

    df_10 = (df[df['Date']==date].sort_values(by='Confirmed', ascending=False).head(10))

    df_10= df_10[::-1] # flip the values

    ax.barh(df_10['State/UnionTerritory'], df_10['Confirmed'] , color ='plum')

    filler = df_10['Confirmed'].max() /100 #  to add space between the States/UnionTerritory Name and the count

    

    for i, (count, region) in enumerate(zip(df_10['Confirmed'], df_10['State/UnionTerritory'])):

        ax.text(count -filler, i,     region,      size=12,  weight =400,    ha='right' )  # Maharashtra

        ax.text(count +filler , i,     count,       size=12  ,  ha='left')   # 9915



    ax.text(1, 0.4, date, transform=ax.transAxes, size=40, ha='right') # add the date 

    ax.text(0, 1.08, 'COVID confirmed cases', transform=ax.transAxes, size=18, color='black')

    ax.xaxis.set_ticks_position('top')

    ax.grid(which='major', axis='x', linestyle='-')

    ax.set_yticks([])

    ax.text(0, 1.20, 'States with highest COVID 19 Confirmed Cases in INDIA',

            transform=ax.transAxes, size=24,  ha='left')

    plt.box(False) # remove the box 

    



draw_horizontal_bar('2020-04-30')
# Lets us get all the dates 



all_dates = df[df['Date'] >= '2020-04-01']['Date'].astype(str).to_list() # this is contain duplicates

dates = list(set(all_dates)) # we get uniques dates - but they are not sorted

dates = sorted(dates) # all dates arranged in order 



import matplotlib.animation as animation # import animation 

from IPython.display import HTML

fig, ax = plt.subplots(figsize=(16, 8))

anim = animation.FuncAnimation(fig, draw_horizontal_bar, frames=dates) # this will call our 

# function for all the dates one by one

HTML(anim.to_jshtml()) 





f = r"covid_india.gif" 

writergif = animation.PillowWriter(fps=4) 

anim.save(f, writer=writergif)