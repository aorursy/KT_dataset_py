# As first step, I have imported all required libraries.
import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import matplotlib.animation as animation

from IPython.display import HTML

import numpy as np

import os

print(os.listdir("../input"))
df = pd.read_csv('../input/API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_1068882.csv', skiprows=(0,3), header = 1)
df.head()
col_to_drop= df.columns[4:33]  # There are multiple columns with no data which, hence deleted. 
df = df.drop(col_to_drop, axis=1)
df.head()
df.shape
df = df.dropna(axis=1,thresh=3)   # we have consistent data from 1990 onwards. Still there are some countries 

# where data is missing. Setting threshold of 3 null values. FOr all rows with 3 or more null values, row will be deleted
df.shape
df.head()
df['Country Name'].unique()   # Fdinding out list of countries for which data is present
a=df['Country Name'].isin(['India','Nigeria','Bangladesh','Philippines', 'Pakistan','Sri Lanka', 'Vietnam'])
df= df[a]
df.shape
df.head()  # details of only chosen countries
col = df.columns[1:4]  # Column 1 to 4 look rdundant, hence removing
df = df.drop(col, axis=1)
df.head()
df_T= df.transpose(copy=True)
df_T.shape
df_T
df_T.index.name = 'Year'  # Setting year as index
new_header = df_T.iloc[0] #grab the first row for the header
df_T = df_T[1:] #take the data less the header row
df_T.columns = new_header #set the header row as the df header
df_T.index
df_T.columns  # columns in new datafram
df_T.index= pd.to_datetime(df_T.index,format='%Y').year  # Setting index year in to date format
type(df_T.index)  # checking data type
## Draw a simple line plot to see movement of Per Capita GDP ($) 
fig, ax = plt.subplots(figsize=(18, 6))

ax.plot(df_T.index, df_T['Bangladesh'], label= 'Bangladesh')

ax.plot(df_T.index, df_T['Sri Lanka'], label= 'Sri Lanka')

ax.plot(df_T.index, df_T['India'], label= 'India')

ax.plot(df_T.index, df_T['Nigeria'], label= 'Nigeria')

ax.plot(df_T.index, df_T['Vietnam'], label= 'Vietnam')

ax.plot(df_T.index, df_T['Pakistan'], label= 'Pakistan')

ax.set_xlabel('Years')

ax.set_ylabel('GDP Per Capita PPP')

ax.set_title('GDP per capita PPP over the years')

plt.legend(loc="upper left")
s=df_T.loc[1990]  # Slicing a series to plot a horizontal bar chart
fig, ax = plt.subplots(figsize=(4, 2.5), dpi=144)

colors = plt.cm.Dark2(range(6))

y = s.index

width = s.values

ax.barh(y=y, width=width, color=colors);

ax.set_xlabel('GDP PPP per Capita ($)')
# Draw barc chart for 3 years
fig, ax_array = plt.subplots(nrows=1, ncols=3, figsize=(7, 2.5), dpi=144, tight_layout=True)

dates = [1990,1991,1992]

for ax, date in zip(ax_array, dates):

    s = df_T.loc[date].sort_values()

    ax.barh(y=s.index, width=s.values, color=colors)

    ax.set_title(date, fontsize='smaller')

    ax.set_xlabel('GDP PPP Per Capita $')
# Rank individual countries based on PPP that year
df_T.loc[1990].rank()
df_T_rank = df_T.rank(axis=1)

df_T_rank
fig, ax_array = plt.subplots(nrows=1, ncols=3, figsize=(7, 2.5), dpi=144, tight_layout=True)

dates = [1990,1991,1992]

for ax, date in zip(ax_array, dates):

    s = df_T.loc[date]

    y = df_T.loc[date].rank().values

    ax.barh(y=y, width=s.values, color=colors,tick_label=s.index)

    ax.set_title(date, fontsize='smaller')

    ax.set_xlabel('GDP PPP Per Capita $')
labels = df_T.columns
from matplotlib.animation import FuncAnimation



def init():

    ax.clear()

    ax.set_ylim(1, 7.5)



def update(i):

    for bar in ax.containers:

        bar.remove()

    y = df_T_rank.iloc[i]

    width = df_T.iloc[i]

    ax.barh(y=y, width=width, color=colors, tick_label=labels)

    date_str = df_T.index[i]

    ax.set_title(f'PPP GDP - {date_str}', fontsize='smaller')

    ax.set_xlabel('GDP Per Capita (PPP $) over years')

    

fig = plt.Figure(figsize=(8, 4), dpi=130)

ax = fig.add_subplot()

anim = FuncAnimation(fig=fig, func=update, init_func=init, frames=len(df_T), 

                     interval=500, repeat=False)
from IPython.display import HTML

html = anim.to_jshtml()

HTML(html)