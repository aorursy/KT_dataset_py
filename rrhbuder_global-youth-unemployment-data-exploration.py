# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode()





import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/API_ILO_country_YU.csv')
df.head()
lar = df.nlargest(10, '2014')

c1 = lar.nlargest(1,'2014')

lar
sml = df.nsmallest(10,'2014')

c2 = sml.nsmallest(1,'2014')

sml 
a = df['2010']

b = df['2014']

df['Percent Change'] = (b - a) * (100/a)
chnge = df.nlargest(10,'Percent Change')

chnge2 = chnge.nlargest(6,'Percent Change')

c3 = chnge.nlargest(1,'Percent Change') #getting the largest for our later plot

c4 = df.nsmallest(1,'Percent Change')

chnge
df2 = pd.concat([c1,c4, c3])

df2
c5 = df[(df['Country Name'] == 'United States') | (df['Country Name'] == 'Egypt') | (df['Country Name'] == 'Germany') ]

df2 = pd.concat([df2,c5])

df2 = df2[['Country Name','2010','2011','2012','2013','2014']].T #transpose the dataframe to prepare for plotting

df2  
df2 = df2.tail(5)

ax = df2.plot(figsize = (8,6),kind='line',subplots=False, title='Global Youth Unemployment Trends', grid=True,colormap='tab20c',linewidth=6)

ax.set_xlabel("Year")

ax.set_ylabel("Percent Unemployed")

ax.set_ylim(0,100)

ax.legend(['Spain', 'Ghana', 'Cyprus', 'Germany','United States'], loc='upper left')

ax.grid(color='lightgrey', linestyle='-', linewidth=1.5)

ax.set_facecolor('white')

ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.spines['bottom'].set_visible(True)

ax.xaxis.grid(False)

dfall = df[['Country Name','2010','2011','2012','2013','2014']]

dfall = dfall.T

dfall2 = dfall.tail(5)
chnge2 = chnge2[['Country Name','2010','2011','2012','2013','2014']]

#pl.legend(['Spain', 'Ghana', 'Cyprus', 'Germany','United States'], loc='upper left')
chnge2 = chnge2.T
chnge3 = chnge2.tail(5)
chnge2
pl = chnge3.plot(figsize = (8,6),kind='line',stacked=False,subplots=False, title='Countries With Largest Increase in Youth Unemployment', grid=True, linewidth=5,colormap='tab20c')

pl.set_xlabel("Year")

pl.set_ylabel("Percent Unemployed")

pl.set_ylim(0,100)

pl.legend(['Cyprus', 'Kuwait', 'Greece', 'Portugal','Egypt','Italy'], loc='upper left')

pl.grid(color='lightgrey', linestyle='-', linewidth=1.5)

pl.set_facecolor('white')

pl.spines['right'].set_visible(False)

pl.spines['top'].set_visible(False)

pl.spines['left'].set_visible(False)

pl.spines['bottom'].set_visible(True)

pl.xaxis.grid(False)
dfg20 = df[['Country Name','2010','2011','2012','2013','2014']]
dfg20.head()
a = dfg20['2010']

b = dfg20['2011']

dfg20['2011c'] = (b - a) * (100/a)

a = dfg20['2011']

b = dfg20['2012']

dfg20['2012c'] = (b - a) * (100/a)

a = dfg20['2012']

b = dfg20['2013']

dfg20['2013c'] = (b - a) * (100/a)

a = dfg20['2013']

b = dfg20['2014']

dfg20['2014c'] = (b - a) * (100/a)

dfg20['2010c'] = dfg20['2010'] * 0

dfg20.head()
dfg20 = dfg20[['Country Name','2010c','2011c','2012c','2013c','2014c']]

dfg20 = dfg20[(dfg20['Country Name'] == 'United States') | (dfg20['Country Name'] == 'Mexico') | (dfg20['Country Name'] == 'Canada') | (dfg20['Country Name'] == 'Venezuela') | (dfg20['Country Name'] == 'Brazil')]
dfg20 = dfg20.T
dfg20.head()
dfg20 = dfg20.tail(4)

pl2 = dfg20.plot(figsize = (8,6),kind='line',stacked=False,subplots=False, title='North American Countries Percent Change in Youth Unemployment', grid=True, linewidth=4.5,colormap='tab20c')

pl2.set_xlabel("Year")

pl2.set_ylabel("Percent Change Unemployed")

pl2.set_ylim(-10,10)

pl2.legend(['Brazil', 'Canada', 'Mexico', 'United States'], loc='upper left')

pl2.grid(color='lightgrey', linestyle='-', linewidth=1.5)

pl2.set_facecolor('white')

pl2.spines['right'].set_visible(False)

pl2.spines['top'].set_visible(False)

pl2.spines['left'].set_visible(False)

pl2.spines['bottom'].set_visible(True)

pl2.axhline(y=0.0, color='black', linestyle='-')



pl2.xaxis.grid(False)
dfglobal = df[['Country Name','Country Code','2014']]
scl =  [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],

            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(120, 150, 250)"]]

scl = scl.reverse()

data = [ dict(

        type='choropleth',

        colorscale = scl,

        autocolorscale = False,

        locations = dfglobal['Country Code'],

        z = dfglobal['2014'],

        text = dfglobal['Country Name'],

        marker = dict(

            line = dict (

                color = 'rgb(255,255,255)',

                width = 2

            ) ),

        colorbar = dict(

            title = "Percentage")

        ) ]



layout = dict(

        title = 'Global Youth Unemployment Trends',

        geo = dict(

            scope='world',

            projection=dict( type='Mercator' ),

            showlakes = True,

            lakecolor = 'rgb(255, 255, 255)'),

             )

    

fig = dict( data=data, layout=layout )

iplot( fig, filename='d3-cloropleth-map' )