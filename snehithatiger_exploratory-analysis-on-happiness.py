# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # plotting

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#loading the .csv data into pandas dataframe 

df1 = pd.read_csv("/kaggle/input/world-happiness/2017.csv")

df2 = pd.read_csv("/kaggle/input/world-happiness/2016.csv")

df3 = pd.read_csv("/kaggle/input/world-happiness/2015.csv")
df1.head()
# columns in the dataset

df1.columns
df1.shape
df1.info()
df2.head()
df2.columns
df2.shape
df2.info()
df3.head()
df3.columns
df3.shape
df3.info()
#plotting the happiness rank based on their region

sns.stripplot(x = 'Region',y = 'Happiness Rank', data = df2)

plt.xticks(rotation =50)
sns.stripplot(x = 'Region',y = 'Happiness Rank', data = df3)

plt.xticks(rotation =50)


import plotly.offline as pyo

import plotly.graph_objs as go

from plotly.offline import iplot



import cufflinks as cf

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 

init_notebook_mode(connected=True)



init_notebook_mode(connected=True)

cf.go_offline()



test_data = [dict(type='choropleth',

                 autocolorscale=False,

                 locations=df1['Country'],

                 locationmode = 'country names',

                 z = df1['Happiness.Score'],

                  marker=dict(

                     line=dict(

                         color='rgb(255,255,255)',

                         width=2

                     )),

                 colorbar=dict(

                     title='Choropleth Map Test')

                 )] 

layout = dict(

            title='Happiness Index 2017',

            geo = dict(showframe = True,                

                projection=dict(type='mollweide'),

                #snowflakes=True,

                lakecolor='rgb(255,255,255)'),

            )



fig = dict(data=test_data, layout=layout)



pyo.iplot(fig,filename='d3-cloropleth-map')
test_data = [dict(type='choropleth',

                 autocolorscale=False,

                 locations=df3['Country'],

                 locationmode = 'country names',

                 z = df3['Happiness Score'],

                  marker=dict(

                     line=dict(

                         color='rgb(300,300,300)',

                         width=2

                     )),

                 colorbar=dict(

                     title='Choropleth Map Test')

                 )] 

layout = dict(

            title='Happiness Index 2015',

            geo = dict(showframe = True,                

                projection=dict(type='robinson'),

                #snowflakes=True,

                lakecolor='rgb(275,275,275)'),

            )



fig = dict(data=test_data, layout=layout)



pyo.iplot(fig,filename='d3-cloropleth-map')
data = dict(type = 'choropleth', 

           locations = df2['Country'],

           locationmode = 'country names',

           z = df2['Happiness Score'], 

           text = df2['Country'],

           colorbar = {'title':'Happiness'})

layout = dict(title = 'Happiness Index 2016', 

             geo = dict(showframe = True, 

                       projection = {'type': 'mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)

df1.rename(columns= {'Economy..GDP.per.Capita.':'Economy'}, inplace = True)

df2.rename(columns = {'Economy (GDP per Capita)':'Economy'}, inplace =True)

df3.rename(columns = {'Economy (GDP per Capita)':'Economy'}, inplace = True)
df1.plot(kind='scatter', x='Economy', y='Happiness.Score',alpha = 0.7,color = 'green',figsize=(15,10),subplots = (3,1,1))

plt.xlabel('Economy')

plt.ylabel('Happiness Score')

plt.title('2017')
df2.plot(kind='scatter', x='Economy', y='Happiness Score',

         alpha = 0.7,

         color = 'orange',

         figsize=(15,10),

         subplots = (4,2,1))



plt.xlabel('Economy')

plt.ylabel('Happiness Score')

plt.title('2016')
df3.plot(kind='scatter', x='Economy', y='Happiness Score',

         alpha = 0.7,

         color = 'brown',

         figsize=(15,10),

         subplots = (4,2,1))



plt.xlabel('Economy')

plt.ylabel('Happiness Score')

plt.title('2017')
plt.figure(figsize = (15,15))

sns.set_style("whitegrid");

sns.pairplot(df1[['Happiness.Rank', 'Happiness.Score',

       'Economy', 'Family',

       'Health..Life.Expectancy.', 'Freedom', 'Generosity',

       'Dystopia.Residual']]);

plt.show()
plt.figure(figsize = (15,15))

sns.set_style("whitegrid");

sns.pairplot(df2[['Happiness Rank', 'Happiness Score',

       'Economy', 'Family','Health (Life Expectancy)',

       'Freedom', 'Generosity', 'Dystopia Residual']]);

plt.show()
plt.figure(figsize = (15,15))

sns.set_style("whitegrid");

sns.pairplot(df3[['Happiness Rank', 'Happiness Score',

       'Economy', 'Family','Health (Life Expectancy)',

       'Freedom', 'Generosity', 'Dystopia Residual']]);

plt.show()
# 2017

counts, bin_edges = np.histogram(df1['Happiness.Rank'], bins=15, 

                                 density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:], cdf)





# 2016

counts, bin_edges = np.histogram(df2['Happiness Rank'], bins=15, 

                                 density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:], cdf)





# 2015

counts, bin_edges = np.histogram(df3['Happiness Rank'], bins=15, 

                                 density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:], cdf)





plt.show();