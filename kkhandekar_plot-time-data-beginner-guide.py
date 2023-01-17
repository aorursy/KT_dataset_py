# Generic Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import re



# Plotting Libraries

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.express as pex

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



from bokeh.layouts import gridplot

from bokeh.plotting import figure, output_file, show
#Load Data

url = '../input/handwashing-vs-childbed-fever/monthly_deaths.csv'

data = pd.read_csv(url, header='infer')
#Check for null values

data.isna().sum()
print("Total Records: ",data.shape[0])
#Processing date column



data.date = pd.to_datetime(data.date, format='%Y-%m-%d')

data['year'] = data.date.apply(lambda x: x.year)

data['month'] = data.date.apply(lambda x: x.month)

data['day'] = data.date.apply(lambda x: x.day)



#Dropping the Date Column

data = data.drop(columns='date', axis=1)

#Inspect

data.head()
# HeatMap of Births

br = data.pivot_table(index="year",columns="month",values="births",fill_value=0)

plt.figure(figsize=(15,8))

plt.title('Births - HeatMap [Year vs Month]', fontsize=15)

plt.tick_params(labelsize=10)

ax = sns.heatmap(br, cmap='Greys', linecolor='grey',linewidths=0.1, cbar=False, annot=True, fmt="")
# HeatMap of Deaths

dt = data.pivot_table(index="year",columns="month",values="deaths",fill_value=0)

plt.figure(figsize=(15,10))

plt.title('Deaths - HeatMap', fontsize=15)

plt.tick_params(labelsize=10)

ax = sns.heatmap(dt, cmap='Greys', linecolor='grey',linewidths=0.1, cbar=False, annot=True, fmt="")
cols = ['births','deaths']



layout = dict(title='Births & Deaths from 1841-1849 (with plotly)',)



fig = go.Figure([{

    'x': data.year,

    'y': data[col],

    'name': col

}  for col in cols], layout)



iplot(fig)
def plot_sumry(dataframe,col1,col2):



    sns.set_palette('pastel')

    sns.set_color_codes()

    fig = plt.figure(figsize=(15, 15))

    plt.subplots_adjust(hspace = 0.9)

    

    plt.subplot(221)

    ax1 = sns.lineplot(x="year", y=col1, data=dataframe, color = 'midnightblue')

    plt.title(f'{col1.capitalize()} from 1841-1849(with seaborn)', fontsize=13)

    

    plt.subplot(222)

    ax2 = sns.lineplot(x="year", y=col2, data=dataframe, color = 'midnightblue')

    plt.title(f'{col1.capitalize()} from 1841-1849(with seaborn)', fontsize=13)
plot_sumry (data,'births','deaths')
x = data.pivot_table(index='year',values=['births','deaths'],aggfunc=np.mean)

x.plot(figsize=(15,8))

plt.title("Average Births & Deaths from 1841-1849", fontsize=15)