import numpy as np 

import pandas as pd 

import plotly.graph_objs as go

import plotly.plotly as py

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import seaborn as sns

import matplotlib.pyplot as plt



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df_15=pd.read_csv('../input/2015.csv')

df_16=pd.read_csv('../input/2016.csv')

df_15.head(1)
df_16.head(1)
print('The number of rows with Missing Values in 2015.csv are: ')

df_15.isnull().any(axis=1).sum()

print('The number of rows with Missing Values in 2016.csv are: ')

df_16.isnull().any(axis=1).sum()
data = dict(type = 'choropleth', 

           locations = df_15['Country'],

           locationmode = 'country names',

           z = df_15['Happiness Rank'], 

           text = df_15['Country'],

           colorscale = 'Viridis', reversescale = False)

layout = dict(title = 'Global Happiness 2015', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap = go.Figure(data = [data], layout=layout)

iplot(choromap)
data = dict(type = 'choropleth', 

           locations = df_16['Country'],

           locationmode = 'country names',

           z = df_16['Happiness Rank'], 

           text = df_16['Country'],

           colorscale = 'Viridis', reversescale = False)

layout = dict(title = 'Global Happiness 2016', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap = go.Figure(data = [data], layout=layout)

iplot(choromap)
print('The Top 20 Happy Countries of 2015:')

top20_df_15=df_15['Country'].head(20)

top20_df_15
print('The Top 20 Happy Countries of 2016:')

top20_df_16=df_16['Country'].head(20)

top20_df_16
print('The countries who have managed to be in the Top 20 for all 3 years: ', np.intersect1d(top20_df_15, top20_df_16)

)
def plot_compare(parameter):    

    fig = plt.figure(figsize=(20, 15))



    ax1 = fig.add_subplot(221)

    ax1.plot(df_15[parameter], 'b', label='2015')

    ax1.plot(df_16[parameter], 'g', label='2016')

    ax1.set_title(parameter+': 2015 (Blue) & 2016 (Green)', fontsize=22)

    plt.xlabel('Rank of Country', fontsize=18)

    plt.ylabel(parameter, fontsize=18)



    ax2 = fig.add_subplot(222)

    ax2.plot(df_15[parameter].rolling(15).sum())

    ax2.plot(df_16[parameter].rolling(15).sum())

    ax2.set_title('Rolling Average of '+parameter+': 2015 (Blue) & 2016 (Green)', fontsize=22)

    plt.xlabel('Rank of Country', fontsize=18)

    plt.ylabel('Rolling Average of '+parameter, fontsize=18)



    plt.tight_layout()

    fig = plt.gcf()

   



def find_avg(parameter):

    print('Average '+parameter+': 2015',df_15[parameter].mean())

    print()

    print('Average '+parameter+': 2016',df_16[parameter].mean())
fig = plt.figure()

plt.plot(df_15['Happiness Score'], 'b', label='2015')

plt.plot(df_16['Happiness Score'], 'g', label='2016')

fig.suptitle('Happiness Score of 2015 (Blue) & 2016 (Green)', fontsize=18)

plt.xlabel('Rank of Country', fontsize=16)

plt.ylabel('Happiness Score', fontsize=16)
find_avg('Happiness Score')
plot_compare('Economy (GDP per Capita)')
find_avg('Economy (GDP per Capita)')
plot_compare('Family')
find_avg('Family')

plot_compare('Health (Life Expectancy)')
find_avg('Health (Life Expectancy)')
plot_compare('Freedom')
find_avg('Freedom')
plot_compare('Trust (Government Corruption)')
find_avg('Trust (Government Corruption)')
plot_compare('Generosity')
find_avg('Generosity')
plot_compare('Dystopia Residual')
find_avg('Dystopia Residual')