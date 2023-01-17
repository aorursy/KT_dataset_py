# if you dont have the libriaries show above you can upload it by using 

# pip install seaborn

# pip install plotly

# pip install ggplot

# pip install matplotlib
# pip install future

from __future__ import (absolute_import, division, print_function, 

                        unicode_literals)

# turn off warnings

import warnings

warnings.simplefilter('ignore')



# inline visualizations 

%pylab inline

# turn the visualization into SVG format

%config InlineBackend.figure_format = 'svg'

# and lets resize default size of graphs

from pylab import rcParams

rcParams['figure.figsize'] = 6,5



# and last point lets import libriaries for working with data and data manipulation

import pandas as pd

import numpy as np

import seaborn as sns
# I forgot the path on Kagle so lets find out where we are now:

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/video_games_sales.csv')

print(df.shape)
df.info()
# As you see there are not all data for "Critic_Score", "Critic_count", "Developer" and "Rating" 

# Usually I delete it (but not always)

df = df.dropna()

print(df.shape)

df.info()
# As mentiod from df.info() not all features have an appropriate data-type. Change it to the another data-type

df['User_Score'] = df.User_Score.astype('float64')

df['Year_of_Release'] = df.Year_of_Release.astype('int64')

df['User_Count'] = df.User_Count.astype('int64')

df['Critic_Count'] = df.Critic_Count.astype('int64')
df.head()
# I think I will analyze not all features. In this dataframe we have 6825 rows and 16 columns(features)

# Lets leave features that are the most meaningfull

meaningfull_cols = ['Name', 'Platform', 'Year_of_Release', 'Genre', 'Global_Sales',

                    'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count', 'Rating']



df[meaningfull_cols].head(5)
[x for x in df.columns if 'Sales' in x]
# And I want to mention that NA_Sales stands for North_American_Sales and not for NA as you could think 

df1 = df[[x for x in df.columns if 'Sales' in x] + ['Year_of_Release']].groupby('Year_of_Release').sum()

df1.head()
df1.plot();
df1.plot(kind = 'bar', rot = 45, figsize = (10, 5));
df1[list(filter(lambda x: x != 'Global_Sales', df1.columns))].plot(kind = 'bar', rot = 45, stacked = True, figsize = (10, 5));
# stacked parameter is for visibility

df1[list(filter(lambda x: x != 'Global_Sales', df1.columns))].plot(kind = 'area', rot = 45, stacked = False, figsize = (10, 5));
df.Critic_Score.hist(figsize = (10, 5));
ax = df.Critic_Score.hist(figsize = (10, 5));

ax.set_title('Critic Score distribution');

ax.set_xlabel('Critic Score');

ax.set_ylabel('Games');
# You can choose the number of bins for the distribution by calling it bins = #

ax = df.Critic_Score.hist(figsize = (10, 5), bins = 25);

ax.set_title('Critic Score distribution');

ax.set_xlabel('Critic Score');

ax.set_ylabel('Games');
import seaborn as sns

%config InlineBackend.figure_format = 'png'

sns_plot = sns.pairplot(df[['Global_Sales', 'User_Score', 'Critic_Score']]);

# you can save the chart by

# sns_plot.savefig('#name_of_chart.png')
# Also seaborn can visualize distribution of quantitative value in different ways

# Joint_plot - is a hybrid of scatter_plot and histogram. 

# Lets see how it works on two values Critic_Score and User_Score

sns.jointplot(x = 'Critic_Score', y = 'User_Score', data = df, kind = 'scatter');
sns.jointplot( x = 'Critic_Score', y = 'User_Score', data = df, kind = 'reg');
platform_gender_sales = df.pivot_table(index = 'Platform', columns = 'Genre', values = 'Global_Sales', aggfunc = sum).fillna(0).applymap(float)

platform_gender_sales.head()
sns.heatmap(platform_gender_sales, annot = True, fmt = '.0f', linewidths = 0.7);
from plotly.offline import init_notebook_mode, iplot

import plotly

import plotly.graph_objs as go



init_notebook_mode(connected = True)
global_sales_df = df.groupby('Year_of_Release')[['Global_Sales']].sum()

global_sales_df.head(5)
released_years_df = df.groupby('Year_of_Release')[['Name']].count()

released_years_df.head(5)
years_df = global_sales_df.join(released_years_df)

years_df.columns = ['Global_Sales', 'Number_of_Games']

years_df.head()
# declare a trace which is an array and specify the design

# go.Scatter - specifies type of chart

# trace0 simply just declaring a firts line on the graph

trace0 = go.Scatter(

    # what should be on x-axis

    x = years_df.index,

    # what should be on y-axis

    y = years_df.Global_Sales,

    # Title for the Chart

    name = 'Global Sales'

)

# declare a second Scatter (second line on the graph)

trace1 = go.Scatter(

    # Specify x-axis

    x = years_df.index,

    # and y-axis

    y = years_df.Number_of_Games,

    # Title of the chart

    name = 'Number of games released'

)



# collect all the traces (arrays) in separate dataframe

data = [trace0, trace1]

# choosing a single title

layout = {'title': 'Statistics of video games'}

# Plotting the figure

fig = go.Figure(data = data, layout = layout)



iplot(fig, show_link = False)
# if you want to save your graph just use the next code

# plotly.offline.plot(fig, filename = 'years_stats_sales.#specify_format_after_dot', show_link = False);
platform_sales_global_df = df.groupby('Platform')[['Global_Sales']].sum()

released_df = df.groupby('Platform')[['Name']].count()

platforms_df = platform_sales_global_df.join(released_df)
platforms_df.columns = ['Global_Sales', 'Number_of_Games']

platforms_df.sort_values('Global_Sales', inplace = True)

platforms_df = platforms_df.apply(lambda x: 100 * x / platforms_df.sum(), axis = 1)

platforms_df.head()
# Finally lets plot the data on chart



# again create a traces where specify the chart type and its axis

trace0 = go.Bar(

    x = platforms_df.index,

    y = platforms_df.Global_Sales,

    name = 'Global Sales',

    orientation = 'v'

)



trace1 = go.Bar(

    x = platforms_df.index,

    y = platforms_df.Number_of_Games,

    name = 'Number of games released',

    orientation = 'v'

)



data = [trace0, trace1]

layout = {'title': 'Platforms share'}



fig = go.Figure(data = data, layout = layout)



iplot(fig, show_link = False)
# We can interactively represent the dependency between mean User_Score and Critic_Score and its influence on Global_Sales

# To do it we need to join two tables with scores and sales

scores_genres = df.groupby('Genre')[['Critic_Score', 'User_Score']].mean()

sales_genres = df.groupby('Genre')[['Global_Sales']].sum()

genres_sales = scores_genres.join(sales_genres)



genres_sales.head()
# So finally plot the data on char. I choose a scatter plot because it will show dependencies

trace0 = go.Scatter(

            x = genres_sales.Critic_Score,

            y = genres_sales.User_Score,

            mode = 'markers+text',

            text = genres_sales.index)



data = [trace0]

layout = {'title': 'Influence of User and Critic Scores on Sales'}



fig = go.Figure(data = data, layout = layout)

iplot(fig, show_link = False)
# From this scatter plot we can modify it and create a bubble chart which will show the amount of sales that was calculated before

genres_sales.index
trace0 = go.Scatter(

    x = genres_sales.Critic_Score,

    y=genres_sales.User_Score,

    mode = 'markers+text',

    text = genres_sales.index,

    marker = dict(

        size = 1/10*genres_sales.Global_Sales,

        color = [

            'aqua', 'azure', 'beige', 'lightgreen',

            'lavender', 'lightblue', 'pink', 'salmon',

            'wheat', 'ivory', 'silver'

        ]

    )

)



data = [trace0]

layout = {

    'title': 'Influence of User and Critic Scores on Sales',

    'xaxis': {'title': 'Critic Score'},

    'yaxis': {'title': 'User Score'}

}



fig = go.Figure(data=data, layout=layout)



iplot(fig, show_link=False)