# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load







# Disable warnings in Anaconda

import warnings

warnings.filterwarnings('ignore')



# Matplotlib forms basis for visualization in Python

import matplotlib.pyplot as plt



# We will use the Seaborn library

import seaborn as sns

sns.set()



# Graphics in retina format are more sharp and legible

%config InlineBackend.figure_format = 'retina' 



# Increase the default plot size and set the color scheme

plt.rcParams['figure.figsize'] = 8, 5

plt.rcParams['image.cmap'] = 'viridis'

import pandas as pd





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/video-game-sales-with-ratings/Video_Games_Sales_as_at_22_Dec_2016.csv').dropna()

print(df.shape)
df.head()
df.info()
df['User_Score'] = df['User_Score'].astype('float64')

df['Year_of_Release'] = df['Year_of_Release'].astype('int64')

df['User_Count'] = df['User_Count'].astype('int64')

df['Critic_Count'] = df['Critic_Count'].astype('int64')
df.info()
useful_cols = ['Name', 'Platform', 'Year_of_Release', 'Genre', 

               'Global_Sales', 'Critic_Score', 'Critic_Count',

               'User_Score', 'User_Count', 'Rating'

              ]

df[useful_cols].head()
df[[i for i in df.columns if 'Sales' in i] + ['Year_of_Release']].groupby('Year_of_Release').sum().plot()
df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales' ] + ['Year_of_Release']].groupby('Year_of_Release').sum().plot()
df[[i for i in df.columns if 'Sales' in i] + ['Year_of_Release']].groupby('Year_of_Release').sum().plot(kind='area', rot=45)
%config InlineBackend.figure_format = 'png'

sns.pairplot(df[['Global_Sales', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']])
%config InlineBackend.figure_format = 'retina'

sns.distplot(df['Critic_Score']);
sns.jointplot('Critic_Score', 'User_Score', data=df, kind="kde", space=0, color="red")
df.Platform.unique()
top_plat = df['Platform'].value_counts().sort_values(ascending=False).index.values

plt.figure(figsize=(11,9))

sns.boxplot(y = "Platform", x = "Critic_Score", data=df[df['Platform'].isin(top_plat)], orient="h")
platform_genre_sales = df.pivot_table(

                        index='Platform', 

                        columns='Genre', 

                        values='Global_Sales', 

                        aggfunc=sum).fillna(0).applymap(float)

plt.figure(figsize=(11,8))

sns.heatmap(platform_genre_sales, annot=True, fmt=".1f", linewidths=.5)
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly

import plotly.graph_objs as go



init_notebook_mode(connected=True)
years_df = df.groupby('Year_of_Release')[['Global_Sales']].sum().join(

    df.groupby('Year_of_Release')[['Name']].count())

years_df.columns = ['Global_Sales', 'Number_of_Games']
# Create a line (trace) for the global sales

trace0 = go.Scatter(

    x=years_df.index,

    y=years_df['Global_Sales'],

    name='Global Sales'

)



# Create a line (trace) for the number of games released

trace1 = go.Scatter(

    x=years_df.index,

    y=years_df['Number_of_Games'],

    name='Number of games released'

)



# Define the data array

data = [trace0, trace1]



# Set the title

layout = {'title': 'Statistics for video games'}



# Create a Figure and plot it

fig = go.Figure(data=data, layout=layout)

iplot(fig, show_link=False)
plotly.offline.plot(fig, filename='years_stats.html', show_link=False);
# Do calculations and prepare the dataset

platforms_df = df.groupby('Platform')[['Global_Sales']].sum().join(

    df.groupby('Platform')[['Name']].count()

)

platforms_df.columns = ['Global_Sales', 'Number_of_Games']

platforms_df.sort_values('Global_Sales', ascending=False, inplace=True)




# Create a bar for the global sales

trace0 = go.Bar(

    x=platforms_df.index,

    y=platforms_df['Global_Sales'],

    name='Global Sales'

)



# Create a bar for the number of games released

trace1 = go.Bar(

    x=platforms_df.index,

    y=platforms_df['Number_of_Games'],

    name='Number of games released'

)



# Get together the data and style objects

data = [trace0, trace1]

layout = {'title': 'Market share by gaming platform'}



# Create a `Figure` and plot it

fig = go.Figure(data=data, layout=layout)

iplot(fig, show_link=False)



data = []



# Create a box trace for each genre in our dataset

for genre in df.Genre.unique():

    data.append(

        go.Box(y=df[df.Genre == genre].Critic_Score, name=genre)

    )

    

# Visualize

iplot(data, show_link=False)