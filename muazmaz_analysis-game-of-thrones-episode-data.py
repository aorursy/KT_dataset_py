from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # for making plots with seaborn

color = sns.color_palette()

sns.set(rc={'figure.figsize':(25,15)})
print(os.listdir('../input/game-of-thrones-episode-data-full'))

df_ep = pd.read_csv('../input/game-of-thrones-episode-data-full/got_csv_full.csv')

df_ep.head(10)

print('Number of episodes in the dataset : ' , len(df_ep))
df_ep_clean = pd.read_csv('../input/game-of-thrones-episode-data-cleaned/got_csv_full_clean.csv')

df_ep_clean.head(10)

print('Number of episodes in the dataset : ' , len(df_ep))
#plotPerColumnDistribution(df_ep, 10, 5)

print(df_ep.dtypes)
%matplotlib inline

from matplotlib import pyplot as plt

plt.style.use('ggplot')



color = sns.color_palette()

sns.set(rc={'figure.figsize':(25,15)})



import plotly

# connected=True means it will download the latest version of plotly javascript library.

plotly.offline.init_notebook_mode(connected=True)

import plotly.graph_objs as go



import plotly.figure_factory as ff

import cufflinks as cf





import warnings

warnings.filterwarnings('ignore')
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()

plotPerColumnDistribution(df_ep, 10, 5)
number_of_deaths_in_category = df_ep['Notable_Death_Count'].value_counts().sort_values(ascending=True)



data = [go.Pie(

        labels = number_of_deaths_in_category.index,

        values = number_of_deaths_in_category.values,

        hoverinfo = 'label+value'

    

)]



plotly.offline.iplot(data, filename='Notable_Death_Count')
data = [go.Histogram(

        x = df_ep.Imdb_Rating,

        xbins = {'start': 1, 'size':0.5, 'end' :10}

)]



print('Average episode rating = ', np.mean(df_ep['Imdb_Rating']))

plotly.offline.iplot(data, filename='overall_rating_distribution')
sns.set_style("darkgrid")

ax = sns.jointplot(df_ep['Season'], df_ep['Imdb_Rating'])
sns.set_style("darkgrid")

ax = sns.jointplot(df_ep['Notable_Death_Count'], df_ep['Imdb_Rating'])
sns.set_style("darkgrid")

ax = sns.jointplot(df_ep['US_viewers_million'], df_ep['Imdb_Rating'])
fig, ax = plt.subplots()

fig.set_size_inches(10, 5)

p = sns.stripplot(x="Imdb_Rating", y="US_viewers_million", data=df_ep, jitter=True, linewidth=1)

title = ax.set_title('Viewers vs. Ratings')
fig, ax = plt.subplots()

fig.set_size_inches(10, 5)

p = sns.stripplot(x="Imdb_Rating", y="Writer", data=df_ep_clean, jitter=True, linewidth=1)

title = ax.set_title('Writers vs. Ratings')
#df_ep_clean_1 = df_ep_clean

#df_ep_clean_1['Writer'] = df_ep_clean['Writer'].apply(lambda x: x.replace(' ', ' ') if ',' in str(x) else x)


fig, ax = plt.subplots()

fig.set_size_inches(10, 5)

p = sns.stripplot(x="Imdb_Rating", y="Director", data=df_ep_clean, jitter=True, linewidth=1)

title = ax.set_title('Directors vs. Ratings')
#!pip install bubbly
from __future__ import division

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()

from bubbly.bubbly import bubbleplot

# Adding Year column

df_ep_clean['Original_Air_Date'] = pd.to_datetime(df_ep_clean['Original_Air_Date'], format='%B %d, %Y')

df_ep_clean['Original_Air_Year'] = df_ep_clean['Original_Air_Date'].dt.year # df_ep_clean['year']
df_ep_clean.head(10)
figure = bubbleplot(dataset=df_ep_clean, x_column='US_viewers_million', y_column='Imdb_Rating'

                   ,     bubble_column='Season' 

                  ,  time_column='Original_Air_Year'

                    , size_column='IMDB_votes'

                  #  , color_column='Writer'

                     , color_column='Number_in_Season'

                       ,x_title="Viewers (millions)", y_title="IMDB Ratings", title='Viewers and Ratings by IMDB Voters over Years',

    x_logscale=True, scale_bubble=3, height=650, show_colorbar=True)



iplot(figure, config={'scrollzoom': True})