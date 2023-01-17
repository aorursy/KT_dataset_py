#Importing necessary libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

import geopandas as gpd

import pycountry

from math import pi

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected = True)

import plotly.graph_objs as go

import warnings

warnings.filterwarnings('ignore')



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)



#Read data

data=pd.read_csv('/kaggle/input/fifa-21-player-ratings/fifa21.csv', sep=';')

data.shape
#Have a look at the data

data.head(5)
#check is there any missing value in the data

data.isnull().any()
#Some team values have space at the end because of scraping. For trimming spaces we use strip function!

data['team'] = data['team'].str.strip()
#Some players play in more than one position. In the initial version of the data, they are being seperated with "|" character. 

#With this function, we are splitting them and storing them in seperate columns.



foo = lambda x: pd.Series([i for i in x.split('|')])

pos = data['position'].apply(foo)

pos.head(5)

#Create new columns and store these position info in those columns.

data['pos_1']=pos[0]

data['pos_2']=pos[1]

data['pos_3']=pos[2]

data['pos_4']=pos[3]

data['pos_5']=pos[4]



data.head(5)
#Because of being splitted into seperate columns, we don't need original position info any more.

data.drop(['position'], axis = 1,inplace=True) 
for col in ['age', 'overall', 'potential', 'hits']:

    data[col] = data[col].astype(int)
#Create functions for getting the data according to player, country and team information.

def player_data(x):

    return data.loc[data['name']==x]



def country_data(x):

    return data[data['nationality'] == x][['name','overall','potential','pos_1','hits','age','team']]



def team_data(x):

    return data[data['team'] == x][['name','overall','potential','pos_1','hits','age']]







player_data('Lionel Messi')
country_data('Turkey')
team_data('Fenerbahçe SK')
#Five eldest players

eldest = data.sort_values('age', ascending = False)[['name', 'nationality', 'age']].head(5)

eldest.set_index('name', inplace=True)

print(eldest)
#Five youngest players

youngest = data.sort_values('age', ascending = True)[['name', 'nationality', 'age']].head(5)

youngest.set_index('name', inplace=True)

print(youngest)
# The oldest team

data.groupby(['team'])['age'].mean().sort_values(ascending = False).head(5)
# The youngest team

data.groupby(['team'])['age'].mean().sort_values(ascending = True).head(5)
# The clubs with largest number of different countries

data.groupby(['team'])['nationality'].nunique().sort_values(ascending = False).head()
# The clubs with smallest number of different countries

data.groupby(['team'])['nationality'].nunique().sort_values(ascending = True).head()
# Top five teams with the best players

data.groupby(['team'])['overall'].max().sort_values(ascending = False).head()
# Top five teams with the most potential players

data.groupby(['team'])['potential'].max().sort_values(ascending = False).head()
# defining the features of players



player_features = ('age', 'overall', 'hits', 'potential')



# Top 2 features for every position in football



for i, val in data.groupby(data['pos_1'])[player_features].mean().iterrows():

    print('Position {}: {}, {}'.format(i, *tuple(val.nlargest(2).index)))



# Correlation heatmap

plt.rcParams['figure.figsize']=(16,9)

hmap=sns.heatmap(data[['age', 'overall', 'potential', 'hits']].corr(), annot = True, linewidths=.5, cmap='BuPu')

hmap.set_title(label='Heatmap of dataset', fontsize=20)

hmap;



# Scater plot shows correlation between potential and other chosen features

def scatter_plot(df):

    feats = ('age', 'overall', 'hits')

    

    for index, feat in enumerate(feats):

        plt.subplot(len(feats)/3+1, 3, index+1)

        ax = sns.regplot(x = 'potential', y = feat, data = df)



plt.figure(figsize = (12, 12))

plt.subplots_adjust(hspace = 0.4)



scatter_plot(data)


# Histogram: number of players's age

sns.set(style ="dark", palette="colorblind", color_codes=True)

x = data.age

plt.figure(figsize=(12,8))

ax = sns.distplot(x, bins = 58, kde = False, color='r')

ax.set_xlabel(xlabel="Player\'s age", fontsize=16)

ax.set_ylabel(ylabel='Number of players', fontsize=16)

ax.set_title(label='Histogram of players age', fontsize=20)

plt.show()



# Compare six teams in relation to age

turkish_teams = ('Fenerbahçe SK', 'Galatasaray SK', 'Besiktas JK', 'Sivasspor', 'Medipol Basaksehir FK', 'Trabzonspor')

df_team = data.loc[data['team'].isin(turkish_teams) & data['age']]







fig, ax = plt.subplots()

fig.set_size_inches(20, 10)

ax = sns.violinplot(x="team", y="age", data=df_team);

ax.set_title(label='Distribution of age in some teams', fontsize=20);
# Compare six teams in relation to overall ratings

turkish_teams = ('Fenerbahçe SK', 'Galatasaray SK', 'Besiktas JK', 'Sivasspor', 'Medipol Basaksehir FK', 'Trabzonspor')

df_team = data.loc[data['team'].isin(turkish_teams)  & data['overall'] ]



ax = sns.barplot(x=df_team['team'], y=df_team['overall'], palette="rocket");

ax.set_title(label='Distribution overall in several teams', fontsize=20);


plt.figure(figsize = (18, 8))

plt.style.use('fivethirtyeight')

ax = sns.countplot('pos_1', data = data, palette = 'bone')

ax.set_xlabel(xlabel = 'Different Positions in Football', fontsize = 16)

ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)

ax.set_title(label = 'Comparison of Positions and Players', fontsize = 20)

plt.show()
#Spider plot



idx = 1

plt.figure(figsize=(15,45))

for position_name, features in data.groupby(data['pos_1'])[player_features].mean().iterrows():

    top_features = dict(features.nlargest(5))

    

    # number of variable

    categories=top_features.keys()

    N = len(categories)



    # We are going to plot the first line of the data frame.

    # But we need to repeat the first value to close the circular graph:

    values = list(top_features.values())

    values += values[:1]



    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)

    angles = [n / float(N) * 2 * pi for n in range(N)]

    angles += angles[:1]



    # Initialise the spider plot

    ax = plt.subplot(9, 3, idx, polar=True)



    # Draw one axe per variable + add labels labels yet

    plt.xticks(angles[:-1], categories, color='grey', size=8)



    # Draw ylabels

    ax.set_rlabel_position(0)

    plt.yticks([25,50,75], ["25","50","75"], color="grey", size=7)

    plt.ylim(0,100)

    

    plt.subplots_adjust(hspace = 0.5)

    

    # Plot data

    ax.plot(angles, values, linewidth=1, linestyle='solid')



    # Fill area

    ax.fill(angles, values, 'b', alpha=0.1)

    

    plt.title(position_name, size=11, y=1.1)

    

    idx += 1 




sns.lineplot(data['age'], data['overall'], palette = 'Wistia')

plt.title('Age vs Overall', fontsize = 20)



plt.show()





sns.lineplot(data['age'], data['potential'], palette = 'Wistia')

plt.title('Age vs Potential', fontsize = 20)



plt.show()




rating = pd.DataFrame(data.groupby(['nationality'])['overall'].sum().reset_index())

count = pd.DataFrame(rating.groupby('nationality')['overall'].sum().reset_index())



trace = [go.Choropleth(

            colorscale = 'YlOrRd',

            locationmode = 'country names',

            locations = count['nationality'],

            text = count['nationality'],

            z = count['overall'],

)]



layout = go.Layout(title = 'Country vs Overall Ratings')



fig = go.Figure(data = trace, layout = layout)

py.iplot(fig)

         


