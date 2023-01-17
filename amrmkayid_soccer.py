%reload_ext autoreload

%autoreload 2

%matplotlib inline
# import os



# os.environ["MODIN_ENGINE"] = "ray"  # Modin will use Ray

# # os.environ["MODIN_ENGINE"] = "dask"  # Modin will use Dask
import io

import math

import base64

import folium

import sqlite3

import warnings

import itertools

import folium.plugins

import time, datetime



import scipy

import numpy as np

import pandas as pd

# import modin.pandas as pd

import seaborn as sns

import plotly.tools as tls

import plotly.offline as py

import plotly.graph_objs as go

import matplotlib.pyplot as plt



import scipy.ndimage



from scipy import stats

from collections import *

from matplotlib.pyplot import imread

from statsmodels.stats.power import TTestIndPower



from pathlib import Path

from datetime import timedelta

from subprocess import check_output

from matplotlib import animation, rc

from mpl_toolkits.basemap import Basemap
warnings.filterwarnings("ignore")

py.init_notebook_mode(connected=True)
## The data folder path

# PATH = Path(f'data/')

PATH = Path('../input/soccer/')
from IPython.core.display import HTML

from IPython.display import display



def display_tables(table_dict):

    ''' 

    Accepts a list of IpyTable objects and returns a table which contains each IpyTable in a cell

    ''' 

    template = """<div style="float: left; padding: 10px;">

                    <p style='font-family:"Courier New", Courier, monospace'>

                    <strong>{0}</strong></p>{1}</div>"""

    

    return HTML(

        '<table><tr style="background-color:white;">' + 

        '\n\n'.join(['<td>' + template.format(repr(key), table._repr_html_()) +

                     '</td>' for key, table in table_dict.items()]) +

        '</tr></table>'

    )
class DataBunch:

    __dfs__ = ['countries', 'leagues', 'matches', 'players', 

               'player_attributes', 'teams', 'team_attributes', 'sqlite_sequences']

    

    @classmethod

    def connect(cls, path):

        connection = sqlite3.connect(str(path/'database.sqlite'))

        return connection

    

    def __init__(self, path):

        self.path = path

        self.connection = DataBunch.connect(path)

        self.tables = self.get_all_tables(self.connection)

        self.dfs = dict.fromkeys(DataBunch.__dfs__ , None)

        self.set_tables(self.tables, self.connection)



    def get_all_tables(self, connection):

        cursor = connection.cursor()

        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table';")

        tables = cursor.fetchall()

        table_names = sorted([table[0] for table in tables])

        return table_names



    def set_tables(self, table_names, connection):

        dataframes = {}

        for i, name in enumerate(table_names):

            print(f'Processing {i}: {DataBunch.__dfs__[i]} dataframe | from {name} table')

            dataframes[DataBunch.__dfs__[i]] = pd.read_sql_query(

                f"SELECT * from {name}", connection)

        for key, value in dataframes.items():

            setattr(self, f'_{key}_df', value)

            self.dfs[key] = value

            

    def describe_and_check_nulls(self):

        

        cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)



        def magnify():

            return [dict(selector="th",

                         props=[("font-size", "7pt")]),

                    dict(selector="td",

                         props=[('padding', "0em 0em")]),

                    dict(selector="th:hover",

                         props=[("font-size", "12pt")]),

                    dict(selector="tr:hover td:hover",

                         props=[('max-width', '200px'),

                                ('font-size', '12pt')])]



        dfs_with_nulls = {}

        for name, df in self.dfs.items():

            print('=' * 50 +  f' {name} ' + '=' * 50)

            print(f'{name} INFO:')

            display(df.info())

            print()

            print(f'{name} Describtion:')

            display(df.describe().transpose())

            print()

            print(f'{name} Correlations:')

            corr = df.corr()

            display(corr.style.background_gradient(cmap, axis=1)\

                    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\

                    .set_caption("Hover to magify")\

                    .set_precision(2)\

                    .set_table_styles(magnify()))

            print()

            print(f'{name} NULLs:')

            display(df.isnull().sum())

            if df.isnull().sum().any():

                print(f'Found df {name} with nulls.....')

                dfs_with_nulls[name] = df

            print('\n'*5)

        return dfs_with_nulls
db = DataBunch(PATH)
db.__dict__.keys()
db.dfs.keys()
display_tables(db.dfs)
dfs_with_nulls = db.describe_and_check_nulls()
dfs_with_nulls.keys()
df = db.dfs['matches'].copy()
df.head(3)
df['league_id'].nunique()
df.isnull().sum()
total_cells = np.product(df.shape) 

total_missing = df.isnull().sum().sum()



percentage = (total_missing / total_cells) * 100

print(f'Total missing: {total_missing} \nPercentage:{percentage}')
df.dropna()['league_id'].nunique()
df = df.fillna(df.mean())
db.dfs['matches'] = df
df = db.dfs['player_attributes'].copy()
df.head(3)
df.isnull().sum()
total_cells = np.product(df.shape) 

total_missing = df.isnull().sum().sum()



percentage = (total_missing / total_cells) * 100

print(f'Total missing: {total_missing} \nPercentage:{percentage}')
df.dropna()
len(df), len(df.dropna()), 'Difference =>', len(df) - len(df.dropna())
df = df.fillna(method='bfill', axis=0).fillna(0)
db.dfs['player_attributes'] = df
df = db.dfs['teams'].copy()
df.head(3)
df.isnull().sum()
total_cells = np.product(df.shape) 

total_missing = df.isnull().sum().sum()



percentage = (total_missing / total_cells) * 100

print(f'Total missing: {total_missing} \nPercentage:{percentage}')
df[df.isnull().any(axis=1)]
df = df.drop(columns=['team_fifa_api_id'])
db.dfs['teams'] = df
df = db.dfs['team_attributes'].copy()
df.head(3)
df.isnull().sum()
total_cells = np.product(df.shape) 

total_missing = df.isnull().sum().sum()



percentage = (total_missing / total_cells) * 100

print(f'Total missing: {total_missing} \nPercentage:{percentage}')
df[df.isnull().any(axis=1)]
df = df.drop(columns=['buildUpPlayDribbling'])
db.dfs['team_attributes'] = df
display_tables(db.dfs)
df = db.dfs['players'].copy()

df.head(3)
df['birthday'] =  pd.to_datetime(df['birthday'])

df.head(3)
now = pd.Timestamp('now')

df['age'] = (now - df['birthday']).astype('<m8[Y]').apply(int)

df.head(3)
db.dfs['players'] = df
df = db.dfs['player_attributes'].copy()

df.head(3)
players_and_attr = (db.dfs['players']

                    .merge(df, on="player_api_id", how='outer')

                   .rename(columns={'player_fifa_api_id_x':"player_fifa_api_id"}))



players_and_attr = players_and_attr.drop(["id_x", "id_y", "player_fifa_api_id_y"], axis = 1)

df = db.dfs['players_and_attr'] = players_and_attr
df
df.groupby('player_name').count()
df.sort_values("date", inplace=True, ascending=False) 
df
df.drop_duplicates(subset ="player_name", 

                     keep = 'first', inplace = True) 
df.groupby('player_name').count()
players_in_2016 = df[df['date'].str.contains('2016')]
db.dfs['players_in_2016'] = players_in_2016
df1, df2 = db.dfs['countries'].copy(), db.dfs['leagues'].copy()

df = db.dfs['leagues_by_countries'] = (df1.merge(df2, left_on="id", right_on="id", how="outer")

                                           .rename(columns={'name_x':"country", 'name_y':"league"}))

df = df.drop("id", axis = 1)

df
df = db.dfs['matches'].copy()

df.head(3)
list(df.columns)
list(df.columns)[:11]
db.dfs['matches_with_less_attr'] = df[list(df.columns)[:11]].drop("id",axis=1)

db.dfs['matches_with_less_attr'].head(3)
leagues_and_matches = db.dfs['matches_with_less_attr'].merge(db.dfs['leagues_by_countries'],

                                              left_on="country_id",

                                              right_on="country_id",

                                              how="outer")



db.dfs['leagues_and_matches'] = leagues_and_matches

leagues_and_matches.head(3)
matches_df = db.dfs['matches']
matches_df['home_team_win'] = np.zeros

matches_df['away_team_win'] = np.zeros
matches_df['home_team_win'].loc[matches_df['home_team_goal'] > matches_df['away_team_goal']] = 1   #WIN

matches_df['home_team_win'].loc[matches_df['home_team_goal'] < matches_df['away_team_goal']] = 0   #LOSS

matches_df['home_team_win'].loc[matches_df['home_team_goal'] == matches_df['away_team_goal']] = 0  #TIE
matches_df['away_team_win'].loc[matches_df['home_team_goal'] < matches_df['away_team_goal']] = 1   #WIN

matches_df['away_team_win'].loc[matches_df['home_team_goal'] > matches_df['away_team_goal']] = 0   #LOSS

matches_df['away_team_win'].loc[matches_df['home_team_goal'] == matches_df['away_team_goal']] = 0  #TIE
# create numpy arrays

home_team_win_array = np.array(matches_df['home_team_win'])

away_team_win_array = np.array(matches_df['away_team_win'])



# the means of each array represent the win rate: win rate = matches won / matches NOT won (tie or loss)

x_bar_home = np.mean(home_team_win_array)

x_bar_away = np.mean(away_team_win_array)



#calculate the difference between the means, using all rows in the dataset

diff = x_bar_home - x_bar_away
diff, len(home_team_win_array), len(away_team_win_array)
n_home = len(home_team_win_array)



n_away = len(away_team_win_array)



home_wins = sum(home_team_win_array)

away_wins = sum(away_team_win_array)



home_win_rate = home_wins/n_home

away_win_rate = away_wins/n_home



diff = home_win_rate-away_win_rate

print(f"Home Win Rate: {home_win_rate} \nAway Win Rate: {away_win_rate} \nDifference: {diff}")
var_home = home_team_win_array.var()

var_away = away_team_win_array.var()

var_home, var_away
pooled_var = (n_home * var_home + n_away * var_away) / (n_home + n_away)

cohens_d = (diff) / np.sqrt(pooled_var)

cohens_d
# Initialize parameters

effect = cohens_d

alpha = 0.05

power = 0.95



# sample 2 / sample 1   

ratio = len(away_team_win_array) / len(home_team_win_array)



# Perform power analysis

analysis = TTestIndPower()

result = analysis.solve_power(effect, power=power, nobs1=None,ratio=ratio, alpha=alpha)

print(f"The minimum sample size: {result}")

print(f"Number of matches played: {len(away_team_win_array)}")
sample_means_home = []

for _ in range(1000):

    sample_mean = np.random.choice(home_team_win_array, size=202).mean()

    sample_means_home.append(sample_mean)



sample_means_away = []

for _ in range(1000):

    sample_mean = np.random.choice(away_team_win_array, size=202).mean()

    sample_means_away.append(sample_mean)

len(sample_means_home), len(sample_means_away)
sample_means_home[:3], sample_means_away[:3]
def calc_variance(sample):

    '''Computes the variance a list of values'''

    sample_mean = np.mean(sample)

    return sum([(i - sample_mean)**2 for i in sample])



def calc_sample_variance(sample1, sample2):

    '''Computes the pooled variance 2 lists of values, using the calc_variance function'''

    n_1, n_2 = len(sample1), len(sample2)

    var1, var2 = calc_variance(sample1), calc_variance(sample2)

    return (var1 + var2) / ((n_1 + n_2) - 2)



def calc_twosample_tstatistic(expr, ctrl):

    '''Computes the 2-sample T-stat of 2 lists of values, using the calc_sample_variance function'''

    expr_mean, ctrl_mean = np.mean(expr), np.mean(ctrl)

    n_e, n_c = len(expr), len(ctrl)

    samp_var = calc_sample_variance(expr,ctrl)

    t = (expr_mean - ctrl_mean) / np.sqrt(samp_var * ((1/n_e)+(1/n_c)))

    return t
t_stat = calc_twosample_tstatistic(sample_means_home, sample_means_away)



t_stat
# Using stats ttest_ind

stats.ttest_ind(sample_means_home, sample_means_away)
sns.set(color_codes=True)

sns.set(rc={'figure.figsize':(12,10)})

plt.title('Bootstrapped Win Rate Frequencies', fontsize='25')

plt.xlabel('Win Rate', fontsize='20')

plt.ylabel('Win Rate Frequency', fontsize='20')

sns.distplot(sample_means_home, label='Home Win Rates') # Blue distribution

sns.distplot(sample_means_away, label='Away Win Rates') # Orange distribution

plt.legend()

plt.show()
def visualize_t(t_stat, n_control, n_experimental):

    # initialize a matplotlib "figure"

    fig = plt.figure(figsize=(8,5))

    ax = fig.gca()

    # generate points on the x axis between -20 and 20:

    xs = np.linspace(-20, 20, 500)



    # use stats.t.pdf to get values on the probability density function for the t-distribution

    ys= stats.t.pdf(xs, (n_control+n_experimental-2), 0, 1)

    ax.plot(xs, ys, linewidth=3, color='darkred')



    ax.axvline(t_stat, color='black', linestyle='--', lw=5)

    ax.axvline(-t_stat, color='black', linestyle='--', lw=5)

    plt.xlabel('t-stat', fontsize='20')

    plt.ylabel('probability density', fontsize='20')

    plt.title('Probability Density of t-test',fontsize='25')



    plt.show()

    return None



n_home = len(home_team_win_array)

n_away = len(away_team_win_array)

visualize_t(t_stat, n_home, n_away)
## Calculate p_value manually

# Lower tail comulative density function returns area under the lower tail curve

df = len(sample_means_home)+len(sample_means_home)-2



tail = stats.t.cdf(-t_stat, df, 0, 1)



p_value = tail*2

print(p_value)
# CHECK WITH SCIPY

stats.ttest_ind(sample_means_home, sample_means_away)
result = pd.merge(matches_df,

                  db.dfs['teams'][['team_long_name','team_api_id']],

                  left_on='home_team_api_id',

                  right_on='team_api_id',

                  how='left')

result.rename(columns={"team_long_name": "home_team_name"}, inplace=True)



result = result.drop(columns='team_api_id')



results = pd.merge(result,

                  db.dfs['teams'][['team_long_name','team_api_id']],

                  left_on='away_team_api_id',

                  right_on='team_api_id',

                  how='left')



results.rename(columns={"team_long_name": "away_team_name"}, inplace=True)



results = results.drop(columns='team_api_id')
results['winning_team'] = np.nan

results.head()
results['winning_team'].loc[results['home_team_goal'] > results['away_team_goal']] = results['home_team_name']

results['winning_team'].loc[results['home_team_goal'] < results['away_team_goal']] = results['away_team_name']
results.head()
home_team_win_df = results.groupby("home_team_name").agg({

        "home_team_win": "mean",

    })



home_team_win_df.sort_values(by= 'home_team_win',ascending=False)
away_team_win_df = results.groupby("away_team_name").agg({

        "away_team_win": "mean",

    })



away_team_win_df.sort_values(by= 'away_team_win',ascending=False)
plt.figure(figsize=(16, 16)) 

plt.plot(home_team_win_df, away_team_win_df, 'o', alpha = 0.4)

plt.plot([0,1],[0,1])

plt.xlabel('Home Win Rate',fontsize='20')

plt.ylabel('Away Win Rate',fontsize='20')

plt.title('Home Win Rate vs Away Win Rate',fontsize='20')

plt.xlim([0,1])

plt.ylim([0,1])
df = db.dfs['player_attributes']
df['overall_rating'].corr(df['penalties'])
potential_features = ['acceleration', 'curve', 'free_kick_accuracy', 'ball_control', 'shot_power', 'stamina']
# check how the features are correlated with the overall ratings



for f in potential_features:

    related = df['overall_rating'].corr(df[f])

    print(f"{f}: {related}")
cols = ['potential',  'crossing', 'finishing', 'heading_accuracy',

       'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',

       'long_passing', 'ball_control', 'acceleration', 'sprint_speed',

       'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',

       'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',

       'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',

       'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',

       'gk_reflexes']
# create a list containing Pearson's correlation between 'overall_rating' with each column in cols

correlations = [ df['overall_rating'].corr(df[f]) for f in cols ]
len(cols), len(correlations)
def plot_dataframe(df, y_label):  

    """

    function for plotting a dataframe with string columns and numeric values

    """

    color='coral'

    fig = plt.gcf()

    fig.set_size_inches(20, 12)

    plt.ylabel(y_label)



    ax = df.correlation.plot(linewidth=3.5, color=color)

    ax.set_xticks(df.index)

    ax.set_xticklabels(df.attributes, rotation=75);

    plt.show()
# create a dataframe using cols and correlations

cols_and_correlations = pd.DataFrame({'attributes': cols, 'correlation': correlations})
# let's plot above dataframe using the function we created

    

plot_dataframe(cols_and_correlations, 'Player\'s Overall Rating')
# create a dataset containing only the birthday, date and overall_ratings from the Field Players dataset.

df = db.dfs['players_and_attr'][['birthday','date','overall_rating']]

df.head(2)
# converting the birthday and date columns to date type

df['birthday'] = pd.to_datetime(df['birthday'])

df['date'] = pd.to_datetime(df['date'])



# adding a column listing the age for each row

df['age'] = df['date'].dt.year - df['birthday'].dt.year

df.head()
# finding the average ratings for each age group in the dataset

df_age_ratings = df.groupby('age').mean()



# finding the number of players in each group

df.groupby('age').count()
# setting up the parameters and plotting the scatter plot



locations = df_age_ratings.index.values

height = df_age_ratings['overall_rating']

plt.style.use('ggplot')



# plotting scatter plot

plt.plot(locations, height,'-o')



# set title and labels

plt.title('Age Vs Overall Rating Relationship Chart')

plt.xlabel('Age of the players')

plt.ylabel('Overall Ratings')

plt.rcParams['figure.figsize'] = (16, 16)
# setting up the parameters and plotting the scatter plot



locations = df.age

height = df['overall_rating']

plt.style.use('ggplot')



# plotting scatter plot

plt.plot(locations, height,'-o')



# set title and labels

plt.title('Age Vs Overall Rating Relationship Chart')

plt.xlabel('Age of the players')

plt.ylabel('Overall Ratings')

plt.rcParams['figure.figsize'] = (25, 25)
db.dfs.keys()
df = db.dfs['leagues_and_matches']
plt.figure(figsize=(12,12))

ax = sns.countplot(y = df["league"],

                   order=df["league"].value_counts().index,

                   linewidth = 1,

                   edgecolor = "k"*df["league"].nunique())



for i, j in enumerate(df["league"].value_counts().values):

    ax.text(3, i, j, weight = "bold")

plt.title("Matches by league")

plt.show()
df.groupby("league").agg({"home_team_goal":"sum","away_team_goal":"sum"}).plot(kind="barh",

                                                                                 figsize = (12,12),

                                                                                 edgecolor = "k",

                                                                                 linewidth = 1

                                                                                )

plt.title("Home and away goals by league")

plt.legend(loc = "best" , prop = {"size" : 14})

plt.xlabel("total goals")

plt.show()
df = db.dfs['players_and_attr']

df
def show_player_stats(name='Lionel Messi'):

    # Players: 'Cristiano Ronaldo', 'Lionel Messi', 'Neymar', 'Heung-Min Son', etc...

    player_info = db.dfs['players_and_attr']

    player = player_info[player_info["player_name"] == name]

    cols = ['player_name','overall_rating', 'finishing', 

            'heading_accuracy', 'short_passing', 'dribbling', 

            'sprint_speed', 'shot_power', 'jumping', 'stamina',

            'strength', 'positioning', 'penalties', 'sliding_tackle']



    player = player[cols]

    player = player.groupby("player_name")[cols].mean().reset_index()



    plt.figure(figsize=(8,8))

    ax = plt.subplot(projection="polar")

    cats = list(player)[1:]

    N    = len(cats)



    mean_values = player_info.iloc[:,:].mean()

    mean_values = mean_values[cols]

    

    values = mean_values.drop("player_name").values.flatten().tolist()

    values += values[:1]

    angles = [n / float(N)*2* math.pi for n in range(N)]

    angles += angles[:1]



    plt.xticks(angles[:-1],cats,color="r",size=7)

    plt.ylim([0,100])

    plt.plot(angles,values,color='r',linewidth=2,linestyle="solid")

    plt.fill(angles,values,color='r',alpha=1)



    values = player.loc[0].drop("player_name").values.flatten().tolist()

    values += values[:1]

    angles = [n / float(N)*2* math.pi for n in range(N)]

    angles += angles[:1]



    plt.xticks(angles[:-1],cats,color="k",size=12)

    plt.ylim([0,100])

    plt.plot(angles,values,color='y',linewidth=3,linestyle="solid")

    plt.fill(angles,values,color='y',alpha=0.5)



    plt.gca().legend(('Average', name), bbox_to_anchor=(1, 0.5, 0.5, 0.5), loc=8)

    plt.title(name,color="b", fontsize=18)

    plt.subplots_adjust(wspace=.4,hspace=.4)
show_player_stats()
show_player_stats('Cristiano Ronaldo')
players = db.dfs['players_and_attr']

players.head(3)
players = players.sort_values('overall_rating', ascending=False)

best_players = players[['player_api_id','player_name']].head(20)

ids = tuple(best_players.player_api_id.unique())



query = '''SELECT player_api_id, date, overall_rating, potential

           FROM Player_attributes WHERE player_api_id in %s''' % (ids,)



evolution = pd.read_sql(query, db.connection)

evolution = pd.merge(evolution, best_players)

evolution['year'] = evolution.date.str[:4].apply(int)

evolution = evolution.groupby(['year', 'player_api_id','player_name']).overall_rating.mean()

evolution = evolution.reset_index()



evolution.head()
a = sns.factorplot(data=evolution[evolution.player_api_id.isin(ids)], x='year',

                   y='overall_rating', hue='player_name', size=10, aspect=2)
def viz_matchday_squads(matches_df=db.dfs['matches'], players_df = db.dfs["players"], 

                        team_df = db.dfs['teams'], league_id = 1729, match_api_id = 489051):

    r"""

    This method is used to get all the data needed to visualize the arena, teams players and strategies of each team

    """

    

    match_details = matches_df[matches_df['match_api_id'] == match_api_id]

#     print(match_details)

    

    # Position of the GK is set to (1,1) in the dataset which is incorrect

    match_details['home_player_X1'] = 5

    match_details['home_player_Y1'] = 0

    match_details['away_player_X1'] = 5

    match_details['away_player_Y1'] = 0

    

    

    home_x_coordinates, home_y_coordinates = [], []

    away_x_coordinates, away_y_coordinates = [], []

    home_players, away_players = [], []

    

    # Obtain the coordinates to denote each player's position on the field

    for i in range(1, 12):

        home_player_x_coordinate = 'home_player_X%d' % i

        home_player_y_coordinate = 'home_player_Y%d' % i

        away_player_x_coordinate = 'away_player_X%d' % i

        away_player_y_coordinate = 'away_player_Y%d' % i



        home_x_coordinates.append(match_details[home_player_x_coordinate].iloc[0])

        home_y_coordinates.append((match_details[home_player_y_coordinate].iloc[0] + 15))

        away_x_coordinates.append(match_details[away_player_x_coordinate].iloc[0])

        away_y_coordinates.append((match_details[away_player_y_coordinate].iloc[0] + 35))



        # Obtain the players' names

        home_players.append(list(players_df[players_df['player_api_id'] 

                                            == match_details['home_player_%d' % i].iloc[0]]['player_name'])[0])



        away_players.append(list(players_df[players_df['player_api_id']

                                            == match_details['away_player_%d' % i].iloc[0]]['player_name'])[0])

        

        

    # Names of the Teams

    home_team = team_df[team_df['team_api_id'] == match_details['home_team_api_id'].iloc[0]]['team_long_name'].iloc[0]

    away_team = team_df[team_df['team_api_id'] == match_details['away_team_api_id'].iloc[0]]['team_long_name'].iloc[0]

    home_team, away_team

    

    

    #Formations of the Teams

    home_formation = np.unique(home_y_coordinates, return_counts = True)[1]

    away_formation = np.unique(away_y_coordinates, return_counts = True)[1]    



    img_path = PATH/'arena.jpg'

    

    

    # Home team in Orange

    plt.figure(figsize=(20,13))

    for label, x, y in zip(home_players, home_x_coordinates, home_y_coordinates):

        plt.annotate(

            label,

            xy = (x, y), xytext = (len(label)*-4, 20),

            textcoords = 'offset points',

            fontsize= 15,

            color = '#F2F3F4'

        )

    img = imread(img_path) #Background field image

    plt.title(home_team, loc = 'left', fontsize = 25)

    plt.title("Home Team", fontsize = 25)



    formation = "Formation: "

    for i in range(1,len(home_formation)):

        formation = formation + str(home_formation[i]) + "-"

    formation = formation[:-1]



    plt.title(formation, loc = 'right', fontsize = 25)

    plt.scatter(home_x_coordinates, home_y_coordinates, s = 500, color = '#F57C00', zorder = 2)

    plt.imshow(scipy.ndimage.rotate(img, 270), zorder = 1, extent=[min(home_x_coordinates)-1, max(home_x_coordinates)+1, min(home_y_coordinates)-1, max(home_y_coordinates)+1.7], aspect = 'auto')

    plt.gca().invert_yaxis() # Invert y axis to start with the goalkeeper at the top





    # Away team in Blue

    plt.figure(figsize=(20, 13))

    plt.gca().invert_xaxis() # Invert x axis to have right wingers on the right

    for label, x, y in zip(away_players, away_x_coordinates, away_y_coordinates):

        plt.annotate(

            label,

            xy = (x, y), xytext = (len(label)*-4, -30),

            textcoords = 'offset points',

            fontsize= 15,

            color = '#F2F3F4'

        )

    img = imread(img_path)

    plt.title(away_team, loc = 'left', fontsize = 25)

    plt.title("Away Team", fontsize = 25)



    formation = "Formation: "

    for i in range(1,len(away_formation)):

        formation = formation + str(away_formation[i]) + "-"

    formation = formation[:-1]



    plt.title(formation, loc = 'right', fontsize = 25)

    plt.scatter(away_x_coordinates, away_y_coordinates, s = 500, color = '#0277BD', zorder = 2)

    plt.imshow(scipy.ndimage.rotate(img, 270), zorder = 1, extent=[min(away_x_coordinates)-1, max(away_x_coordinates)+1, min(away_y_coordinates)-1, max(away_y_coordinates)+1.6], aspect = 'auto')

    plt.show()
# viz_matchday_squads()
# viz_matchday_squads(league_id=24558, match_api_id=1992095)