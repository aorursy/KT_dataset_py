import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import collections

import warnings



from IPython.core.display import display, HTML



# import plotly 

import plotly

import plotly.figure_factory as ff

import plotly.graph_objs as go

import plotly.offline as py

from plotly.offline import iplot, init_notebook_mode

import plotly.tools as tls



# configure things

warnings.filterwarnings('ignore')



pd.options.display.float_format = '{:,.2f}'.format  

pd.options.display.max_columns = 999



py.init_notebook_mode(connected=True)



%load_ext autoreload

%autoreload 2



%matplotlib inline

sns.set()



# !pip install plotly --upgrade
# # func to make plotly work in Collaboratory (not necessary on Kaggle)

# def configure_plotly_browser_state():

#   import IPython

#   display(IPython.core.display.HTML('''

# <script src="/static/components/requirejs/require.js"></script>

# <script>

#   requirejs.config({

#     paths: {

#       base: 'static/base',

#       plotly: 'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',

#     },

#   });

# </script>

# '''))
# import os

# for dirname, _, filenames in os.walk('../input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



df = pd.read_csv('../input/extended-football-stats-for-european-leagues-xg/understat.com.csv')

df = df.rename(index=int, columns={'Unnamed: 0': 'league', 'Unnamed: 1': 'year'}) 

df.head()
f = plt.figure(figsize=(25,12))

ax = f.add_subplot(2,3,1)

plt.xticks(rotation=45)

sns.barplot(x='team', y='pts', hue='year', data=df[(df['league'] == 'Bundesliga') & (df['position'] <= 4)], ax=ax)

ax = f.add_subplot(2,3,2)

plt.xticks(rotation=45)

sns.barplot(x='team', y='pts', hue='year', data=df[(df['league'] == 'EPL') & (df['position'] <= 4)], ax=ax)

ax = f.add_subplot(2,3,3)

plt.xticks(rotation=45)

sns.barplot(x='team', y='pts', hue='year', data=df[(df['league'] == 'La_liga') & (df['position'] <= 4)], ax=ax)

ax = f.add_subplot(2,3,4)

plt.xticks(rotation=45)

sns.barplot(x='team', y='pts', hue='year', data=df[(df['league'] == 'Serie_A') & (df['position'] <= 4)], ax=ax)

ax = f.add_subplot(2,3,5)

plt.xticks(rotation=45)

sns.barplot(x='team', y='pts', hue='year', data=df[(df['league'] == 'Ligue_1') & (df['position'] <= 4)], ax=ax)

ax = f.add_subplot(2,3,6)

plt.xticks(rotation=45)

sns.barplot(x='team', y='pts', hue='year', data=df[(df['league'] == 'RFPL') & (df['position'] <= 4)], ax=ax)
# Removing unnecessary for our analysis columns 

df_xg = df[['league', 'year', 'position', 'team', 'scored', 'xG', 'xG_diff', 'missed', 'xGA', 'xGA_diff', 'pts', 'xpts', 'xpts_diff']]



outlier_teams = ['Wolfsburg', 'Schalke 04', 'Leicester', 'Villareal', 'Sevilla', 'Lazio', 'Fiorentina', 'Lille', 'Saint-Etienne', 'FC Rostov', 'Dinamo Moscow']
# Checking if getting the first place requires fenomenal execution

first_place = df_xg[df_xg['position'] == 1]



# Get list of leagues

leagues = df['league'].drop_duplicates()

leagues = leagues.tolist()



# Get list of years

years = df['year'].drop_duplicates()

years = years.tolist()
first_place[first_place['league'] == 'Bundesliga']
pts = go.Bar(x = years, y = first_place['pts'][first_place['league'] == 'Bundesliga'], name = 'PTS')

xpts = go.Bar(x = years, y = first_place['xpts'][first_place['league'] == 'Bundesliga'], name = 'Expected PTS')



data = [pts, xpts]



layout = go.Layout(

    barmode='group',

    title="Comparing Actual and Expected Points for Winner Team in Bundesliga",

    xaxis={'title': 'Year'},

    yaxis={'title': "Points",

    }

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
# and from this table we see that Bayern dominates here totally, even when they do not play well

df_xg[(df_xg['position'] <= 2) & (df_xg['league'] == 'Bundesliga')].sort_values(by=['year','xpts'], ascending=False)
first_place[first_place['league'] == 'La_liga']
pts = go.Bar(x = years, y = first_place['pts'][first_place['league'] == 'La_liga'], name = 'PTS')

xpts = go.Bar(x = years, y = first_place['xpts'][first_place['league'] == 'La_liga'], name = 'Expected PTS')



data = [pts, xpts]



layout = go.Layout(

    barmode='group',

    title="Comparing Actual and Expected Points for Winner Team in La Liga",

    xaxis={'title': 'Year'},

    yaxis={'title': "Points",

    }

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
# comparing with runner-up

df_xg[(df_xg['position'] <= 2) & (df_xg['league'] == 'La_liga')].sort_values(by=['year','xpts'], ascending=False)
first_place[first_place['league'] == 'EPL']
pts = go.Bar(x = years, y = first_place['pts'][first_place['league'] == 'EPL'], name = 'PTS')

xpts = go.Bar(x = years, y = first_place['xpts'][first_place['league'] == 'EPL'], name = 'Expected PTS')



data = [pts, xpts]



layout = go.Layout(

    barmode='group',

    title="Comparing Actual and Expected Points for Winner Team in EPL",

    xaxis={'title': 'Year'},

    yaxis={'title': "Points",

    }

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
# comparing with runner-ups

df_xg[(df_xg['position'] <= 2) & (df_xg['league'] == 'EPL')].sort_values(by=['year','xpts'], ascending=False)
first_place[first_place['league'] == 'Ligue_1']
pts = go.Bar(x = years, y = first_place['pts'][first_place['league'] == 'Ligue_1'], name = 'PTS')

xpts = go.Bar(x = years, y = first_place['xpts'][first_place['league'] == 'Ligue_1'], name = 'Expected PTS')



data = [pts, xpts]



layout = go.Layout(

    barmode='group',

    title="Comparing Actual and Expected Points for Winner Team in Ligue 1",

    xaxis={'title': 'Year'},

    yaxis={'title': "Points",

    }

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
# comparing with runner-ups

df_xg[(df_xg['position'] <= 2) & (df_xg['league'] == 'Ligue_1')].sort_values(by=['year','xpts'], ascending=False)
first_place[first_place['league'] == 'Serie_A']
pts = go.Bar(x = years, y = first_place['pts'][first_place['league'] == 'Serie_A'], name = 'PTS')

xpts = go.Bar(x = years, y = first_place['xpts'][first_place['league'] == 'Serie_A'], name = 'Expecetd PTS')



data = [pts, xpts]



layout = go.Layout(

    barmode='group',

    title="Comparing Actual and Expected Points for Winner Team in Serie A",

    xaxis={'title': 'Year'},

    yaxis={'title': "Points",

    }

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
# comparing to runner-ups

df_xg[(df_xg['position'] <= 2) & (df_xg['league'] == 'Serie_A')].sort_values(by=['year','xpts'], ascending=False)
first_place[first_place['league'] == 'RFPL']
pts = go.Bar(x = years, y = first_place['pts'][first_place['league'] == 'RFPL'], name = 'PTS')

xpts = go.Bar(x = years, y = first_place['xpts'][first_place['league'] == 'RFPL'], name = 'Expected PTS')



data = [pts, xpts]



layout = go.Layout(

    barmode='group',

    title="Comparing Actual and Expected Points for Winner Team in RFPL",

    xaxis={'title': 'Year'},

    yaxis={'title': "Points",

    }

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
# comparing to runner-ups

df_xg[(df_xg['position'] <= 2) & (df_xg['league'] == 'RFPL')].sort_values(by=['year','xpts'], ascending=False)
# Creating separate DataFrames per each league

laliga = df_xg[df_xg['league'] == 'La_liga']

laliga.reset_index(inplace=True)

epl = df_xg[df_xg['league'] == 'EPL']

epl.reset_index(inplace=True)

bundesliga = df_xg[df_xg['league'] == 'Bundesliga']

bundesliga.reset_index(inplace=True)

seriea = df_xg[df_xg['league'] == 'Serie_A']

seriea.reset_index(inplace=True)

ligue1 = df_xg[df_xg['league'] == 'Ligue_1']

ligue1.reset_index(inplace=True)

rfpl = df_xg[df_xg['league'] == 'RFPL']

rfpl.reset_index(inplace=True)
laliga.describe()
def print_records_antirecords(df):

  print('Presenting some records and antirecords: \n')

  for col in df.describe().columns:

    if col not in ['index', 'year', 'position']:

      team_min = df['team'].loc[df[col] == df.describe().loc['min',col]].values[0]

      year_min = df['year'].loc[df[col] == df.describe().loc['min',col]].values[0]

      team_max = df['team'].loc[df[col] == df.describe().loc['max',col]].values[0]

      year_max = df['year'].loc[df[col] == df.describe().loc['max',col]].values[0]

      val_min = df.describe().loc['min',col]

      val_max = df.describe().loc['max',col]

      print('The lowest value of {0} had {1} in {2} and it is equal to {3:.2f}'.format(col.upper(), team_min, year_min, val_min))

      print('The highest value of {0} had {1} in {2} and it is equal to {3:.2f}'.format(col.upper(), team_max, year_max, val_max))

      print('='*100)

      

# replace laliga with any league you want

print_records_antirecords(laliga)
trace0 = go.Scatter(

    x = laliga['position'][laliga['year'] == 2014], 

    y = laliga['xG_diff'][laliga['year'] == 2014],

    name = '2014',

    mode = 'lines+markers'

)



trace1 = go.Scatter(

    x = laliga['position'][laliga['year'] == 2015], 

    y = laliga['xG_diff'][laliga['year'] == 2015],

    name='2015',

    mode = 'lines+markers'

)



trace2 = go.Scatter(

    x = laliga['position'][laliga['year'] == 2016], 

    y = laliga['xG_diff'][laliga['year'] == 2016],

    name='2016',

    mode = 'lines+markers'

)



trace3 = go.Scatter(

    x = laliga['position'][laliga['year'] == 2017], 

    y = laliga['xG_diff'][laliga['year'] == 2017],

    name='2017',

    mode = 'lines+markers'

)



trace4 = go.Scatter(

    x = laliga['position'][laliga['year'] == 2018], 

    y = laliga['xG_diff'][laliga['year'] == 2018],

    name='2018',

    mode = 'lines+markers'

)



data = [trace0, trace1, trace2, trace3, trace4]



layout = go.Layout(

    title="Comparing xG gap between positions",

    xaxis={'title': 'Year'},

    yaxis={'title': "xG difference",

    }

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
trace0 = go.Scatter(

    x = laliga['position'][laliga['year'] == 2014], 

    y = laliga['xGA_diff'][laliga['year'] == 2014],

    name = '2014',

    mode = 'lines+markers'

)



trace1 = go.Scatter(

    x = laliga['position'][laliga['year'] == 2015], 

    y = laliga['xGA_diff'][laliga['year'] == 2015],

    name='2015',

    mode = 'lines+markers'

)



trace2 = go.Scatter(

    x = laliga['position'][laliga['year'] == 2016], 

    y = laliga['xGA_diff'][laliga['year'] == 2016],

    name='2016',

    mode = 'lines+markers'

)



trace3 = go.Scatter(

    x = laliga['position'][laliga['year'] == 2017], 

    y = laliga['xGA_diff'][laliga['year'] == 2017],

    name='2017',

    mode = 'lines+markers'

)



trace4 = go.Scatter(

    x = laliga['position'][laliga['year'] == 2018], 

    y = laliga['xGA_diff'][laliga['year'] == 2018],

    name='2018',

    mode = 'lines+markers'

)



data = [trace0, trace1, trace2, trace3, trace4]



layout = go.Layout(

    title="Comparing xGA gap between positions",

    xaxis={'title': 'Year'},

    yaxis={'title': "xGA difference",

    }

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
trace0 = go.Scatter(

    x = laliga['position'][laliga['year'] == 2014], 

    y = laliga['xpts_diff'][laliga['year'] == 2014],

    name = '2014',

    mode = 'lines+markers'

)



trace1 = go.Scatter(

    x = laliga['position'][laliga['year'] == 2015], 

    y = laliga['xpts_diff'][laliga['year'] == 2015],

    name='2015',

    mode = 'lines+markers'

)



trace2 = go.Scatter(

    x = laliga['position'][laliga['year'] == 2016], 

    y = laliga['xpts_diff'][laliga['year'] == 2016],

    name='2016',

    mode = 'lines+markers'

)



trace3 = go.Scatter(

    x = laliga['position'][laliga['year'] == 2017], 

    y = laliga['xpts_diff'][laliga['year'] == 2017],

    name='2017',

    mode = 'lines+markers'

)



trace4 = go.Scatter(

    x = laliga['position'][laliga['year'] == 2018], 

    y = laliga['xpts_diff'][laliga['year'] == 2018],

    name='2018',

    mode = 'lines+markers'

)



data = [trace0, trace1, trace2, trace3, trace4]



layout = go.Layout(

    title="Comparing xPTS gap between positions",

    xaxis={'title': 'Position'},

    yaxis={'title': "xPTS difference",

    }

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
# Check mean differences

def get_diff_means(df):  

  dm = df.groupby('year')[['xG_diff', 'xGA_diff', 'xpts_diff']].mean()

  

  return dm



means = get_diff_means(laliga)

means
# Check median differences

def get_diff_medians(df):  

  dm = df.groupby('year')[['xG_diff', 'xGA_diff', 'xpts_diff']].median()

  

  return dm



medians = get_diff_medians(laliga)

medians
# Getting outliers for xG using zscore

from scipy.stats import zscore

# laliga[(np.abs(zscore(laliga[['xG_diff']])) > 2.0).all(axis=1)]

df_xg[(np.abs(zscore(df_xg[['xG_diff']])) > 3.0).all(axis=1)]
# outliers for xGA

# laliga[(np.abs(zscore(laliga[['xGA_diff']])) > 2.0).all(axis=1)]

df_xg[(np.abs(zscore(df_xg[['xGA_diff']])) > 3.0).all(axis=1)]
# Outliers for xPTS

# laliga[(np.abs(zscore(laliga[['xpts_diff']])) > 2.0).all(axis=1)]

df_xg[(np.abs(zscore(df_xg[['xpts_diff']])) > 3.0).all(axis=1)]
# Trying different method of outliers detection

df_xg.describe()
# using Interquartile Range Method to identify outliers

# xG_diff

iqr_xG = (df_xg.describe().loc['75%','xG_diff'] - df_xg.describe().loc['25%','xG_diff']) * 1.5

upper_xG = df_xg.describe().loc['75%','xG_diff'] + iqr_xG

lower_xG = df_xg.describe().loc['25%','xG_diff'] - iqr_xG



print('IQR for xG_diff: {:.2f}'.format(iqr_xG))

print('Upper border for xG_diff: {:.2f}'.format(upper_xG))

print('Lower border for xG_diff: {:.2f}'.format(lower_xG))



outliers_xG = df_xg[(df_xg['xG_diff'] > upper_xG) | (df_xg['xG_diff'] < lower_xG)]

print('='*50)



# xGA_diff

iqr_xGA = (df_xg.describe().loc['75%','xGA_diff'] - df_xg.describe().loc['25%','xGA_diff']) * 1.5

upper_xGA = df_xg.describe().loc['75%','xGA_diff'] + iqr_xGA

lower_xGA = df_xg.describe().loc['25%','xGA_diff'] - iqr_xGA



print('IQR for xGA_diff: {:.2f}'.format(iqr_xGA))

print('Upper border for xGA_diff: {:.2f}'.format(upper_xGA))

print('Lower border for xGA_diff: {:.2f}'.format(lower_xGA))



outliers_xGA = df_xg[(df_xg['xGA_diff'] > upper_xGA) | (df_xg['xGA_diff'] < lower_xGA)]

print('='*50)



# xpts_diff

iqr_xpts = (df_xg.describe().loc['75%','xpts_diff'] - df_xg.describe().loc['25%','xpts_diff']) * 1.5

upper_xpts = df_xg.describe().loc['75%','xpts_diff'] + iqr_xpts

lower_xpts = df_xg.describe().loc['25%','xpts_diff'] - iqr_xpts



print('IQR for xPTS_diff: {:.2f}'.format(iqr_xpts))

print('Upper border for xPTS_diff: {:.2f}'.format(upper_xpts))

print('Lower border for xPTS_diff: {:.2f}'.format(lower_xpts))



outliers_xpts = df_xg[(df_xg['xpts_diff'] > upper_xpts) | (df_xg['xpts_diff'] < lower_xpts)]

print('='*50)



outliers_full = pd.concat([outliers_xG, outliers_xGA, outliers_xpts])

outliers_full = outliers_full.drop_duplicates()
# Adding ratings bottom to up to find looser in each league (different amount of teams in every league so I can't do just n-20)

max_position = df_xg.groupby('league')['position'].max()

df_xg['position_reverse'] = np.nan

outliers_full['position_reverse'] = np.nan



for i, row in df_xg.iterrows():

  df_xg.at[i, 'position_reverse'] = np.abs(row['position'] - max_position[row['league']])+1

  

for i, row in outliers_full.iterrows():

  outliers_full.at[i, 'position_reverse'] = np.abs(row['position'] - max_position[row['league']])+1
total_count = df_xg[(df_xg['position'] <= 4) | (df_xg['position_reverse'] <= 3)].count()[0]

outlier_count = outliers_full[(outliers_full['position'] <= 4) | (outliers_full['position_reverse'] <= 3)].count()[0]

outlier_prob = outlier_count / total_count

print('Probability of outlier in top or bottom of the final table: {:.2%}'.format(outlier_prob))
# 1-3 outliers among all leagues in a year

data = pd.DataFrame(outliers_full.groupby('league')['year'].count()).reset_index()

data = data.rename(index=int, columns={'year': 'outliers'})

sns.barplot(x='league', y='outliers', data=data)

# no outliers in Bundesliga
top_bottom = outliers_full[(outliers_full['position'] <= 4) | (outliers_full['position_reverse'] <= 3)].sort_values(by='league')

top_bottom
# Let's get back to our list of teams that suddenly got into top. Was that because of unbeliavable mix of luck and skill?

ot = [x for x  in outlier_teams if x in top_bottom['team'].drop_duplicates().tolist()]

ot

# The answer is absolutely no. They just played well during 1 season. Sometimes that happen.