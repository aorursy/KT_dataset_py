# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



import matplotlib



import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
matches_csv = '/kaggle/input/ipl/matches.csv'

deliveries_csv = '/kaggle/input/ipl/deliveries.csv'



matches_df = pd.read_csv(matches_csv)

deliveries_df = pd.read_csv(deliveries_csv) 
matches_df.info()
column = 'umpire3'

matches_df = matches_df.drop(columns= column)
column = 'winner'

matches_df[matches_df[column].isnull()]
column = 'winner'

matches_df[column] = matches_df[column].fillna('Undeclared')

column = 'player_of_match'

matches_df[column] = matches_df[column].fillna('Undeclared')
column = 'umpire1'

matches_df[matches_df[column].isnull()]
column = 'umpire1'

matches_df[column] = matches_df[column].fillna('Unknown')



column = 'umpire2'

matches_df[column] = matches_df[column].fillna('Unknown')
column = 'city'

matches_df[matches_df[column].isnull()]
column = 'city'

matches_df[column] = matches_df[column].fillna('Dubai')
matches_df.info()
deliveries_df.info()
deliveries_df.head()
columns = ['player_dismissed', 'dismissal_kind', 'fielder']



for column in columns:

    deliveries_df[column] = deliveries_df[column].fillna(0)
deliveries_df.info()
#source : https://www.schemecolor.com/ipl-cricket-team-color-codes.php

team_colors = {

    'MI'  : '#004BA0',

#     'CSK' : '#FFFF3C',

    'CSK' : '#E1AD01',

    'SRH' : '#FF822A',

    'KKR' : '#2E0854',

#     'RCB' : '#EC1C24',

    'RCB' : '#8B0000',

    'RR'  : '#254AA5',

    'KXIP': '#ED1B24',

    'GL'  : '#8C411E',

    'DC'  : '#366293',

    'RPS' : '#2C04A2',

    'PW'  : 'black',

    'KTK' : 'gray',

    'DD'  : '#00008B',

    'Undeclared': 'white'

}



def drawCountOnBar(axes, orient = "v"):

    for p in axes.patches:

        if orient == "v":

            height = p.get_height()

            axes.text(x = p.get_x()+p.get_width()/2., y = height + 1 ,s = height ,ha="center")

        else:

            width = p.get_width()

            axes.text(x = p.get_x() + width, y = p.get_y() + p.get_height()/2 ,s = width ,ha="left")



def autopct_format(values):

    def my_format(pct):

        total = sum(values)

        val = int(round(pct*total/100.0))

        return '{v:d}'.format(v=val)

    return my_format



def drawPieChart(axes, values, top = 5, title ='Top 5', colors = None):

    

    # axes : axes object of matplotlib

    # values : Pandas Series

    # top : to filter top values for charting

    color_map = colors

    if colors == True:

        color_map = []

        for team in values.index:

            color_map.append(team_colors[team])

        

    explode = (0.2, ) + (0.02,) * (top - 1)

    axes.pie(x = count_values[:top], explode = explode, labels = count_values[:top].index ,

            autopct=autopct_format(count_values[:top]),

            colors = color_map,

            shadow=True, startangle=0)

    axes.set_title(title,fontsize=16 )

    axes.axis('equal')

    

    return axes
teams = ['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',

                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',

                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors','Rising Pune Supergiant']

team_abbrevations = ['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW','RPS']



matches_df.replace(teams,team_abbrevations,inplace=True)

deliveries_df.replace(teams,team_abbrevations,inplace=True)
matches_df.columns
column = 'toss_winner'



# fig=plt.figure(figsize=(15,5))



# axes = sns.countplot(x=column, data=matches_df, 

#                      order=matches_df[column].value_counts().sort_values(ascending = False).index,

#                     palette="rocket")

# axes.set_title('Toss winners count across all seasons (By Matches)')

# axes.set_xlabel('Teams')

# axes.set_ylabel('Count')

# drawCountOnBar(axes, orient = "v")



count_values = matches_df[column].value_counts().sort_values(ascending = False)



fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(20, 7))

fig1.suptitle('Toss winners count across all seasons (By Matches)', fontsize=16)

ax1 = drawPieChart(ax1, count_values, colors = True)



ax2.bar(x = count_values.index, height =count_values)

for index, team in enumerate(count_values.index):

    ax2.get_children()[index].set_color(team_colors[team]) 



ax2.set_title("All Teams")

drawCountOnBar(ax2, orient = "v")

plt.show()

column = 'toss_winner'

winner_seasons = matches_df.groupby('season')[column].value_counts()





season_start = 2008

season_end = 2016

rows = 3

cols = 3

fig1, axes = plt.subplots(nrows= rows, ncols= cols,figsize=(18, 15))

fig1.suptitle('Toss winners (Top 5)', fontsize=16)



for season in range(season_end, season_start -1 , -1):

    count_values = winner_seasons[season]

    row = (season_end - season) // cols

    col = (season_end - season) % cols

    axes[row][col] = drawPieChart(axes[row][col], count_values, title = season, colors=True)

plt.show()
column = 'winner'



count_values = matches_df[column].value_counts().sort_values(ascending = False)



fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(20, 7))

fig1.suptitle('Winners count across all seasons (By Matches)', fontsize=16)



ax1 = drawPieChart(ax1, count_values, colors = True)



ax2.bar(x = count_values.index, height =count_values)

for index, team in enumerate(count_values.index):

    ax2.get_children()[index].set_color(team_colors[team]) 



ax2.set_title("All Teams")

drawCountOnBar(ax2, orient = "v")

plt.show()
column = 'winner'

winner_seasons = matches_df.groupby('season')[column].value_counts()





season_start = 2008

season_end = 2016

rows = 3

cols = 3

fig1, axes = plt.subplots(nrows= rows, ncols= cols,figsize=(18, 15))

fig1.suptitle('Most Match Winners (Top 5)', fontsize=16)



for season in range(season_end, season_start -1 , -1):

    count_values = winner_seasons[season]

    row = (season_end - season) // cols

    col = (season_end - season) % cols

    axes[row][col] = drawPieChart(axes[row][col], count_values, title = season, colors=True)

plt.show()
column = 'player_of_match'



fig=plt.figure(figsize=(20, 10))



axes = sns.countplot(y=column, data=matches_df, 

                     order=matches_df[column].value_counts()[:20].sort_values(ascending = False).index,

                    palette="rocket", orient = "h")

axes.set_title('Player of match across all seasons (By Matches)')

axes.set_xlabel('Matches')

axes.set_ylabel('Playet')

drawCountOnBar(axes, orient = "h")
#install Bar chart package

!pip install bar-chart-race

!apt install ffmpeg
column = 'player_of_match'

# print(len(matches_df[column].unique()))



pre_bar_race_df = matches_df[['date', 'winner', 'season','player_of_match']]

pre_bar_race_df = pre_bar_race_df.sort_values(by=['season'], ignore_index=True, ascending = True)



players = matches_df[column].unique().tolist()

columns = ['season'] + players



bar_race_df = pd.DataFrame(columns = columns)



row_dict = dict.fromkeys(columns, 0)



for index in range(len(pre_bar_race_df)):

    counts = pre_bar_race_df[: index + 1]['player_of_match'].value_counts()

    row_dict['season'] = pre_bar_race_df['season'][index]

    for player in counts.index:

        row_dict[player] = counts[player]

    

#     print(index,bar_race_df)

    bar_race_df = bar_race_df.append(row_dict, ignore_index=True)



bar_race_df.index = bar_race_df['season']

bar_race_df = bar_race_df.drop(columns = 'season')

bar_race_df = bar_race_df.apply(pd.to_numeric)





### Uncomment to create bar chart again



# import bar_chart_race as bcr



# bcr.bar_chart_race(

#     df=bar_race_df,

#     filename='horiz.mp4',

#     orientation='h',

#     sort='desc',

#     n_bars=8,

#     fixed_order=False,

#     fixed_max=False,

#     steps_per_period=50,

#     interpolate_period=False,

#     label_bars=True,

#     bar_size=.95,

#     period_label={'x': .99, 'y': .25, 'ha': 'right', 'va': 'center'},

#     period_fmt='Season {x:.0f}',

#     period_summary_func=lambda v, r: {'x': .99, 'y': .18,

#                                       's': f'Total Matches: {v.sum():,.0f}',

#                                       'ha': 'right', 'size': 8, 'family': 'Courier New'},

# #     perpendicular_bar_func='median',

# #     period_length=500,

#     figsize=(5, 3),

#     dpi=144,

#     cmap='dark12',

#     title='Player of the Match',

#     title_size='',

#     bar_label_size=7,

#     tick_label_size=7,

#     shared_fontdict={'family' : 'Helvetica', 'color' : '.1'},

#     scale='linear',

#     writer=None,

#     fig=None,

#     bar_kwargs={'alpha': .7},)
bcr.bar_chart_race(

    df=bar_race_df.head(10),

    filename='horiz.mp4',

    orientation='h',

    sort='desc',

    n_bars=8,

    fixed_order=False,

    fixed_max=True,

    steps_per_period=10,

    interpolate_period=False,

    label_bars=True,

    bar_size=.95,

    period_label={'x': .99, 'y': .25, 'ha': 'right', 'va': 'center'},

    period_fmt='%B %d, %Y',

    period_summary_func=lambda v, r: {'x': .99, 'y': .18,

                                      's': f'Total Matches: {v.nlargest(6).sum():,.0f}',

                                      'ha': 'right', 'size': 8, 'family': 'Courier New'},

#     perpendicular_bar_func='median',

#     period_length=500,

    figsize=(5, 3),

    dpi=144,

    cmap='dark12',

    title='COVID-19 Deaths by Country',

    title_size='',

    bar_label_size=7,

    tick_label_size=7,

    shared_fontdict={'family' : 'Helvetica', 'color' : '.1'},

    scale='linear',

    writer=None,

    fig=None,

    bar_kwargs={'alpha': .7},)
!ls '/kaggle/working/horiz.mp4'
pwd
!'/kaggle/working/horiz.mp4'