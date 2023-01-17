# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing the required packages



import numpy as np 

import pandas as pd 



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")

import plotly.express as px

# Read the input files

play = pd.read_csv('../input/nfl-playing-surface-analytics/PlayList.csv')

injury = pd.read_csv('../input/nfl-playing-surface-analytics/InjuryRecord.csv')

player = pd.read_csv('../input/nfl-playing-surface-analytics/PlayerTrackData.csv')
play.head()
injury.head()
player.head()
unique_players = play.PlayerKey.nunique()

unique_games = play.GameID.nunique()

unique_plays = play.PlayKey.nunique()



print('There are {} players in the dataset.'.format(unique_players))

print('There are {} games in the dataset.'.format(unique_games))

print('There are {} plays in the dataset.'.format(unique_plays))
play.info()
# create a dataframe with game-level information

game = play[['GameID', 'StadiumType', 'FieldType', 'Weather', 'Temperature']].drop_duplicates().reset_index().drop(columns=['index'])

game.head()
a = play[['PlayKey','PlayType']].drop_duplicates().groupby('PlayType').count()['PlayKey'].sort_values()

a = pd.DataFrame({'PlayType':a.index, 'Count':a.values}).sort_values(by='Count',ascending=False)

a
ax = sns.barplot(x="Count",y="PlayType", data = a)
play.RosterPosition.unique()
play.StadiumType.unique()
play.FieldType.unique()
play.Temperature.unique()
play.Weather.unique()
play.PlayType.unique()
def add_value_labels(ax, spacing=5):

    # For each bar: Place a label

    for rect in ax.patches:

        # Get X and Y placement of label from rect.

        y_value = rect.get_height()

        x_value = rect.get_x() + rect.get_width() / 2



        # Number of points between bar and label. Change to your liking.

        space = spacing

        # Vertical alignment for positive values

        va = 'bottom'



        # If value of bar is negative: Place label below bar

        if y_value < 0:

            # Invert space to place label below

            space *= -1

            # Vertically align label at top

            va = 'top'



        # Use Y value as label and format number with one decimal place

        label = "{:.0f}".format(y_value)



        # Create annotation

        ax.annotate(

            label,                      # Use `label` as label

            (x_value, y_value),         # Place label at end of the bar

            xytext=(0, space),          # Vertically shift label by `space`

            textcoords="offset points", # Interpret `xytext` as offset in points

            ha='center',                # Horizontally center label

            va=va)                      # Vertically align label differently for

                                        # positive and negative values.

def visualize_game_features(play, rotation = 90, add_labels = False, figsize=(10,10)):

    #fig, axs = plt.subplots(ncols=2,nrows=3)

    plt.style.use('ggplot')

    fig = plt.figure(figsize=figsize)

    grid = plt.GridSpec(4, 3, hspace=0.5, wspace=0.5)

    roster_ax = fig.add_subplot(grid[1,0:])

    stadium_ax = fig.add_subplot(grid[2,:2])

    fieldtype_ax = fig.add_subplot(grid[2,2])

    weather_ax = fig.add_subplot(grid[3,1])

    #temperature_ax = fig.add_subplot(grid[2, 0:])

    #temperature_box_ax = fig.ad d_subplot(grid[3, 0:])



    roster_ax.bar(play.RosterPosition.value_counts().keys(), play.RosterPosition.value_counts().values,color='#00c2c7')

    roster_ax.set_title('RosterPosition')

    roster_ax.set_xticklabels(play.RosterPosition.value_counts().keys(),rotation=25)

    

    if add_labels:

        add_value_labels(roster_ax, spacing=5)

    

    stadium_ax.bar(play.StadiumType.value_counts().keys(), play.StadiumType.value_counts().values, color='#00c2c7')

    stadium_ax.set_title('StadiumType')

    stadium_ax.set_xticklabels(play.StadiumType.value_counts().keys(), rotation=rotation)

    

    if add_labels:

        add_value_labels(stadium_ax, spacing=5)



    fieldtype_ax.bar(play.FieldType.value_counts().keys(), play.FieldType.value_counts().values, color=['#00c2c7', '#ff9e15'])

    fieldtype_ax.set_title('FieldType')

    fieldtype_ax.set_xticklabels(play.FieldType.value_counts().keys(), rotation=0)

    

    if add_labels:

        add_value_labels(fieldtype_ax, spacing=5)



    weather_ax.bar(play.Weather.value_counts().keys(), play.Weather.value_counts().values, color='#00c2c7')

    weather_ax.set_title('Weather')

    weather_ax.set_xticklabels(play.Weather.value_counts().keys(), rotation=rotation)

    

    if add_labels:

        add_value_labels(weather_ax, spacing=5)

        

    temperature_ax.hist(play.Temperature.astype(int).values, bins=30, range=(0,90))

    temperature_ax.set_xlim(0,110)

    temperature_ax.set_xticks(range(0,110,10))

    temperature_ax.set_xticklabels(range(0,110,10))

    temperature_ax.set_title('Temperature')

    

    temperature_box_ax.boxplot(play.Temperature.astype(int).values, vert=False)

    temperature_box_ax.set_xlim(0,110)

    temperature_box_ax.set_xticks(range(0,110,10))

    temperature_box_ax.set_xticklabels(range(0,110,10))

    temperature_box_ax.set_yticklabels(['Temperature'])



    plt.suptitle('Game-Level Exploration', fontsize=16)

    plt.show()
def clean_weather(row):

    cloudy = ['Cloudy 50% change of rain', 'Hazy', 'Cloudy.', 'Overcast', 'Mostly Cloudy',

          'Cloudy, fog started developing in 2nd quarter', 'Partly Cloudy',

          'Mostly cloudy', 'Rain Chance 40%',' Partly cloudy', 'Party Cloudy',

          'Rain likely, temps in low 40s', 'Partly Clouidy', 'Cloudy, 50% change of rain','Mostly Coudy', '10% Chance of Rain',

          'Cloudy, chance of rain', '30% Chance of Rain', 'Cloudy, light snow accumulating 1-3"',

          'cloudy', 'Coudy', 'Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.',

         'Cloudy fog started developing in 2nd quarter', 'Cloudy light snow accumulating 1-3"',

         'Cloudywith periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.',

         'Cloudy 50% change of rain', 'Cloudy and cold',

       'Cloudy and Cool', 'Partly cloudy']

    

    clear = ['Clear, Windy',' Clear to Cloudy', 'Clear, highs to upper 80s',

             'Clear and clear','Partly sunny',

             'Clear, Windy', 'Clear skies', 'Sunny', 'Partly Sunny', 'Mostly Sunny', 'Clear Skies',

             'Sunny Skies', 'Partly clear', 'Fair', 'Sunny, highs to upper 80s', 'Sun & clouds', 'Mostly sunny','Sunny, Windy',

             'Mostly Sunny Skies', 'Clear and Sunny', 'Clear and sunny','Clear to Partly Cloudy', 'Clear Skies',

            'Clear and cold', 'Clear and warm', 'Clear and Cool', 'Sunny and cold', 'Sunny and warm', 'Sunny and clear']

    

    rainy = ['Rainy', 'Scattered Showers', 'Showers', 'Cloudy Rain', 'Light Rain', 'Rain shower', 'Rain likely, temps in low 40s.', 'Cloudy, Rain']

    

    snow = ['Heavy lake effect snow']

    

    indoor = ['Controlled Climate', 'Indoors', 'N/A Indoor', 'N/A (Indoors)']

        

    if row.Weather in cloudy:

        return 'Cloudy'

    

    if row.Weather in indoor:

        return 'Indoor'

    

    if row.Weather in clear:

        return 'Clear'

    

    if row.Weather in rainy:

        return 'Rain'

    

    if row.Weather in snow:

        return 'Snow'

      

    if row.Weather in ['Cloudy.', 'Heat Index 95', 'Cold']:

        return np.nan

    

    return row.Weather



def clean_stadiumtype(row):

    if row.StadiumType in ['Bowl', 'Heinz Field', 'Cloudy']:

        return np.nan

    else:

        return row.StadiumType



def clean_play_df(play_df):

    play_df_cleaned = play_df.copy()

    

    # clean StadiumType

    play_df_cleaned['StadiumType'] = play_df_cleaned['StadiumType'].str.replace(r'Oudoor|Outdoors|Ourdoor|Outddors|Outdor|Outside', 'Outdoor')

    play_df_cleaned['StadiumType'] = play_df_cleaned['StadiumType'].str.replace(r'Indoors|Indoor, Roof Closed|Indoor, Open Roof', 'Indoor')

    play_df_cleaned['StadiumType'] = play_df_cleaned['StadiumType'].str.replace(r'Closed Dome|Domed, closed|Domed, Open|Domed, open|Dome, closed|Domed', 'Dome')

    play_df_cleaned['StadiumType'] = play_df_cleaned['StadiumType'].str.replace(r'Retr. Roof-Closed|Outdoor Retr Roof-Open|Retr. Roof - Closed|Retr. Roof-Open|Retr. Roof - Open|Retr. Roof Closed', 'Retractable Roof')

    play_df_cleaned['StadiumType'] = play_df_cleaned.apply(lambda row: clean_stadiumtype(row), axis=1)

    

    # clean Weather

    play_df_cleaned['Weather'] = play_df_cleaned.apply(lambda row: clean_weather(row), axis=1)

    

    return play_df_cleaned
play_df_cleaned = clean_play_df(play)

game_df_cleaned = play_df_cleaned[['GameID', 'StadiumType', 'FieldType', 'Weather', 'Temperature','RosterPosition']].drop_duplicates().reset_index().drop(columns=['index'])

visualize_game_features(game_df_cleaned, rotation=45, add_labels = True, figsize=(12,16))