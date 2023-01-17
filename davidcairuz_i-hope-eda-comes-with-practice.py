import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
features_dict = {

    'GameId': "a unique game identifier",

    'PlayId': "a unique play identifier",

    'Team': "home or away",

    'X': "player position along the long axis of the field. See figure below.",

    'Y': "player position along the short axis of the field. See figure below.",

    'S': "speed in yards/second",

    'A': "acceleration in yards/second^2",

    'Dis': "distance traveled from prior time point, in yards",

    'Orientation': "orientation of player (deg)",

    'Dir': "angle of player motion (deg)",

    'NflId': "a unique identifier of the player",

    'DisplayName': "player's name",

    'JerseyNumber': "jersey number",

    'Season': "year of the season",

    'YardLine': "the yard line of the line of scrimmage",

    'Quarter': "game quarter (1-5, 5 == overtime)",

    'GameClock': "time on the game clock",

    'PossessionTeam': "team with possession",

    'Down': "the down (1-4)",

    'Distance': "yards needed for a first down",

    'FieldPosition': "which side of the field the play is happening on",

    'HomeScoreBeforePlay': "home team score before play started",

    'VisitorScoreBeforePlay': "visitor team score before play started",

    'NflIdRusher': "the NflId of the rushing player",

    'OffenseFormation': "offense formation",

    'OffensePersonnel': "offensive team positional grouping",

    'DefendersInTheBox': "number of defenders lined up near the line of scrimmage, spanning the width of the offensive line",

    'DefensePersonnel': "defensive team positional grouping",

    'PlayDirection': "direction the play is headed",

    'TimeHandoff': "UTC time of the handoff",

    'TimeSnap': "UTC time of the snap",

    'Yards': "the yardage gained on the play (you are predicting this)",

    'PlayerHeight': "player height (ft-in)",

    'PlayerWeight': "player weight (lbs)",

    'PlayerBirthDate': "birth date (mm/dd/yyyy)",

    'PlayerCollegeName': "where the player attended college",

    'Position': "the player's position (the specific role on the field that they typically play)",

    'HomeTeamAbbr': "home team abbreviation",

    'VisitorTeamAbbr': "visitor team abbreviation",

    'Week': "week into the season",

    'Stadium': "stadium where the game is being played",

    'Location': "city where the game is being player",

    'StadiumType': "description of the stadium environment",

    'Turf': "description of the field surface",

    'GameWeather': "description of the game weather",

    'Temperature': "temperature (deg F)",

    'Humidity': "humidity",

    'WindSpeed': "wind speed in miles/hour",

    'WindDirection': "wind direction",

}



pd.DataFrame.from_dict(features_dict, orient='index', columns=['Description']).style.set_properties(**{'text-align': 'left'})
game_information = ['GameId', 'Team', 'Season', 'Week' ,'HomeTeamAbbr', 'VisitorTeamAbbr', 'Stadium', 'Location', 'StadiumType', 'Turf']

player_information = ['NflId', 'DisplayName', 'JerseyNumber', 'PlayerHeight', 'PlayerWeight', 'PlayerBirthDate', 'PlayerCollegeName', 'Position', 'NflIdRusher']

weather_information = ['GameWeather', 'Temperature', 'Humidity', 'WindSpeed', 'WindDirection']

formation_information = ['OffenseFormation', 'OffensePersonnel', 'DefendersInTheBox', 'DefensePersonnel']

stats_information = ['X', 'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir']

play_information = ['PlayId', 'Yards', 'Down', 'Quarter', 'YardLine', 'GameClock', 'PossessionTeam', 'Distance', 'TimeHandoff', 'TimeSnap', 'NflIdRusher', 'FieldPosition', 

                    'HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 'Yards', 'PlayDirection']
df.sample(1).T
df.describe().T
game_information_dict = {key:features_dict[key] for key in game_information}

pd.DataFrame.from_dict(game_information_dict, orient='index', columns=['Description']).style.set_properties(**{'text-align': 'left'})
df['StadiumType'].value_counts(normalize=True)
df.groupby('StadiumType')['Yards'].mean().sort_values(ascending=True).plot(kind='barh', figsize=(15, 10), title='Yards mean by Stadium Type')
def clean_stadium_type(txt):

    if pd.isna(txt):

        return np.nan

    txt=txt.lower()# lower case

    txt=txt.strip()# return a copy

    txt=txt.replace("outdoors","outdoor")

    txt=txt.replace("oudoor","outdoor")

    txt=txt.replace("ourdoor","outdoor")

    txt=txt.replace("outdor","outdoor")

    txt=txt.replace("outddors","outdoor")

    txt=txt.replace("outside","outdoor")

    txt=txt.replace("indoors","indoor")

    txt=txt.replace("retractable ","retr")

    return txt



def transform_stadium_type(txt):

    if pd.isna(txt):

        return np.nan

    if 'outdoor' in txt or 'open' in txt:

        return 1

    if 'indoor' in txt or 'closed' in txt:

        return 0

    

    return np.nan



df["StadiumType"] = df["StadiumType"].apply(clean_stadium_type)

df["StadiumType"] = df["StadiumType"].apply(transform_stadium_type)
df['StadiumType'].value_counts(normalize=True)
df.groupby('StadiumType')['Yards'].mean().sort_values(ascending=True).plot(kind='barh', figsize=(15, 5), title='Yards mean by Stadium Type')
plt.figure(figsize=(15, 10))

sns.boxplot(x=df['StadiumType'], y = df['Yards'], data=df, showfliers=False)
df['Turf'].value_counts(normalize=True)
df.groupby('Turf')['Yards'].mean().sort_values(ascending=True).plot(kind='barh', figsize=(15, 10), title='Yards mean by Turf')
turf_groups = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural', 'UBU Sports Speed S5-M':'Artificial', 

        'Artificial':'Artificial', 'DD GrassMaster':'Artificial', 'Natural Grass':'Natural', 

        'UBU Speed Series-S5-M':'Artificial', 'FieldTurf':'Artificial', 'FieldTurf 360':'Artificial', 'Natural grass':'Natural', 'grass':'Natural', 

        'Natural':'Natural', 'Artifical':'Artificial', 'FieldTurf360':'Artificial', 'Naturall Grass':'Natural', 'Field turf':'Artificial', 

        'SISGrass':'Artificial', 'Twenty-Four/Seven Turf':'Artificial', 'natural grass':'Natural'} 
df['Turf'] = df['Turf'].replace(turf_groups)

df['Turf'].value_counts(normalize=True)
df.groupby('Turf')['Yards'].mean().sort_values(ascending=True).plot(kind='barh', figsize=(15, 5), title='Yards mean by Turf')
df['Stadium'].value_counts(normalize=True)
df.groupby('Stadium')['Yards'].mean().sort_values(ascending=True).plot(kind='barh', figsize=(15, 13), title='Yards mean by Stadium')
play_information_dict = {key:features_dict[key] for key in play_information}

pd.DataFrame.from_dict(play_information_dict, orient='index', columns=['Description']).style.set_properties(**{'text-align': 'left'})
plt.figure(figsize=(15, 10))

sns.distplot(df['Yards'], kde=False).set_title("Yards gained distribution")

plt.axvline(df['Yards'].mean(), linewidth=4, color='r')
df['Yards'].value_counts()[:15]
plt.figure(figsize=(15,10))

sns.boxplot(x='Down', y='Yards', data=df, showfliers=False).set_title('Gained yards by down')
plt.figure(figsize=(15,10))

sns.boxplot(x='Quarter', y='Yards', data=df, showfliers=False).set_title('Gained yards by quarter')
plt.figure(figsize=(15,10))

sns.boxplot(x='Down', y='Distance', data=df, showfliers=False)