import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



database = '../input/soccer/database.sqlite'

conn = sqlite3.connect(database)



tables = pd.read_sql("""SELECT *

                        FROM sqlite_master

                        WHERE type='table';""", conn)

tables
Player_Attributes = pd.read_sql("""SELECT * 

                        FROM Player_Attributes

                        """, conn)

Player = pd.read_sql("""SELECT * 

                        FROM Player

                        """, conn)

Match = pd.read_sql("""SELECT * 

                        FROM Match

                        """, conn)

League = pd.read_sql("""SELECT * 

                        FROM League

                        """, conn)

Country = pd.read_sql("""SELECT * 

                        FROM Country

                        """, conn)

Team = pd.read_sql("""SELECT * 

                        FROM Team

                        """, conn)

Team_Attributes = pd.read_sql("""SELECT * 

                        FROM Team_Attributes

                        """, conn)
print("Team")

print(Team.shape)

print(Team.columns)

print("-"*100)



print("League")

print(League.shape)

print(League.columns)

print("-"*100)



print("Country")

print(Country.shape)

print(Country.columns)

print("-"*100)



print("Player_Attributes")

print(Player_Attributes.shape)

print(Player_Attributes.columns)

print("-"*100)



print("Team_Attributes")

print(Team_Attributes.shape)

print(Team_Attributes.columns)

print("-"*100)



print("Match")

print(Match.shape)

print(Match.columns)

print("-"*100)



print("Player")

print(Player.shape)

print(Player.columns)
Player
Team
pd.set_option('display.max_columns', None)



Team_Attributes.head()
Real_Madrid_Statistics = pd.read_sql("""SELECT *

 

                                        FROM Team 



                                        left join Team_Attributes

                                        

                                        on Team_Attributes.team_api_id = Team.team_api_id

                                        

                                        where team_long_name= "Real Madrid CF"

                                        

                        

                        """, conn)

Real_Madrid_Statistics
plt.figure(figsize=(15, 7))



sns.lineplot(x="date",y="buildUpPlaySpeed",data=Real_Madrid_Statistics, label="buildUpPlaySpeed").set_title("Changes and improvments in Real Madrid style from 2010 to 2015", fontsize=18)

sns.lineplot(x="date",y="chanceCreationCrossing",data=Real_Madrid_Statistics, label="chanceCreationCrossing")

sns.lineplot(x="date",y="buildUpPlayPassing",data=Real_Madrid_Statistics, label="buildUpPlayPassing")

sns.lineplot(x="date",y="chanceCreationPassing",data=Real_Madrid_Statistics, label="chanceCreationPassing")

sns.lineplot(x="date",y="chanceCreationShooting",data=Real_Madrid_Statistics, label="chanceCreationShooting")

sns.lineplot(x="date",y="defencePressure",data=Real_Madrid_Statistics, label="defencePressure")

sns.lineplot(x="date",y="defenceAggression",data=Real_Madrid_Statistics, label="defenceAggression")

sns.lineplot(x="date",y="defenceTeamWidth",data=Real_Madrid_Statistics, label="defenceTeamWidth")







plt.tick_params(axis='x', rotation=90)

Matchs_results = pd.read_sql("""SELECT Match.id

                            ,Country.name  country_name

                            ,League.name  League_name

                            ,date

                            ,season

                            ,Home_team.team_long_name  Home_team

                            ,away_team.team_long_name  away_team

                            ,home_team_goal

                            ,away_team_goal



 

                        FROM Country 

                        

                        join Match

                        on Country.id = Match.Country_id

                        

                        join League

                        on Country.id = League.Country_id

                        

                        LEFT JOIN Team AS Home_team 

                        on Home_team.team_api_id = Match.home_team_api_id

                        

                        LEFT JOIN Team AS away_team 

                        on away_team.team_api_id = Match.away_team_api_id

                        

                        """, conn)



Matchs_results
home_RM = Matchs_results[Matchs_results["Home_team"] == "Real Madrid CF"]

away_RM = Matchs_results[Matchs_results["away_team"] == "Real Madrid CF"]

home_RM['match_result'] = np.where(home_RM['home_team_goal'] > home_RM['away_team_goal'], 'win', 'lose')

home_RM['match_result'] = np.where(home_RM['home_team_goal'] == home_RM['away_team_goal'], 'draw', home_RM['match_result'])



away_RM['match_result'] = np.where(away_RM['home_team_goal'] < away_RM['away_team_goal'], 'win', 'lose')

away_RM['match_result'] = np.where(away_RM['home_team_goal'] == away_RM['away_team_goal'], 'draw', away_RM['match_result'])





RM = pd.concat([home_RM, away_RM])

RM
plt.figure(figsize=(14, 5))



sns.countplot("match_result",data=RM ).set_title("Real Madrid match's results from 2008 to 2015", fontsize=18)

goals_scored = home_RM['home_team_goal'].sum() + away_RM['away_team_goal'].sum()

goals_conceded = home_RM["away_team_goal"].sum() + home_RM['home_team_goal'].sum()

print("Real Madrid goals scored from 2008 to 2015 :  " ,   goals_scored)

print("Real Madrid goals conceded from 2008 to 2015 :" , goals_conceded)
Country
co_ord = pd.read_csv("../input/world-coordinates/world_coordinates.csv")



co_ord
new_row_1 = {'Code':'ENG', 'Country':"England", 'latitude':52.3555, 'longitude':1.1743}

new_row_2 = {'Code':'SCT', 'Country':"Scotland", 'latitude':56.8642, 'longitude':-4.2026}



#new_row_1 = {'Code':'ENG', 'Country':"England", 'latitude':1.1743, 'longitude':52.3555}

#new_row_2 = {'Code':'SCT', 'Country':"Scotland", 'latitude':4.2026, 'longitude':56.4907}



co_ord = co_ord.append(new_row_1, ignore_index=True)

co_ord = co_ord.append(new_row_2, ignore_index=True)

maping = co_ord[co_ord["Country"].isin(Country["name"])]

maping



maping = maping.assign(League_name = ['Belgium Jupiler League'

                                   ,'Switzerland Super League'

                                   ,'Germany 1. Bundesliga'

                                   ,'Spain LIGA BBVA'

                                   ,'France Ligue 1'

                                   ,'Italy Serie A'

                                   ,'Netherlands Eredivisie'

                                   ,'Poland Ekstraklasa'

                                   ,'Portugal Liga ZON Sagres'

                                   ,'England Premier League'

                                   ,'Scotland Premier League'])



maping
import folium

incidents = folium.Map(location=[54.5260, 15.2551], zoom_start=3.4,tiles='Stamen Terrain')



## loop through the 100 crimes and add each to the incidents feature group

for lat, log,Country,League in zip( maping["latitude"],maping["longitude"],maping["Country"],maping["League_name"]):

    

    folium.CircleMarker(

            [lat, log],

            radius=15,

            popup = ('<strong>name</strong>: ' + str(League) + '<br>'

                     '<strong>Nationality</strong>: ' + str(Country).capitalize()), 

              # define how big you want the circle markers to be

            color='yellow',

            fill_color='yellow',

            fill_opacity=0.7

        ).add_to(incidents)

    



# add incidents to map

incidents
pd.set_option('display.max_columns', None)



Player_Attributes.head()
Player.head()
all_players = pd.read_sql("""SELECT player_name

                                    ,birthday

                                    ,date

                                    ,overall_rating

                                    ,height

                                    ,attacking_work_rate

                                    ,crossing

                                    ,finishing

                                    ,shot_power

                                    ,heading_accuracy

                                    ,sprint_speed,agility

                                    ,defensive_work_rate

                                    ,preferred_foot

                                    ,free_kick_accuracy

                                    ,penalties

                                    

                                    

                        FROM Player 

                        LEFT JOIN Player_Attributes

                        on Player.player_api_id = Player_Attributes.player_api_id

                                                

                        where date LIKE '2015%'

                       

                        ORDER by date

                        """, conn)







pd.set_option('display.max_columns', None)

all_players.head()
plt.figure(figsize=(14, 5))



sns.countplot("preferred_foot",data=all_players ).set_title("preferred foot count in 2015", fontsize=18)

print("Player_Attributes")

print(Player_Attributes.columns)

print("-"*100)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[15, 15])

plt.tick_params(axis='x', rotation=10)



fast = all_players.sort_values(by=['sprint_speed'], ascending=False)

fast = fast.drop_duplicates(subset='player_name', keep="first")

fast = fast.head()

sns.lineplot(x="player_name",y="sprint_speed",data=fast ,ax = axes[0,0]).set_title("top 5 fast players in 2015", fontsize=18)







highst = all_players.sort_values(by=['height'], ascending=False)

highst = highst.drop_duplicates(subset='player_name', keep="first")

highst = highst.head()

sns.lineplot(x="player_name",y="height",data=highst ,ax = axes[0,1]).set_title("top 5 tallest players in 2015", fontsize=18)







FK_best = all_players.sort_values(by=['free_kick_accuracy'], ascending=False)

FK_best = FK_best.drop_duplicates(subset='player_name', keep="first")

FK_best = FK_best.head()

sns.lineplot(x="player_name",y="free_kick_accuracy",data=FK_best ,ax = axes[1,0]).set_title("top 5 FK accuracy players in 2015", fontsize=18)







PK_best = all_players.sort_values(by=['penalties'], ascending=False)

PK_best = PK_best.drop_duplicates(subset='player_name', keep="first")

PK_best = PK_best.head()

sns.lineplot(x="player_name",y="free_kick_accuracy",data=PK_best ,ax = axes[1,1]).set_title("top 5 PK accuracy players in 2015", fontsize=18)



Cristiano = pd.read_sql("""SELECT player_name

                                    ,date

                                    ,overall_rating

                                    ,attacking_work_rate

                                    ,crossing

                                    ,finishing

                                    ,shot_power

                                    ,heading_accuracy

                                    ,free_kick_accuracy

                                    ,sprint_speed,agility

                                    

                        FROM Player 

                        LEFT JOIN Player_Attributes

                        on Player.player_api_id = Player_Attributes.player_api_id

                        

                        WHERE player_name = 'Cristiano Ronaldo'

                        

                       

                        ORDER by date

                        """, conn)







pd.set_option('display.max_columns', None)

Cristiano.head()
plt.figure(figsize=(15, 7))



sns.lineplot(Cristiano['date'], Cristiano["overall_rating"], palette = 'Wistia', label="overall_rating")

sns.lineplot(Cristiano['date'], Cristiano["free_kick_accuracy"], palette = 'Wistia', label="free_kick_accuracy")

sns.lineplot(Cristiano['date'], Cristiano["sprint_speed"], palette = 'Wistia', label="sprint_speed")

sns.lineplot(Cristiano['date'], Cristiano["agility"], palette = 'Wistia', label="agility")



plt.tick_params(axis='x', rotation=90)

plt.title("Cristiano Ronaldo from 2008 to 2015", fontsize=20)
Messi = pd.read_sql("""SELECT player_name

                                    ,date

                                    ,overall_rating

                                    ,attacking_work_rate

                                    ,crossing

                                    ,finishing

                                    ,shot_power

                                    ,heading_accuracy

                                    ,free_kick_accuracy

                                    ,sprint_speed,agility

                                    

                        FROM Player 

                        LEFT JOIN Player_Attributes

                        on Player.player_api_id = Player_Attributes.player_api_id

                        

                        WHERE player_name = 'Lionel Messi'

                        

                       

                        ORDER by  date

                        """, conn)







pd.set_option('display.max_columns', None)

Messi.head()
plt.figure(figsize=(15, 7))



sns.lineplot(Messi['date'], Messi["overall_rating"], palette = 'Wistia', label="overall_rating")

sns.lineplot(Messi['date'], Messi["free_kick_accuracy"], palette = 'Wistia', label="free_kick_accuracy")

sns.lineplot(Messi['date'], Messi["sprint_speed"], palette = 'Wistia', label="sprint_speed")

sns.lineplot(Messi['date'], Messi["agility"], palette = 'Wistia', label="agility")



plt.tick_params(axis='x', rotation=90)

plt.title("Lionel Messi from 2008 to 2015", fontsize=20)