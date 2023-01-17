import sqlite3

import pandas as pd

import missingno as msno

import matplotlib.pyplot as plt
database = '../input/database.sqlite'

conn = sqlite3.connect(database)
League = pd.read_sql_query("Select * from League",conn)

country = pd.read_sql_query("SELECT * FROM Country",conn)

Team = pd.read_sql_query("Select * from Team",conn)

Team_attributes = pd.read_sql_query("Select * from Team_Attributes",conn)

player = pd.read_sql_query("Select * from Player",conn)

Player_Attributes = pd.read_sql_query("Select * from Player_Attributes",conn)

Match = pd.read_sql_query("Select * from Match",conn)





country.info()

player.info()

Team_attributes.info()

League.info()

Player_Attributes.info()

Match.info()

Team.info()
ply_null =  pd.read_sql_query("SELECT COUNT (*) as Numero from Player WHERE id IS NULL OR player_api_id IS NULL OR player_name IS NULL OR player_fifa_api_id IS NULL OR birthday IS NULL OR height IS NULL OR weight IS NULL",conn)

print("la tabla Player tiene",ply_null.iloc[0]['Numero'],"valores null")

team_att_null =  pd.read_sql_query("SELECT COUNT (*) as Numero from Team_Attributes WHERE id IS NULL OR team_fifa_api_id IS NULL OR team_api_id IS NULL OR date IS NULL OR buildUpPlaySpeed IS NULL OR buildUpPlaySpeedClass IS NULL OR buildUpPlayDribbling IS NULL OR buildUpPlayDribblingClass IS NULL OR buildUpPlayPassing IS NULL OR buildUpPlayPassingClass IS NULL OR buildUpPlayPositioningClass IS NULL OR chanceCreationPassing IS NULL OR chanceCreationPassingClass IS NULL OR chanceCreationCrossing IS NULL OR chanceCreationCrossingClass IS NULL OR chanceCreationShooting IS NULL OR chanceCreationShootingClass IS NULL OR chanceCreationPositioningClass IS NULL OR defencePressure IS NULL OR defencePressureClass IS NULL OR defenceAggression IS NULL OR defenceAggressionClass IS NULL OR defenceTeamWidth IS NULL OR defenceTeamWidthClass IS NULL OR defenceDefenderLineClass IS NULL",conn)

print("la tabla Team_Atributes tiene",team_att_null.iloc[0]['Numero'],"valores null")

League_null =  pd.read_sql_query("SELECT COUNT (*) as Numero from League WHERE id IS NULL OR country_id IS NULL OR name IS NULL",conn)

print("la tabla League tiene",League_null.iloc[0]['Numero'],"valores null")

Ply_Att_null = pd.read_sql_query("SELECT COUNT (*) as Numero from Player_Attributes WHERE id IS NULL OR player_fifa_api_id IS NULL OR player_api_id IS NULL OR date IS NULL OR overall_rating IS NULL OR potential IS NULL OR preferred_foot IS NULL OR attacking_work_rate IS NULL OR defensive_work_rate IS NULL OR crossing IS NULL OR finishing IS NULL OR heading_accuracy IS NULL OR short_passing IS NULL OR volleys IS NULL OR dribbling IS NULL OR curve IS NULL OR free_kick_accuracy IS NULL OR long_passing IS NULL OR ball_control IS NULL OR acceleration IS NULL OR sprint_speed IS NULL OR agility IS NULL OR reactions IS NULL OR balance IS NULL OR shot_power IS NULL OR jumping IS NULL OR stamina IS NULL OR strength IS NULL OR long_shots IS NULL OR aggression IS NULL OR interceptions IS NULL OR positioning IS NULL OR vision IS NULL OR penalties IS NULL OR marking IS NULL OR standing_tackle IS NULL OR sliding_tackle IS NULL OR gk_diving IS NULL OR gk_handling IS NULL OR gk_kicking IS NULL OR gk_positioning IS NULL OR gk_reflexes IS NULL",conn)

print("la tabla Player_Attributes tiene",Ply_Att_null.iloc[0]['Numero'],"valores null")

Match_null = pd.read_sql_query("SELECT COUNT (*) as Numero from Match WHERE id IS NULL OR country_id IS NULL OR league_id IS NULL OR season IS NULL OR stage IS NULL OR date IS NULL OR match_api_id IS NULL OR home_team_api_id IS NULL OR away_team_api_id IS NULL OR home_team_goal IS NULL OR away_team_goal IS NULL OR home_player_X1 IS NULL OR home_player_X2 IS NULL OR home_player_X3 IS NULL OR home_player_X4 IS NULL OR home_player_X5 IS NULL OR home_player_X6 IS NULL OR home_player_X7 IS NULL OR home_player_X8 IS NULL OR home_player_X9 IS NULL OR home_player_X10 IS NULL OR home_player_X11 IS NULL OR away_player_X1 IS NULL OR away_player_X2 IS NULL OR away_player_X3 IS NULL OR away_player_X4 IS NULL OR away_player_X5 IS NULL OR away_player_X6 IS NULL OR away_player_X7 IS NULL OR away_player_X8 IS NULL OR away_player_X9 IS NULL OR away_player_X10 IS NULL OR away_player_X11 IS NULL OR home_player_Y1 IS NULL OR home_player_Y2 IS NULL OR home_player_Y3 IS NULL OR home_player_Y4 IS NULL OR home_player_Y5 IS NULL OR home_player_Y6 IS NULL OR home_player_Y7 IS NULL OR home_player_Y8 IS NULL OR home_player_Y9 IS NULL OR home_player_Y10 IS NULL OR home_player_Y11 IS NULL OR away_player_Y1 IS NULL OR away_player_Y2 IS NULL OR away_player_Y3 IS NULL OR away_player_Y4 IS NULL OR away_player_Y5 IS NULL OR away_player_Y6 IS NULL OR away_player_Y7 IS NULL OR away_player_Y8 IS NULL OR away_player_Y9 IS NULL OR away_player_Y10 IS NULL OR away_player_Y11 IS NULL OR home_player_1 IS NULL OR home_player_2 IS NULL OR home_player_3 IS NULL OR home_player_4 IS NULL OR home_player_5 IS NULL OR home_player_6 IS NULL OR home_player_7 IS NULL OR home_player_8 IS NULL OR home_player_9 IS NULL OR home_player_10 IS NULL OR home_player_11 IS NULL OR away_player_1 IS NULL OR away_player_2 IS NULL OR away_player_3 IS NULL OR away_player_4 IS NULL OR away_player_5 IS NULL OR away_player_6 IS NULL OR away_player_7 IS NULL OR away_player_8 IS NULL OR away_player_9 IS NULL OR away_player_10 IS NULL OR away_player_11 IS NULL OR goal IS NULL OR shoton IS NULL OR shotoff IS NULL OR foulcommit IS NULL OR card IS NULL OR cross IS NULL OR corner IS NULL OR possession IS NULL OR B365H IS NULL OR B365D IS NULL OR B365A IS NULL OR BWH IS NULL OR BWD IS NULL OR BWA IS NULL OR IWH IS NULL OR IWD IS NULL OR IWA IS NULL OR LBH IS NULL OR LBD IS NULL OR LBA IS NULL OR PSH IS NULL OR PSD IS NULL OR PSA IS NULL OR WHH IS NULL OR WHD IS NULL OR WHA IS NULL OR SJH IS NULL OR SJD IS NULL OR SJA IS NULL OR VCH IS NULL OR VCD IS NULL OR VCA IS NULL OR GBH IS NULL OR GBD IS NULL OR GBA IS NULL OR BSH IS NULL OR BSD IS NULL OR BSA IS NULL", conn)

print("la tabla Match tiene",Match_null.iloc[0]['Numero'],"valores null")

Team_null = pd.read_sql_query("SELECT COUNT (*) as Numero from Team WHERE id IS NULL OR team_api_id IS NULL OR team_fifa_api_id IS NULL OR team_long_name IS NULL OR team_short_name IS NULL", conn)

print("la tabla Team tiene",Team_null.iloc[0]['Numero'],"valores null")
msno.bar(Player_Attributes)
fk_m = Player_Attributes.free_kick_accuracy.mean()

fk_max = Player_Attributes.free_kick_accuracy.max()

print("Los jugadores tienen una exactitud media de: ",fk_m,"con un máximo de:",fk_max)

Player_Attributes.info()
Player_Attributes.head()
best_kickers = pd.read_sql_query("select   substr(a.date,1,4) as date ,b.player_name,avg(a.overall_rating+a.preferred_foot+a.finishing+a.free_kick_accuracy+a.vision+a.penalties) as media FROM Player_Attributes a inner join (SELECT a.player_api_id,p.player_name,avg(a.overall_rating+a.preferred_foot+a.finishing+a.free_kick_accuracy+a.vision+a.penalties) as media FROM Player_Attributes a INNER JOIN Player p 	on a.player_api_id=p.player_api_id group by  a.player_api_id,p.player_name order by avg(a.overall_rating+a.finishing+a.free_kick_accuracy+a.vision+a.penalties) desc limit 50) b on a.player_api_id=b.player_api_id Group by substr(a.date,1,4),b.player_name order by substr(a.date,1,4),avg(a.overall_rating+a.finishing+a.free_kick_accuracy+a.vision+a.penalties) desc;",conn)
plt.plot(best_kickers['date'].str[:4], best_kickers['media'], marker='o', linestyle='-', color='r', label = "Capacidades de lanzadores")

plt.title("Evaluación de los 50 mejores lanzadores por temporada") 

plt.xlabel("Fecha Evaluación") 

plt.ylabel("Media")

plt.legend()

plt.show()
best_kickers.groupby(['date'])['media'].mean()
best_kickers.groupby(['date'])['media'].max()
best_kickers.groupby(['date'])['media'].min()
best_kickers.groupby(['player_name'])['media'].mean().sort_values(ascending=False)
best_kickers_left = pd.read_sql_query("select   substr(a.date,1,4) as date ,b.player_name,avg(a.overall_rating+a.preferred_foot+a.finishing+a.free_kick_accuracy+a.vision+a.penalties) as media FROM Player_Attributes a inner join (SELECT a.player_api_id,p.player_name,avg(a.overall_rating+a.preferred_foot+a.finishing+a.free_kick_accuracy+a.vision+a.penalties) as media FROM Player_Attributes a INNER JOIN Player p on a.player_api_id=p.player_api_id where a.preferred_foot='left' group by  a.player_api_id,p.player_name order by avg(a.overall_rating+a.finishing+a.free_kick_accuracy+a.vision+a.penalties) desc limit 50) b on a.player_api_id=b.player_api_id Group by substr(a.date,1,4),b.player_name order by substr(a.date,1,4),avg(a.overall_rating+a.finishing+a.free_kick_accuracy+a.vision+a.penalties) desc;",conn)
plt.plot(best_kickers_left['date'].str[:4], best_kickers_left['media'], marker='o', linestyle='-', color='r', label = "Capacidades de lanzadores")

plt.title("Evaluación de los 50 mejores lanzadores Zurdos por temporada") 

plt.xlabel("Fecha Evaluación") 

plt.ylabel("Media")

plt.legend()

plt.show()
best_kickers_left.groupby(['player_name'])['media'].mean().sort_values(ascending=False)
best_kickers_right = pd.read_sql_query("select   substr(a.date,1,4) as date ,b.player_name,avg(a.overall_rating+a.preferred_foot+a.finishing+a.free_kick_accuracy+a.vision+a.penalties) as media FROM Player_Attributes a inner join (SELECT a.player_api_id,p.player_name,avg(a.overall_rating+a.preferred_foot+a.finishing+a.free_kick_accuracy+a.vision+a.penalties) as media FROM Player_Attributes a INNER JOIN Player p on a.player_api_id=p.player_api_id where a.preferred_foot='right' group by  a.player_api_id,p.player_name order by avg(a.overall_rating+a.finishing+a.free_kick_accuracy+a.vision+a.penalties) desc limit 50) b on a.player_api_id=b.player_api_id Group by substr(a.date,1,4),b.player_name order by substr(a.date,1,4),avg(a.overall_rating+a.finishing+a.free_kick_accuracy+a.vision+a.penalties) desc;",conn)
plt.plot(best_kickers_right['date'].str[:4], best_kickers_right['media'], marker='o', linestyle='-', color='r', label = "Capacidades de lanzadores")

plt.title("Evaluación de los 50 mejores lanzadores Zurdos por temporada") 

plt.xlabel("Fecha Evaluación") 

plt.ylabel("Media")

plt.legend()

plt.show()
best_kickers_right.groupby(['player_name'])['media'].mean().sort_values(ascending=False)