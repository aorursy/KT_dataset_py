#Imports
#Here we import the libraries that we will be using for data analysis in this project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import sqlite3
import pandasql as pdsql
from pandasql import sqldf
#Start with creating a connection to the sqlite database and see what tables we have
conn = sqlite3.connect("database.sqlite")

list_of_tables = pd.read_sql("""SELECT *
                        FROM sqlite_master
                        WHERE type='table';""", conn)
list_of_tables
Country_df = pd.read_sql_query("select * from Country;", conn)
Country_df
#Writing a function to summarize/quickly analyze each table

def summarize_table(dataframename):
    
#         """" This function summarizes the content of a table/dataframe.
#              Args:
#                  dataframename (Pandas dataframe or sqlite table): .
#              Returns:
#                  Dimension of dataframe.
#                  Number and label and data type for each column.
#                  Total number of rows.
#                  Number of unique rows.
#                  First row.
#                  Summary statistics including count, mean, std, min, and max.
#                  Number of missing (null) values in each column.
#          """"
         
    print("Dimension of "+str("dataframename")+"table is: {}".format(dataframename.shape))
    print(100*"*")
    print(dataframename.info())
    print(100*"*")
    print(dataframename.select_dtypes(exclude=['float64','int64']).describe())
    print(100*"*")
    print(dataframename.describe())
    print(100*"*")
    print("Missing values of table is:")
    print(dataframename.isnull().sum(axis=0))
summarize_table(Country_df)
League_df = pd.read_sql_query("select * from League;", conn)
League_df
summarize_table(League_df)
#Since there are no missing values, I am going to use an inner join

Coubtry_League = pd.read_sql("""SELECT League.name AS League_Name, Country.name AS Country_Name
                        FROM League
                        JOIN Country ON Country.id = League.country_id;""", conn)
Coubtry_League
Match_df = pd.read_sql_query("select * from Match;", conn)
Match_df.head()
list(Match_df)
summarize_table(Match_df)
Player_df = pd.read_sql_query("select * from Player;", conn)
Player_df.head()
summarize_table(Player_df)
Player_Attributes_df = pd.read_sql_query("select * from Player_Attributes;", conn)
Player_Attributes_df.head()
list(Player_Attributes_df)
summarize_table(Player_Attributes_df)
Team_df = pd.read_sql_query("select * from Team;", conn)
Team_df.head()
summarize_table(Team_df)
Team_Attributes_df = pd.read_sql_query("select * from Team_Attributes;", conn)
Team_Attributes_df.head()
list(Team_Attributes_df)
summarize_table(Team_Attributes_df)
sqlite_sequence_df = pd.read_sql_query("select * from sqlite_sequence;", conn)
sqlite_sequence_df
#Joining Match and League tables to get league, home and away team names, and season
Team_League_Season = pd.read_sql("""SELECT League.name AS League_Name, 
                                        HT.team_long_name AS Home_Team,
                                        AT.team_long_name AS Away_Team,
                                        Match.season AS Match_Season
                                    
                                FROM Match
                                JOIN League on League.id = Match.league_id
                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
                                GROUP BY League.name, HT.team_long_name, Match.season
                                ORDER BY League.name, HT.team_long_name, Match.season DESC
                                ;""", conn)

Team_League_Season.head()
Count_Team_League_Season = pd.read_sql("""SELECT League.name AS League_Name, 
                                        count(distinct HT.team_long_name) AS Number_of_Home_Teams,
                                        Match.season AS Match_Season
                                    
                                FROM Match
                                JOIN League on League.id = Match.league_id
                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
                                GROUP BY League.name,  Match.season
                                ORDER BY League.name, Match.season ASC
                                ;""", conn)

Count_Team_League_Season.dtypes
#Creating a numpy array for the x-axis of the plot
seasons = np.array(['2008/2009', '2009/2010', '2010/2011', '2011/2012', '2012/2013',
       '2013/2014', '2014/2015', '2015/2016'], dtype=object)
seasons
%matplotlib inline 

#set ggplot style
plt.style.use('ggplot')


# plot data
fig, ax = plt.subplots(figsize=(12,7))

# use unstack()
Count_Team_League_Season.groupby(['Match_Season','League_Name']).sum()['Number_of_Home_Teams'].unstack().plot(ax=ax)
ax.set_xlabel('Match Season')
ax.set_ylabel('Number of Teams in the League')
plt.xticks(range(len(seasons)),seasons)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Number of Teams in Each League During Each Season')
#Defining a function to determine win, lose, tie for home team
def home_match_result(row):
    
#          """" This function calculates the result of a match for home team.
#              Args:
#                  dataframename row(Pandas dataframe): .
#              Returns:
#                  a value representing the result (string).
#          """"
        
    if row['home_team_goal'] == row['away_team_goal']:
        val = 'Home_Tie'
    elif row['home_team_goal'] > row['away_team_goal']:
        val = 'Home_Win'
    else:
        val = 'Home_Loss'
    return val
#Defining a function to determine win, lose, tie for away team
def away_match_result(row):
    
#          """" This function calculates the result of a match for away team.
#              Args:
#                  dataframename row(Pandas dataframe): .
#              Returns:
#                  a value representing the result (string).
#          """"
            
    if row['home_team_goal'] == row['away_team_goal']:
        val = 'Away_Tie'
    elif row['home_team_goal'] < row['away_team_goal']:
        val = 'Away_Win'
    else:
        val = 'Away_Loss'
    return val
#Using the merging of Pandas dataframe instead of SQL query to join Match and League

Leagues_Matches = Match_df[Match_df.league_id.isin(League_df['id'])]
Leagues_Matches = Leagues_Matches[['id','league_id','home_team_api_id','away_team_api_id','home_team_goal','away_team_goal','season']]
Leagues_Matches["total_goals"] = Leagues_Matches['home_team_goal'] + Leagues_Matches['away_team_goal']
Leagues_Matches["home_result"] = Leagues_Matches.apply(home_match_result,axis = 1)
Leagues_Matches["away_result"] = Leagues_Matches.apply(away_match_result,axis = 1)
Leagues_Matches.dropna(inplace=True)
Leagues_Matches.head()

#Separating the leagues for plotting and further analysis
Leagues_Matches_with_Results = pd.merge(Leagues_Matches,League_df,left_on='league_id', right_on='id')
Leagues_Matches_with_Results = Leagues_Matches_with_Results.drop(['id_x','id_y','country_id'],axis = 1)

Belgium_Jupiler_League = Leagues_Matches_with_Results[Leagues_Matches_with_Results.name == "Belgium Jupiler League"]
England_Premier_League = Leagues_Matches_with_Results[Leagues_Matches_with_Results.name == "England Premier League"]
France_Ligue = Leagues_Matches_with_Results[Leagues_Matches_with_Results.name == "France Ligue 1"]
Germany_Bundesliga = Leagues_Matches_with_Results[Leagues_Matches_with_Results.name == "Germany 1. Bundesliga"]
Italy_Serie_A = Leagues_Matches_with_Results[Leagues_Matches_with_Results.name == "Italy Serie A"]
Netherlands_Eredivisie = Leagues_Matches_with_Results[Leagues_Matches_with_Results.name == "Netherlands Eredivisie"]
Poland_Ekstraklasa = Leagues_Matches_with_Results[Leagues_Matches_with_Results.name == "Poland Ekstraklasa"]
Portugal_Liga_ZON_Sagres = Leagues_Matches_with_Results[Leagues_Matches_with_Results.name == "Portugal Liga ZON Sagres"]
Scotland_Premier_League = Leagues_Matches_with_Results[Leagues_Matches_with_Results.name == "Scotland Premier League"]
Spain_LIGA_BBVA = Leagues_Matches_with_Results[Leagues_Matches_with_Results.name == "Spain LIGA BBVA"]
Switzerland_Super_League = Leagues_Matches_with_Results[Leagues_Matches_with_Results.name == "Switzerland Super League"]

a = Belgium_Jupiler_League.groupby('season')
b = England_Premier_League.groupby('season')
c = France_Ligue.groupby('season')
d = Germany_Bundesliga.groupby('season')
e = Italy_Serie_A.groupby('season')
f = Netherlands_Eredivisie.groupby('season')
g = Poland_Ekstraklasa.groupby('season')
h = Portugal_Liga_ZON_Sagres.groupby('season')
i = Scotland_Premier_League.groupby('season')
j = Spain_LIGA_BBVA.groupby('season')
k = Switzerland_Super_League.groupby('season')
# seasons
Leagues_Matches_with_Results.head()
#Plotting total goals scored each season
fig = plt.figure(figsize=(14, 10))
plt.title("Total goals of each league in each season")
plt.xticks(range(len(seasons)),seasons)
plt.style.use('ggplot')
plt.xlabel("Season")
plt.ylabel("Total Goals Each League Scored")
num_seasons = range(len(seasons))
plt.plot(num_seasons,a.total_goals.sum().values,label = "Belgium Jupiler League", marker = 'o')
plt.plot(num_seasons,b.total_goals.sum().values,label = "England Premier League", marker = 'o')
plt.plot(num_seasons,c.total_goals.sum().values,label = "France Ligue 1", marker = 'o')
plt.plot(num_seasons,d.total_goals.sum().values,label = "Germany 1. Bundesliga", marker = 'o')
plt.plot(num_seasons,e.total_goals.sum().values,label = "Italy Serie A", marker = 'o')
plt.plot(num_seasons,f.total_goals.sum().values,label = "Netherlands Eredivisie", marker = 'o')
plt.plot(num_seasons,g.total_goals.sum().values,label = "Poland Ekstraklasa", marker = 'o')
plt.plot(num_seasons,h.total_goals.sum().values,label = "Portugal Liga ZON Sagres", marker = 'o')
plt.plot(num_seasons,i.total_goals.sum().values,label = "Scotland Premier League", marker = 'o')
plt.plot(num_seasons,j.total_goals.sum().values,label = "Spain LIGA BBVA", marker = 'o')
plt.plot(num_seasons,k.total_goals.sum().values,label = "ISwitzerland Super League", marker = 'o')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#Plotting average goals scored each season
fig = plt.figure(figsize=(14, 10))
plt.title("Average goals of each league in each season")
plt.xticks(range(len(seasons)),seasons)
plt.style.use('ggplot')
plt.xlabel("Season")
plt.ylabel("Average Goals Each League Scored")
num_seasons = range(len(seasons))
plt.plot(num_seasons,a.total_goals.mean().values,label = "Belgium Jupiler League", marker = 'o')
plt.plot(num_seasons,b.total_goals.mean().values,label = "England Premier League", marker = 'o')
plt.plot(num_seasons,c.total_goals.mean().values,label = "France Ligue 1", marker = 'o')
plt.plot(num_seasons,d.total_goals.mean().values,label = "Germany 1. Bundesliga", marker = 'o')
plt.plot(num_seasons,e.total_goals.mean().values,label = "Italy Serie A", marker = 'o')
plt.plot(num_seasons,f.total_goals.mean().values,label = "Netherlands Eredivisie", marker = 'o')
plt.plot(num_seasons,g.total_goals.mean().values,label = "Poland Ekstraklasa", marker = 'o')
plt.plot(num_seasons,h.total_goals.mean().values,label = "Portugal Liga ZON Sagres", marker = 'o')
plt.plot(num_seasons,i.total_goals.mean().values,label = "Scotland Premier League", marker = 'o')
plt.plot(num_seasons,j.total_goals.mean().values,label = "Spain LIGA BBVA", marker = 'o')
plt.plot(num_seasons,k.total_goals.mean().values,label = "ISwitzerland Super League", marker = 'o')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#Plotting average home/away goals scored in each league each season
fig = plt.figure(figsize=(14, 10))
plt.title("Average Home/Away goals of each league in each season")
plt.xticks(range(len(seasons)),seasons)
plt.style.use('ggplot')
plt.xlabel("Season")
plt.ylabel("Average Home/Away Goals Scored in Each League Each Season")
num_seasons = range(len(seasons))
plt.plot(num_seasons,a.home_team_goal.mean().values / a.away_team_goal.mean().values,label = "Belgium Jupiler League", marker = 'o')
plt.plot(num_seasons,b.home_team_goal.mean().values / b.away_team_goal.mean().values,label = "England Premier League", marker = 'o')
plt.plot(num_seasons,c.home_team_goal.mean().values / c.away_team_goal.mean().values,label = "France Ligue 1", marker = 'o')
plt.plot(num_seasons,d.home_team_goal.mean().values / d.away_team_goal.mean().values,label = "Germany 1. Bundesliga", marker = 'o')
plt.plot(num_seasons,e.home_team_goal.mean().values / e.away_team_goal.mean().values,label = "Italy Serie A", marker = 'o')
plt.plot(num_seasons,f.home_team_goal.mean().values / f.away_team_goal.mean().values,label = "Netherlands Eredivisie", marker = 'o')
plt.plot(num_seasons,g.home_team_goal.mean().values / g.away_team_goal.mean().values,label = "Poland Ekstraklasa", marker = 'o')
plt.plot(num_seasons,h.home_team_goal.mean().values / h.away_team_goal.mean().values,label = "Portugal Liga ZON Sagres", marker = 'o')
plt.plot(num_seasons,i.home_team_goal.mean().values / i.away_team_goal.mean().values,label = "Scotland Premier League", marker = 'o')
plt.plot(num_seasons,j.home_team_goal.mean().values / j.away_team_goal.mean().values,label = "Spain LIGA BBVA", marker = 'o')
plt.plot(num_seasons,k.home_team_goal.mean().values / k.away_team_goal.mean().values,label = "ISwitzerland Super League", marker = 'o')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#Looking into a very similar metric (average away-home goal difference), now using SQL query
Home_vs_Away_Goal_Diff = pd.read_sql("""SELECT Country.name AS country_name, 
                                        League.name AS league_name, 
                                        season,
                                        count(distinct stage) AS number_of_stages,
                                        count(distinct HT.team_long_name) AS number_of_teams,
                                        avg(home_team_goal) AS avg_home_team_goals, 
                                        avg(away_team_goal) AS avg_away_team_goals, 
                                        avg(home_team_goal-away_team_goal) AS avg_goal_dif, 
                                        avg(home_team_goal+away_team_goal) AS avg_goals, 
                                        sum(home_team_goal+away_team_goal) AS total_goals                                       
                                FROM Match
                                JOIN Country on Country.id = Match.country_id
                                JOIN League on League.id = Match.league_id
                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
                                GROUP BY Country.name, League.name, season
                                ORDER BY Country.name, League.name, season DESC
                                ;""", conn)
Home_vs_Away_Goal_Diff.head()
#set ggplot style
plt.style.use('ggplot')


# plot data
fig, ax = plt.subplots(figsize=(12,7))

# use unstack()
Home_vs_Away_Goal_Diff.groupby(['season','league_name']).sum()['avg_goal_dif'].unstack().plot(ax=ax)
ax.set_xlabel('Match Season')
ax.set_ylabel('Average Home-Away goal difference in the League')
plt.xticks(range(len(seasons)),seasons)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Average Home-Away goal difference of each league in each season')
#Only looking at 'Spain', 'Germany', 'France', 'Italy', 'England'
Home_vs_Away_Goal_Diff_Top_Five = pd.read_sql("""SELECT Country.name AS country_name, 
                                        League.name AS league_name, 
                                        season,
                                        count(distinct stage) AS number_of_stages,
                                        count(distinct HT.team_long_name) AS number_of_teams,
                                        avg(home_team_goal) AS avg_home_team_goals, 
                                        avg(away_team_goal) AS avg_away_team_goals, 
                                        avg(home_team_goal-away_team_goal) AS avg_goal_dif, 
                                        avg(home_team_goal+away_team_goal) AS avg_goals, 
                                        sum(home_team_goal+away_team_goal) AS total_goals                                       
                                FROM Match
                                JOIN Country on Country.id = Match.country_id
                                JOIN League on League.id = Match.league_id
                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
                                WHERE country_name in ('Spain', 'Germany', 'France', 'Italy', 'England')
                                GROUP BY Country.name, League.name, season
                                HAVING count(distinct stage) > 10
                                ORDER BY Country.name, League.name, season DESC
                                ;""", conn)
Home_vs_Away_Goal_Diff_Top_Five.head()
#Only looking at 'Spain', 'Germany', 'France', 'Italy', 'England'
#set ggplot style
plt.style.use('ggplot')


# plot data
fig, ax = plt.subplots(figsize=(12,7))

# use unstack()
Home_vs_Away_Goal_Diff_Top_Five.groupby(['season','league_name']).sum()['avg_goal_dif'].unstack().plot(ax=ax)
ax.set_xlabel('Match Season')
ax.set_ylabel('Average Home-Away goal difference in the League')
plt.xticks(range(len(seasons)),seasons)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Average Home-Away goal difference of each league in each season')
#Merging home team and team dataframe to get teams and leagues
Home_Teams_df = Leagues_Matches_with_Results[['home_team_api_id', 'season', 'home_result', 'name']]
Home_Teams_Names_df = pd.merge(Home_Teams_df,Team_df,left_on='home_team_api_id', right_on='team_api_id')
Home_Teams_Names_df = Home_Teams_Names_df.drop(['id','home_team_api_id','team_fifa_api_id', 'team_short_name'],axis = 1)
Home_Teams_Names_df.columns = ['season', 'game_result', 'league','team_api_id', 'team_name']
Home_Teams_Names_df.head()
#Merging away team and team dataframe to get teams and leagues
Away_Teams_df = Leagues_Matches_with_Results[['away_team_api_id', 'season', 'away_result', 'name']]
Away_Teams_Names_df = pd.merge(Away_Teams_df,Team_df,left_on='away_team_api_id', right_on='team_api_id')
Away_Teams_Names_df = Away_Teams_Names_df.drop(['id','away_team_api_id','team_fifa_api_id', 'team_short_name'],axis = 1)
Away_Teams_Names_df.columns = ['season', 'game_result', 'league','team_api_id', 'team_name']
Away_Teams_Names_df.head()
#Combining home and away teams to get all teams and leagues
Teams_Results_Names = Home_Teams_Names_df.append(Away_Teams_Names_df)
Teams_Results_Names.head()
#Function that assigns a score to the performance of the team depending on match results

def team_performance(value):
        
#          """" This function calculates a performance score based on the match results. {"Home loss"=-2, "Home tie"=0,
#               "Home win=1, "Away loss"=-1, "Away tie"=1,"Away win=2, "}
#              Args:
#                  a single match result value (string) .
#              Returns:
#                  a value representing the performance score (integer).
#          """"
            
    if value == "Home_loss":
        performace = -2
    elif value == "Home_Tie":
        performance = 0
    elif value == "Home_Win":
        performance = 1
    elif value == "Away_Loss":
        performance = -1
    elif value == "Away_Tie":
        performance = 1
    else:
        performance = 2
    return performance
        
print (team_performance("Away_Win"))
#Now that the function is working, we will apply the function to the column "game_result"
Teams_Results_Names["Team_Performance"] = Teams_Results_Names['game_result'].apply(team_performance)
Teams_Results_Names.dropna(inplace=True)
Teams_Results_Names.head()
Team_Performance = """SELECT
        t.team_name AS Team_Name, t.season AS Season, sum(t.Team_Performance) AS Team_Performance, league as League
     FROM
        Teams_Results_Names t
     GROUP BY t.team_name, t.season
     ORDER BY Season, Team_Performance, Team_Name DESC   
           ;"""
Team_Performance_df = sqldf(Team_Performance)
Team_Performance_df.head()
Season_One_Top_Ten = """SELECT
            Team_Name, Season, Team_Performance, League
         FROM
            Team_Performance_df t
         WHERE t.Season = '2008/2009'
         ORDER BY Team_Performance DESC
         LIMIT 10
               ;"""
Season_One_Top_Ten_df = sqldf(Season_One_Top_Ten)
Season_Two_Top_Ten = """SELECT
            Team_Name, Season, Team_Performance, League
         FROM
            Team_Performance_df t
         WHERE t.Season = '2009/2010'
         ORDER BY Team_Performance DESC
         LIMIT 10
               ;"""
Season_Two_Top_Ten_df = sqldf(Season_Two_Top_Ten)
Season_Three_Top_Ten = """SELECT
            Team_Name, Season, Team_Performance, League
         FROM
            Team_Performance_df t
         WHERE t.Season = '2010/2011'
         ORDER BY Team_Performance DESC
         LIMIT 10
               ;"""
Season_Three_Top_Ten_df = sqldf(Season_Three_Top_Ten)
Season_Four_Top_Ten = """SELECT
            Team_Name, Season, Team_Performance, League
         FROM
            Team_Performance_df t
         WHERE t.Season = '2011/2012'
         ORDER BY Team_Performance DESC
         LIMIT 10
               ;"""
Season_Four_Top_Ten_df = sqldf(Season_Four_Top_Ten)
Season_Five_Top_Ten = """SELECT
            Team_Name, Season, Team_Performance, League
         FROM
            Team_Performance_df t
         WHERE t.Season = '2012/2013'
         ORDER BY Team_Performance DESC
         LIMIT 10
               ;"""
Season_Five_Top_Ten_df = sqldf(Season_Five_Top_Ten)
Season_Six_Top_Ten = """SELECT
            Team_Name, Season, Team_Performance, League
         FROM
            Team_Performance_df t
         WHERE t.Season = '2013/2014'
         ORDER BY Team_Performance DESC
         LIMIT 10
               ;"""
Season_Six_Top_Ten_df = sqldf(Season_Six_Top_Ten)
Season_Seven_Top_Ten = """SELECT
            Team_Name, Season, Team_Performance, League
         FROM
            Team_Performance_df t
         WHERE t.Season = '2014/2015'
         ORDER BY Team_Performance DESC
         LIMIT 10
               ;"""
Season_Seven_Top_Ten_df = sqldf(Season_Seven_Top_Ten)
Season_Eight_Top_Ten = """SELECT
            Team_Name, Season, Team_Performance, League
         FROM
            Team_Performance_df t
         WHERE t.Season = '2015/2016'
         ORDER BY Team_Performance DESC
         LIMIT 10
               ;"""
Season_Eight_Top_Ten_df = sqldf(Season_Eight_Top_Ten)
Season_Three_Top_Ten_df
###Plotting all eight seasons

###2008-2009
# Empty list to append later
grouped_list_One = []
label_list_One = []

# Iterating through each group key grouped by league
for label, key in Season_One_Top_Ten_df.groupby(['League'])['Team_Performance']:
    grouped_list_One.append(key)
    label_list_One.append(label)

# Concatenating the grouped list column-wise and filling Nans with 0's 
df_grouped_bar_One = pd.concat(grouped_list_One, axis=1).fillna(0)

# Renaming columns created to take on new names
df_grouped_bar_One.columns = label_list_One

# Bar plot with a chosen colormap
ax1 = df_grouped_bar_One.plot(kind='bar', stacked=True, figsize=(10,7), 
                         width=0.4, ylim=(0,60), cmap=plt.cm.rainbow)
# Figure Aesthetics
ax1.set_xticklabels(np.array((Season_One_Top_Ten_df[['Team_Name']].values)))
ax1.set_ylabel('Team Performance Score')
ax1.set_xlabel('Team Name')
ax1.set_title("Top 10 Teams Performance Score 2008/2009")
ax1.legend(label_list_One)
ax1.legend(label_list_One, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


###2009-2010
# Empty list to append later
grouped_list_Two = []
label_list_Two = []

# Iterating through each group key grouped by league
for label, key in Season_Two_Top_Ten_df.groupby(['League'])['Team_Performance']:
    grouped_list_Two.append(key)
    label_list_Two.append(label)

# Concatenating the grouped list column-wise and filling Nans with 0's 
df_grouped_bar_Two = pd.concat(grouped_list_Two, axis=1).fillna(0)

# Renaming columns created to take on new names
df_grouped_bar_Two.columns = label_list_Two

# Bar plot with a chosen colormap
ax2 = df_grouped_bar_Two.plot(kind='bar', stacked=True, figsize=(10,7), 
                         width=0.4, ylim=(0,60), cmap=plt.cm.rainbow)
# Figure Aesthetics
ax2.set_xticklabels(np.array((Season_Two_Top_Ten_df[['Team_Name']].values)))
ax2.set_ylabel('Team Performance Score')
ax2.set_xlabel('Team Name')
ax2.set_title("Top 10 Teams Performance Score 2009/2010")
ax2.legend(label_list_Two)
ax2.legend(label_list_Two, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


###2010-2011
# Empty list to append later
grouped_list_Three = []
label_list_Three = []

# Iterating through each group key grouped by league
for label, key in Season_Three_Top_Ten_df.groupby(['League'])['Team_Performance']:
    grouped_list_Three.append(key)
    label_list_Three.append(label)

# Concatenating the grouped list column-wise and filling Nans with 0's 
df_grouped_bar_Three = pd.concat(grouped_list_Three, axis=1).fillna(0)

# Renaming columns created to take on new names
df_grouped_bar_Three.columns = label_list_Three

# Bar plot with a chosen colormap
ax3 = df_grouped_bar_Three.plot(kind='bar', stacked=True, figsize=(10,7), 
                         width=0.4, ylim=(0,60), cmap=plt.cm.rainbow)
# Figure Aesthetics
ax3.set_xticklabels(np.array((Season_Three_Top_Ten_df[['Team_Name']].values)))
ax3.set_ylabel('Team Performance Score')
ax3.set_xlabel('Team Name')
ax3.set_title("Top 10 Teams Performance Score 2010/2011")
ax3.legend(label_list_Three)
ax3.legend(label_list_Three, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


###2011-2012
# Empty list to append later
grouped_list_Four = []
label_list_Four = []

# Iterating through each group key grouped by league
for label, key in Season_Four_Top_Ten_df.groupby(['League'])['Team_Performance']:
    grouped_list_Four.append(key)
    label_list_Four.append(label)

# Concatenating the grouped list column-wise and filling Nans with 0's 
df_grouped_bar_Four = pd.concat(grouped_list_Four, axis=1).fillna(0)

# Renaming columns created to take on new names
df_grouped_bar_Four.columns = label_list_Four

# Bar plot with a chosen colormap
ax4 = df_grouped_bar_Four.plot(kind='bar', stacked=True, figsize=(10,7), 
                         width=0.4, ylim=(0,60), cmap=plt.cm.rainbow)
# Figure Aesthetics
ax4.set_xticklabels(np.array((Season_Four_Top_Ten_df[['Team_Name']].values)))
ax4.set_ylabel('Team Performance Score')
ax4.set_xlabel('Team Name')
ax4.set_title("Top 10 Teams Performance Score 2011/2012")
ax4.legend(label_list_Four)
ax4.legend(label_list_Four, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


###2012-2013
# Empty list to append later
grouped_list_Five = []
label_list_Five = []

# Iterating through each group key grouped by league
for label, key in Season_Five_Top_Ten_df.groupby(['League'])['Team_Performance']:
    grouped_list_Five.append(key)
    label_list_Five.append(label)

# Concatenating the grouped list column-wise and filling Nans with 0's 
df_grouped_bar_Five = pd.concat(grouped_list_Five, axis=1).fillna(0)

# Renaming columns created to take on new names
df_grouped_bar_Five.columns = label_list_Five

# Bar plot with a chosen colormap
ax5 = df_grouped_bar_Five.plot(kind='bar', stacked=True, figsize=(10,7), 
                         width=0.4, ylim=(0,60), cmap=plt.cm.rainbow)
# Figure Aesthetics
ax5.set_xticklabels(np.array((Season_Five_Top_Ten_df[['Team_Name']].values)))
ax5.set_ylabel('Team Performance Score')
ax5.set_xlabel('Team Name')
ax5.set_title("Top 10 Teams Performance Score 2012/2013")
ax5.legend(label_list_Five)
ax5.legend(label_list_Five, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


###2013-2014
# Empty list to append later
grouped_list_Six = []
label_list_Six = []

# Iterating through each group key grouped by league
for label, key in Season_Six_Top_Ten_df.groupby(['League'])['Team_Performance']:
    grouped_list_Six.append(key)
    label_list_Six.append(label)

# Concatenating the grouped list column-wise and filling Nans with 0's 
df_grouped_bar_Six = pd.concat(grouped_list_Six, axis=1).fillna(0)

# Renaming columns created to take on new names
df_grouped_bar_Six.columns = label_list_Six

# Bar plot with a chosen colormap
ax6 = df_grouped_bar_Six.plot(kind='bar', stacked=True, figsize=(10,7), 
                         width=0.4, ylim=(0,60), cmap=plt.cm.rainbow)
# Figure Aesthetics
ax6.set_xticklabels(np.array((Season_Six_Top_Ten_df[['Team_Name']].values)))
ax6.set_ylabel('Team Performance Score')
ax6.set_xlabel('Team Name')
ax6.set_title("Top 10 Teams Performance Score 2013/2014")
ax6.legend(label_list_Six)
ax6.legend(label_list_Six, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

###2014-2015
# Empty list to append later
grouped_list_Seven = []
label_list_Seven = []

# Iterating through each group key grouped by league
for label, key in Season_Seven_Top_Ten_df.groupby(['League'])['Team_Performance']:
    grouped_list_Seven.append(key)
    label_list_Seven.append(label)

# Concatenating the grouped list column-wise and filling Nans with 0's 
df_grouped_bar_Seven = pd.concat(grouped_list_Seven, axis=1).fillna(0)

# Renaming columns created to take on new names
df_grouped_bar_Seven.columns = label_list_Seven

# Bar plot with a chosen colormap
ax7 = df_grouped_bar_Seven.plot(kind='bar', stacked=True, figsize=(10,7), 
                         width=0.4, ylim=(0,60), cmap=plt.cm.rainbow)
# Figure Aesthetics
ax7.set_xticklabels(np.array((Season_Seven_Top_Ten_df[['Team_Name']].values)))
ax7.set_ylabel('Team Performance Score')
ax7.set_xlabel('Team Name')
ax7.set_title("Top 10 Teams Performance Score 2014/2015")
ax7.legend(label_list_Seven)
ax7.legend(label_list_Seven, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


###2015-2016
# Empty list to append later
grouped_list_Eight = []
label_list_Eight = []

# Iterating through each group key grouped by league
for label, key in Season_Eight_Top_Ten_df.groupby(['League'])['Team_Performance']:
    grouped_list_Eight.append(key)
    label_list_Eight.append(label)

# Concatenating the grouped list column-wise and filling Nans with 0's 
df_grouped_bar_Eight = pd.concat(grouped_list_Eight, axis=1).fillna(0)

# Renaming columns created to take on new names
df_grouped_bar_Eight.columns = label_list_Eight

# Bar plot with a chosen colormap
ax8 = df_grouped_bar_Eight.plot(kind='bar', stacked=True, figsize=(10,7), 
                         width=0.4, ylim=(0,60), cmap=plt.cm.rainbow)
# Figure Aesthetics
ax8.set_xticklabels(np.array((Season_Eight_Top_Ten_df[['Team_Name']].values)))
ax8.set_ylabel('Team Performance Score')
ax8.set_xlabel('Team Name')
ax8.set_title("Top 10 Teams Performance Score 2014/2015")
ax8.legend(label_list_Eight)
ax8.legend(label_list_Eight, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
#Joining all dataframes with top ten teams from all eight seasons
Joined_Top_Tens_All_Seasons = [Season_One_Top_Ten_df, Season_Two_Top_Ten_df, Season_Three_Top_Ten_df,\
                              Season_Four_Top_Ten_df, Season_Five_Top_Ten_df, Season_Six_Top_Ten_df,\
                              Season_Seven_Top_Ten_df, Season_Eight_Top_Ten_df]
Joined_Top_Tens_All_Seasons_df = pd.concat(Joined_Top_Tens_All_Seasons)
Joined_Top_Tens_All_Seasons_df.head()
#With this query we get a table that gets the number of times each league had its teams in the top ten performers
League_Performance = """SELECT
            League, COUNT(League) AS Times_in_Top_Ten
         FROM
            Joined_Top_Tens_All_Seasons_df 
         GROUP BY League
         ORDER BY Times_in_Top_Ten DESC
               ;"""
League_Performance_df = sqldf(League_Performance)
League_Performance_df
#With this query we get a table that gets the number of times each league had its teams in the top ten performers
Player_Penalties = """SELECT
            p.player_name AS Name, SUM(penalties) AS Total_Penalties
         FROM
            Player_df p
         INNER JOIN Player_Attributes_df a
         ON p.player_api_id=a.player_api_id
         GROUP BY p.player_name
         ORDER BY Total_Penalties DESC
         LIMIT 10
               ;"""
Player_Penalties_df = sqldf(Player_Penalties)
Player_Penalties_df
Player_Attributes_df.head()
players_height = pd.read_sql("""SELECT CASE
                                        WHEN ROUND(height)<165 then 165
                                        WHEN ROUND(height)>195 then 195
                                        ELSE ROUND(height)
                                        END AS calc_height, 
                                        COUNT(height) AS distribution, 
                                        (avg(PA_Grouped.avg_overall_rating)) AS avg_overall_rating,
                                        (avg(PA_Grouped.avg_potential)) AS avg_potential,
                                        (avg(PA_Grouped.avg_vision)) AS avg_vision,
                                        (avg(PA_Grouped.avg_penalties)) AS avg_penalties,
                                        AVG(weight) AS avg_weight 
                            FROM PLAYER
                            LEFT JOIN (SELECT Player_Attributes.player_api_id, 
                                        avg(Player_Attributes.overall_rating) AS avg_overall_rating,
                                        avg(Player_Attributes.vision) AS avg_vision,
                                        avg(Player_Attributes.potential) AS avg_potential,
                                        avg(Player_Attributes.penalties) AS avg_penalties
                                        FROM Player_Attributes
                                        GROUP BY Player_Attributes.player_api_id) 
                                        AS PA_Grouped ON PLAYER.player_api_id = PA_Grouped.player_api_id
                            GROUP BY calc_height
                            ORDER BY calc_height
                                ;""", conn)
players_height
fig, ax1 = plt.subplots()
ax1.scatter(players_height['avg_weight'], players_height['avg_potential'])

ax1.set_xlabel("Average Weight", fontsize=15)
ax1.set_ylabel("Average Potential", fontsize=15)
ax1.set_title('Correlation between weight and potential')

ax1.grid(True)


fig, ax2 = plt.subplots()
ax2.scatter(players_height['calc_height'], players_height['avg_potential'])

ax2.set_xlabel("Average Height", fontsize=15)
ax2.set_ylabel("Average Potential", fontsize=15)
ax2.set_title('Correlation between height and potential')

ax2.grid(True)


fig, ax3 = plt.subplots()
ax3.scatter(players_height['avg_overall_rating'], players_height['avg_potential'])

ax3.set_xlabel("Average Rating", fontsize=15)
ax3.set_ylabel("Average Potential", fontsize=15)
ax3.set_title('Correlation between rating and potential')

ax3.grid(True)

fig, ax4 = plt.subplots()
ax4.scatter(players_height['avg_vision'], players_height['avg_potential'])

ax4.set_xlabel("Average Vision", fontsize=15)
ax4.set_ylabel("Average Potential", fontsize=15)
ax4.set_title('Correlation between vision and potential')

ax4.grid(True)

fig, ax5 = plt.subplots()
ax5.scatter(players_height['avg_vision'], players_height['avg_penalties'])

ax5.set_xlabel("Average Vision", fontsize=15)
ax5.set_ylabel("Average Penalties", fontsize=15)
ax5.set_title('Correlation between vision and penalties')

ax5.grid(True)


fig.tight_layout()

plt.show()

