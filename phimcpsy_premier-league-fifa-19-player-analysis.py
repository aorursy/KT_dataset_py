# Get all the required libraries 

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

import matplotlib.cm as cm

import re

sns.set_style("darkgrid")

from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import PolynomialFeatures



# Reading the Input data

player_Data = pd.read_csv("../input/data.csv")
# Show first five results

player_Data.head()

# Show the columns

player_Data.columns



# Separate out the required columns

req_Cols = ["Name","Age","Nationality","Overall","Potential","Club","Value","Wage","Special","International Reputation",

            "Skill Moves","Work Rate","Position","Crossing","Finishing", "HeadingAccuracy", "ShortPassing", 

            "Volleys", "Dribbling","Curve", "FKAccuracy", "LongPassing", "BallControl", "Acceleration",

            "SprintSpeed", "Agility", "Reactions", "Balance", "ShotPower","Jumping", "Stamina", "Strength",

            "LongShots", "Aggression","Interceptions", "Positioning", "Vision", "Penalties", "Composure",

            "Marking", "StandingTackle", "SlidingTackle", "GKDiving", "GKHandling","GKKicking",

            "GKPositioning", "GKReflexes"]



# Create new dataset with the trimmed set of columns

player_Data_Trimmed = player_Data[req_Cols]

req_Cols_Simple = ["Name","Age","Nationality","Overall","Potential","Club","Value","Position"]

player_Data_Simple = player_Data[req_Cols_Simple]



# Manipulating the Potential and Overall values to get the possible growth of a player

player_Data_Simple["Growth"] = player_Data_Simple["Potential"] - player_Data_Simple["Overall"]



# Convert the value to a player to number and converting all values... 

# ...(including the values which appear in hundred thousands) to same unit - Millions

player_Data_Simple['Unit'] = player_Data_Simple['Value'].str[-1]

player_Data_Simple['Value (M)'] = np.where(player_Data_Simple['Unit'] == '0', 0, 

                                           player_Data_Simple['Value'].str[1:-1].replace(r'[a-zA-Z]',''))

player_Data_Simple['Value (M)'] = player_Data_Simple['Value (M)'].astype(float)

player_Data_Simple['Value (M)'] = np.where(player_Data_Simple['Unit'] == 'M', 

                                           player_Data_Simple['Value (M)'], 

                                           player_Data_Simple['Value (M)']/1000)

player_Data_Simple = player_Data_Simple.drop('Unit', 1)



# Creating a list of Premier League clubs

prem_Leg_Clubs = ["Arsenal","Bournemouth", "Brighton & Hove Albion", "Burnley", "Cardiff City", "Chelsea", "Crystal Palace",

                  "Everton", "Fulham", "Huddersfield Town", "Leicester City", "Liverpool", "Manchester City", 

                  "Manchester United", "Newcastle United", "Southampton", "Tottenham Hotspur", 

                  "Watford", "West Ham United", "Wolverhampton Wanderers"]



# We get all the players in the premier league into a single dataset

player_Data_Simple_Prem = player_Data_Simple[player_Data_Simple['Club'].isin(prem_Leg_Clubs)]



player_Data_Simple_Prem.isnull().sum()

player_Data_Simple_Prem = player_Data_Simple_Prem.dropna()
# Find the mean of the various parameters grouped across the teams

player_Data_Simple_Prem.groupby("Club").mean()



# Sort the data to view the teams which have the highest mean player overall ratings

player_Data_Simple_Prem.groupby("Club").mean().sort_values('Overall', ascending = False)



# Findings: 

# Top Four: Manchester United, Chelsea, Manchester City, Tottenham Hotspur

# Bottom Four: Fulham, Huddersfield Town, Wolverhampton Wanderers, Cardiff City



# Sort the data to view the teams which have highest mean value of players

player_Data_Simple_Prem.groupby("Club").mean().sort_values('Value (M)', ascending = False)



# Findings:

# Unsurprisingly, Manchester City takes the top spot with respect to the value of players in its roster

# Top four: Manchester City, Tottenham Hotspur, Chelsea, Manchester United

# Bottom four: Brighton & Hove Albion, Bournemouth, Huddersfield Town, Cardiff City



# A rudimentarly attempt at seeing a chart with the mean values of the different parameters spread across the teams

player_Data_Simple_Prem.groupby("Club").mean().plot(kind="bar")
# Function to get the best squad based on overall rating

def get_best_squad(position):

    player_Data_Simple_Prem_copy = player_Data_Simple_Prem.copy()

    # instantiate an array 

    store = []

    # Loop through the players based on the positions passed into the function

    for i in position:

        store.append([i,player_Data_Simple_Prem_copy.loc[[player_Data_Simple_Prem_copy[player_Data_Simple_Prem_copy['Position'] == i]['Overall'].idxmax()]]['Name'].to_string(index = False), 

                      player_Data_Simple_Prem_copy[player_Data_Simple_Prem_copy['Position'] == i]['Overall'].max(),player_Data_Simple_Prem_copy[player_Data_Simple_Prem_copy['Position'] == i]['Club']])

        player_Data_Simple_Prem_copy.drop(player_Data_Simple_Prem_copy[player_Data_Simple_Prem_copy['Position'] == i]['Overall'].idxmax(), inplace = True)

    # Return the store array

    return pd.DataFrame(np.array(store).reshape(11,4), columns = ['Position', 'Player', 'Overall','Club']).to_string(index = False)



# 4-3-3

squad_433 = ['GK', 'LB', 'CB', 'CB', 'RB', 'LM', 'CDM', 'RM', 'LW', 'ST', 'RW']

print ('4-3-3')

print (get_best_squad(squad_433))
df_p_Prem = player_Data_Simple_Prem.groupby(['Age'])['Potential'].mean()

df_o_Prem = player_Data_Simple_Prem.groupby(['Age'])['Overall'].mean()



df_summary_Prem = pd.concat([df_p_Prem, df_o_Prem], axis=1)



ax = df_summary_Prem.plot()

ax.set_ylabel('Rating')

ax.set_title('Average Rating by Age')
laLiga_Clubs = ["Atlético Madrid","FC Barcelona", "Real Madrid", "Athletic Club de Bilbao", "Real Betis", 

                "RC Celta", "RCD Espanyol","Real Sociedad", "Valencia CF", "Real Valladolid CF", 

                "Deportivo Alavés", "SD Eibar", "Rayo Vallecano", "Sevilla FC", 

                "Villarreal CF", "Levante UD", "Getafe CF","CD Leganés", "Girona FC", 

                "SD Huesca"]

bundes_Clubs = ["FC Bayern München","Borussia Dortmund", "Borussia Mönchengladbach", "SC Freiburg", 

                "Bayer 04 Leverkusen", "FC Schalke 04", "VfB Stuttgart",

                  "SV Werder Bremen", "Hertha BSC", "1. FSV Mainz 05", "1. FC Nürnberg", 

                "VfL Wolfsburg", "Hannover 96", "Eintracht Frankfurt", "TSG 1899 Hoffenheim", 

                "FC Augsburg", "Fortuna Düsseldorf", "RB Leipzig"]

ligue1_Clubs = ["FC Girondins de Bordeaux","En Avant de Guingamp", "LOSC Lille", "Olympique Lyonnais", 

                "AS Monaco", "Montpellier HSC", "FC Nantes","OGC Nice", "Paris Saint-Germain", 

                "Stade Rennais FC", "RC Strasbourg Alsace", "Stade Malherbe Caen", "Olympique de Marseille", 

                  "Nîmes Olympique", "Stade de Reims", "Angers SCO", "Toulouse Football Club", 

                  "Amiens SC", "AS Saint-Étienne", "Dijon FCO"]

serieA_Clubs = ["Atalanta","Inter", "Juventus", "Lazio", "Milan", "Napoli", "Parma",

                  "Roma", "Torino", "Udinese", "Bologna", "Chievo Verona", "Empoli", 

                  "Sampdoria", "Cagliari", "Fiorentina", "Genoa", 

                  "Frosinone", "Sassuolo", "SPAL"]



# We get all the players in the premier league into a single dataset

player_Data_Simple_LaLiga = player_Data_Simple[player_Data_Simple['Club'].isin(laLiga_Clubs)]

# We get all the players in the premier league into a single dataset

player_Data_Simple_Bundes = player_Data_Simple[player_Data_Simple['Club'].isin(bundes_Clubs)]

# We get all the players in the premier league into a single dataset

player_Data_Simple_Ligue1 = player_Data_Simple[player_Data_Simple['Club'].isin(ligue1_Clubs)]

# We get all the players in the premier league into a single dataset

player_Data_Simple_SerieA = player_Data_Simple[player_Data_Simple['Club'].isin(serieA_Clubs)]



# Checking the players with highest overall rating in each club. 

# It also helps to see a unique row for each club to ascertain that all clubs in the league are represented

player_Data_Simple_LaLiga.sort_values('Overall',ascending = False).groupby('Club').head(1)

player_Data_Simple_Bundes.sort_values('Overall',ascending = False).groupby('Club').head(1)

player_Data_Simple_Ligue1.sort_values('Overall',ascending = False).groupby('Club').head(1)

player_Data_Simple_SerieA.sort_values('Overall',ascending = False).groupby('Club').head(1)



# Bundesliga

df_o_Bundes = player_Data_Simple_Bundes.groupby(['Age'])['Overall'].mean()

# Laliga

df_o_Laliga = player_Data_Simple_LaLiga.groupby(['Age'])['Overall'].mean()

# SerieA

df_o_SerieA = player_Data_Simple_SerieA.groupby(['Age'])['Overall'].mean()

#Ligue1

df_o_Ligue1 = player_Data_Simple_Ligue1.groupby(['Age'])['Overall'].mean()



df_summary = pd.concat([df_o_Prem,df_o_Bundes,df_o_Laliga,df_o_SerieA,df_o_Ligue1], axis=1, 

                       keys=['Premier League','Bundesliga','La Liga','Serie A','Ligue 1'])

df_summary

ax = df_summary.plot()

ax.set_ylabel('Rating')

ax.set_title('Average Rating by Age')
# Function to get the best squad for a single team. 

# Takes in parameters of position of player, Club name and the measure of interest - Overall player rating/Potential

def get_best_squad(position, club = '*', measurement = 'Overall'):

    df_copy = player_Data_Simple_Prem.copy()

    df_copy = df_copy[df_copy['Club'] == club]

    store = []

    for i in position:

        store.append([df_copy.loc[[df_copy[df_copy['Position'].str.contains(i)][measurement].idxmax()]]

                      ['Position'].to_string(index = False),

                      df_copy.loc[[df_copy[df_copy['Position'].str.contains(i)][measurement].idxmax()]]

                      ['Name'].to_string(index = False), 

                      df_copy[df_copy['Position'].str.contains(i)][measurement].max(), 

                      float(df_copy.loc[[df_copy[df_copy['Position'].str.contains(i)][measurement].idxmax()]]

                            ['Value (M)'].to_string(index = False))])

        df_copy.drop(df_copy[df_copy['Position'].str.contains(i)][measurement].idxmax(), inplace = True)

    #return store

    return np.sum([x[2] for x in store]).round(1), pd.DataFrame(np.array(store).reshape(11,4), columns = ['Position', 'Player', measurement, 'Value (M)']).to_string(index = False), np.sum([x[3] for x in store]).round(1)



# Setting the constraints for the positions of the team

squad_352_adj = ['GK', 'B$', 'B$', 'B$', 'M$|W$|T$', 'M$|W$|T$', 'M$|W$|T$', 'M$|W$|T$', 'M$|W$|T$', 'W$|T$|M$', 'W$|T$|M$']



By_club = player_Data_Simple_Prem.groupby(['Club'])['Overall'].mean()



# Function to get the 

def get_summary(squad):

    OP = []

    # only get top 100 clubs for shorter run-time

    for i in By_club.sort_values(ascending = False).index[0:100]:

        # for overall rating

        O_temp_rating, _, _  = get_best_squad(squad, club = i, measurement = 'Overall')

        # for potential rating & corresponding value

        P_temp_rating, _, P_temp_value = get_best_squad(squad, club = i, measurement = 'Potential')

        OP.append([i, O_temp_rating, P_temp_rating, P_temp_value])

    return OP



OP_df = pd.DataFrame(np.array(get_summary(squad_352_adj)).reshape(-1,4), columns = ['Club', 'Overall', 'Potential', 'Value of highest Potential squad'])

OP_df.set_index('Club', inplace = True)

OP_df = OP_df.astype(float)



OP_df_rank = OP_df.sort_values('Overall')



# Visualising the results

# Function to produce a horizontal bar plot

def barplot(x_data, y_data, x_label="", y_label="", title=""):

    _, ax = plt.subplots()

    # Draw bars, position them in the center of the tick mark on the x-axis

    ax.barh(x_data, y_data, color = '#539caf', align = 'center')

    ax.set_ylabel(y_label)

    ax.set_xlabel(x_label)

    ax.set_title(title)



barplot(x_data = OP_df_rank.index,y_data = OP_df_rank['Overall'],x_label='OverallRating',y_label = 'Club',title = 'Premier League Team Rating')
