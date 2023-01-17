#Data manipulation

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Data visualisation

import matplotlib.pyplot as plt

import seaborn as sns



#Import the data

players = pd.read_csv('../input/data.csv')
#5 first lines of the dataframe

players.head()
#Some info about the features of the dataframe

players.info()
#A quick look to some interesting features

players.loc[:, ['ID', 'Name', 'Position', 'Overall', 'ST', 'CB']].head()
#Still info.. (for float and integer features)

players.describe()
#Check if there are good correlations between features among the dataset



plt.figure(figsize= (25, 16))

hm=sns.heatmap(players.loc[:,['Crossing','Finishing', 'HeadingAccuracy', 

                          'ShortPassing', 'Volleys', 'Dribbling', 'Curve',

                          'FKAccuracy', 'LongPassing', 'BallControl', 

                          'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 

                          'Balance', 'ShotPower', 'Jumping', 'Stamina', 

                          'Strength', 'LongShots', 'Aggression',

                          'Interceptions', 'Positioning', 'Vision', 

                          'Penalties', 'Composure', 'Marking', 'StandingTackle', 

                          'SlidingTackle', 'Overall']].corr(), annot = True, linewidths=.5, cmap='Reds')

hm.set_title(label='Heatmap of dataset', fontsize=20)

hm;
#We want to clean the notes given for a player for each position



position_attributes = ['RWB','LWB','RCB','CB', 'LCB', 'LB', 'RB','RM',  'RAM',

                       'RDM', 'RCM', 'LDM', 'LCM', 'CM', 'CDM', 'CAM','LM', 

                       'LAM', 'RW', 'RS', 'RF','LW', 'LS', 'LF','ST', 'CF']



def change_overall(position) :

    new_position = []

    for elt in position :

        if  isinstance(elt, float) :

            new_position.append(elt)

        else : 

            new_position.append(float(elt.split('+')[0])+float(elt.split('+')[1]))

    return new_position



#Rewrite the interesting columns

for elt in position_attributes :

    

    players[elt] = players[elt].fillna(0.0) #If the value for a given player & position is missing,

                                            #it means that the player (GK) cannot play at this position

                                            #We attribute to these positions the value 0 to fill our dataframes

    players[elt] = change_overall(players[elt])



    

#Check the dataframe  

players.loc[:, ['ID', 'Name', 'Position', 'Overall', 'ST', 'CB']].head()
#Top 5 strikers in FIFA19

players.loc[:,['Name','Overall','ST']].sort_values(by='ST', ascending=False).head()
#Top 5 left-wing backs

players.loc[:,['Name','Overall','LWB']].sort_values(by='LWB', ascending=False).head()
#Top characteristics for a centre-back



player_characteristics = ['Crossing','Finishing', 'HeadingAccuracy', 

                          'ShortPassing', 'Volleys', 'Dribbling', 'Curve',

                          'FKAccuracy', 'LongPassing', 'BallControl', 

                          'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 

                          'Balance', 'ShotPower', 'Jumping', 'Stamina', 

                          'Strength', 'LongShots', 'Aggression',

                          'Interceptions', 'Positioning', 'Vision', 

                          'Penalties', 'Composure', 'Marking', 'StandingTackle', 

                          'SlidingTackle']



corr_matrix = players.corr()



corr_matrix.loc[player_characteristics, 'CB'].sort_values(ascending=False).head()
#Top characteristics for a striker



corr_matrix.loc[player_characteristics, 'ST'].sort_values(ascending=False).head()
#For which position sprintspeed is the most important



corr_matrix.loc[position_attributes, 'SprintSpeed'].sort_values(ascending=False).head()
#We select the Ligue 1 clubs

Ligue1_Conforama_Clubs = ['Paris Saint-Germain', 

                          'Olympique de Marseille',

                          'Stade Malherbe Caen', 

                          'FC Nantes',

                          'Olympique Lyonnais',

                          'AS Saint-Étienne',

                          'AS Monaco',

                          'OGC Nice',

                          'Toulouse Football Club',

                          'Montpellier HSC',

                          'FC Girondins de Bordeaux',

                          'Stade Rennais FC',

                          'RC Strasbourg Alsace',

                          'Amiens SC',

                          'En Avant de Guingamp',

                          'LOSC Lille',

                          'Angers SCO',

                          'Dijon FCO',

                          'Nîmes Olympique',

                          'Stade de Reims',

                          ]

Ligue1_Conforama_Clubs.sort()



#We creae a new dataframe with only Ligue 1 players

df_ligue1 = players[players.Club.isin(Ligue1_Conforama_Clubs)].sort_values(by='Club')
#Number of players for each Ligue 1 Club

df_ligue1.groupby(by='Club')['ID'].count().sort_values(ascending=False)
#Check the values of the clubs



#The value of each player has to be converted to an integer

def value_to_int(df_value):

    try:

        value = float(df_value[1:-1])

        suffix = df_value[-1:]

        if suffix == 'M':

            value = value * 1000000

        elif suffix == 'K':

            value = value * 1000

    except ValueError:

        value = 0

    return value



players['Value_float'] = players['Value'].apply(value_to_int)

df_ligue1['Value_float'] = df_ligue1['Value'].apply(value_to_int)



# Comparison between values and overalls

plt.figure(figsize=(20,10))

value = players.Value_float

ax = sns.regplot(x = 'Overall', y = value / 10000000, fit_reg = False, data = players);

ax.set_title(label='Comparison between values and overalls');

plt.ylabel('Value (10M)')
#Cumulate value of the players for each Ligue 1 club

df_ligue1.groupby(by='Club')['Value_float'].sum().sort_values(ascending=False)
#We compare the Ligue 1 clubs and their players overall



plt.figure(figsize = (10,5))

ax = sns.boxplot(x=df_ligue1['Club'], y=df_ligue1['Overall'], palette='hls');

ax.set_title(label='Distribution Overall in Ligue 1 clubs', fontsize=20);

plt.xlabel('Club', fontsize=20)

plt.ylabel('Overall', fontsize=20)

plt.xticks(rotation = 90)

plt.ylim(45, 95)
#Explore the FC Nantes team



df_fcnantes = df_ligue1[df_ligue1.Club == 'FC Nantes']

df_fcnantes.loc[:, ['Name', 'Position', 'Club', 'Overall']].sort_values(by=['Overall', 'Name'], 

                                                                        ascending=[False, True])
#Positions in the team



plt.figure(figsize = (12,10))



plt.subplot(211)

sns.countplot(x = 'Position', data = df_fcnantes, palette = 'hls', order=position_attributes);

plt.title('Count of players by prefered position', fontsize=20);

plt.xlabel('Prefered position')



plt.subplot(212, label='Rating of FC Nantes players by position')

sns.boxplot(data=df_fcnantes[position_attributes], palette='hls', order=position_attributes);

plt.title('Rating of FC Nantes players by position', fontsize=20);

plt.xlabel('Position')

plt.ylabel('Rating')
#Looking for LWs and RWs that FC Nantes can recruit for 8M€



#We plot the 5 best LW that are less 8M value 

players.loc[:,['Name', 

               'Overall',

               'LW', 

               'RW', 

               'Value_float']][players.Value_float < 8000000].sort_values(by=['LW', 

                                                                              'Value_float'], ascending=[False, True]).head()
#Top characteristics for a left winger

corr_matrix.loc[player_characteristics, 'LW'].sort_values(ascending=False).head()
#Defining a LW Score

players['LW_score'] = players[player_characteristics].dot(corr_matrix.loc[player_characteristics, 'LW'])



players.loc[:,['Name', 

               'Overall',

               'LW', 

               'LW_score', 

               'Value_float']][players.Value_float < 8000000].sort_values(by=

                                                                  ['LW_score', 

                                                                   'Value_float'], ascending=[False, True]).head()
players.loc[:,['Name', 

               'Overall',

               'LW', 

               'LW_score',

               'Age',

               'Value_float', 

               'Potential']][players.Value_float < 8000000].sort_values(by=

                                                                  ['Age',

                                                                   'LW_score'], ascending=[True, False]).head()