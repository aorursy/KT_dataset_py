#All required library

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style("darkgrid")

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data= pd.read_csv('/kaggle/input/fifa19/data.csv') # It show which dataset we use

data.head(10) # it gives 10 result 

# Show some statistics about dataset

data.describe()
# I check where there are NaN values

data.isnull().any()
data.columns # Shows that we have which columns
req_Cols = ["Name","Age","Nationality","Overall","Potential","Club","Value","Wage","Special","International Reputation",

            "Skill Moves","Work Rate","Position","Crossing","Finishing", "HeadingAccuracy", "ShortPassing", 

            "Volleys", "Dribbling","Curve", "FKAccuracy", "LongPassing", "BallControl", "Acceleration",

            "SprintSpeed", "Agility", "Reactions", "Balance", "ShotPower","Jumping", "Stamina", "Strength",

            "LongShots", "Aggression","Interceptions", "Positioning", "Vision", "Penalties", "Composure",

            "Marking", "StandingTackle", "SlidingTackle", "GKDiving", "GKHandling","GKKicking",

            "GKPositioning", "GKReflexes"]







# Create new dataset with the Trimmed set of columns

data_Trimmed = data[req_Cols]

req_Cols_Simple = ["Name","Age","Nationality","Overall","Potential","Club","Value","Position"]

data_Simple = data[req_Cols_Simple]





# We subtract from the potential to overall for get the possible growth of a player

data_Simple["Growth"] = data_Simple["Potential"] - data_Simple["Overall"]











# Convert the value to a player to number and converting all values...

data_Simple['Unit'] = data_Simple['Value'].str[-1]

data_Simple['Value (M)'] = np.where(data_Simple['Unit'] == '0', 0, 

                                           data_Simple['Value'].str[1:-1].replace(r'[a-zA-Z]',''))

data_Simple['Value (M)'] = data_Simple['Value (M)'].astype(float)

data_Simple['Value (M)'] = np.where(data_Simple['Unit'] == 'M', 

                                           data_Simple['Value (M)'], 

                                           data_Simple['Value (M)']/1000)

data_Simple = data_Simple.drop('Unit', 1)



# Creating a list of Serie A League clubs

SerieA_Leg_Clubs = ["Atalanta","Bologna,", "Cagliari","Chievo", "Empoli","Fiorentina","Frosinone","Genoa", "Inter", "Juventus",

                  "Lazio", "Milan", "Napoli", "Parma", "Roma", 

                  "Sampdoria", "Sassuolo", "Spal", "Torino", 

                  "Udinese", ]



# We take all the players in the Serie A in a dataset

data_Simple_SerieA = data_Simple[data_Simple['Club'].isin(SerieA_Leg_Clubs)]



data_Simple_SerieA.isnull().sum()

data_Simple_SerieA = data_Simple_SerieA.dropna()
# Correlation heatmap

plt.rcParams['figure.figsize']=(25,16)

hm=sns.heatmap(data[['Age', 'Overall', 'Potential', 'Value', 'Wage',

                'Acceleration', 'Aggression', 'Agility', 'Balance', 'BallControl', 

                'Body Type','Composure', 'Crossing','Dribbling', 'FKAccuracy', 'Finishing', 

                'HeadingAccuracy', 'Interceptions','International Reputation',

                'Joined', 'Jumping', 'LongPassing', 'LongShots',

                'Marking', 'Penalties', 'Position', 'Positioning',

                'ShortPassing', 'ShotPower', 'Skill Moves', 'SlidingTackle',

                'SprintSpeed', 'Stamina', 'StandingTackle', 'Strength', 'Vision',

                'Volleys']].corr(), annot = True, linewidths=.5, cmap='Reds')

hm.set_title(label='Korelasyon Isı haritası', fontsize=20)

hm;
#Line Plot

data.ShotPower.plot(kind = 'line', color = 'g', label = 'ShotPower', linewidth = 1, alpha = 0.5,grid = True,linestyle = ':')

data.Overall.plot(color = 'r',label = 'Overall', linewidth = 1, alpha = 0.5, grid = True, linestyle = '--')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Analysing of ShotPower and Overall')

plt.show()
# Histogram Plot

data.Potential.plot(kind='hist',bins = 30, figsize=(15,15))

plt.show()
data[['Age','Overall','Potential' ,'Finishing','Stamina']].hist(figsize=(15,10),bins=20,color='g',linewidth='1.5',edgecolor='k')

plt.tight_layout()

plt.show()
data_Simple_SerieA = data_Simple[data_Simple['Club'].isin(SerieA_Leg_Clubs)]



data_Simple_SerieA.isnull().sum()

data_Simple_SerieA = data_Simple_SerieA.dropna()



# Find the mean of Serie A teams

data_Simple_SerieA.groupby("Club").mean()



# Sort the data large to small

data_Simple_SerieA.groupby("Club").mean().sort_values('Overall', ascending = False)



# Findings: 

# Top Four: Juventus, Inter,Milan ,Napoli

# Bottom Four: Frosinone, Chievo , Empoli, Genoa,



# Sort the data to most value players

data_Simple_SerieA.groupby("Club").mean().sort_values('Value (M)', ascending = False)



# Findings:

# Unsurprisingly, Juventus takes the top spot with respect to the value of players.





# A rudimentarly attempt at seeing a chart with the mean values of the different parameters spread across the teams

data_Simple_SerieA.groupby("Club").mean().plot(kind="bar")
#we can define 'club' so that we can see about what we chose about information of team 

def club(x):

    return data[data['Club'] == x][['Name','Jersey Number','Position','Overall','Nationality','Age','Wage',

                                    'Value','Contract Valid Until']]



club('Juventus')
# Class Function to get the best squad based on overall rating

def get_best_squad(position):

    data_Simple_SerieA_copy = data_Simple_SerieA.copy() 

    store = []

    # Loop through the players based on the positions passed into the function

    for i in position:

        store.append([i,data_Simple_SerieA_copy.loc[[data_Simple_SerieA_copy[data_Simple_SerieA_copy['Position'] == i]['Overall'].idxmax()]]['Name'].to_string(index = False), 

                      data_Simple_SerieA_copy[data_Simple_SerieA_copy['Position'] == i]['Overall'].max(),data_Simple_SerieA_copy[data_Simple_SerieA_copy['Position'] == i]['Club']])

        data_Simple_SerieA_copy.drop(data_Simple_SerieA_copy[data_Simple_SerieA_copy['Position'] == i]['Overall'].idxmax(), inplace = True)

    # Return the store array

    return pd.DataFrame(np.array(store).reshape(11,4), columns = ['Position', 'Player', 'Overall','Club']).to_string(index = False)



# 4-2-3-1

squad_4231 = ['GK', 'LB', 'CB', 'CB', 'RB', 'CM', 'CM', 'CAM', 'LM', 'ST', 'RM']

print ('4-2-3-1')

print (get_best_squad(squad_4231))
player_features = (

    'Acceleration', 'Aggression', 'Agility', 

    'Balance', 'BallControl', 'Composure', 

    'Crossing', 'Dribbling', 'FKAccuracy', 

    'Finishing', 'GKDiving', 'GKHandling', 

    'GKKicking', 'GKPositioning', 'GKReflexes', 

    'HeadingAccuracy', 'Interceptions', 'Jumping', 

    'LongPassing', 'LongShots', 'Marking', 'Penalties'

)



# Top three features per position

for i, val in data.groupby(data['Position'])[player_features].mean().iterrows():

    print('Position {}: {}, {}, {}'.format(i, *tuple(val.nlargest(3).index)))
data_p_SerieA = data_Simple_SerieA.groupby(['Age'])['Potential'].mean()

data_o_SerieA = data_Simple_SerieA.groupby(['Age'])['Overall'].mean()



data_summary_SerieA = pd.concat([data_p_SerieA, data_o_SerieA], axis=1)



ax = data_summary_SerieA.plot()

ax.set_ylabel('Rating')

ax.set_title('Average Rating by Age')