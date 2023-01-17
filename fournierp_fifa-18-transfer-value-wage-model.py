import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

py.init_notebook_mode(connected=True)



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error

import xgboost as xgb



from itertools import product

import itertools

from collections import OrderedDict

import operator

import re

from math import sqrt

import pickle
original_df = pd.read_csv('../input/CompleteDataset.csv', error_bad_lines=False, low_memory=False)
processed_df = original_df.copy()

processed_df.head()
processed_df.info()
# Drop the colums that cannot be used

processed_df = processed_df.drop(['Photo', 'Flag', 'Club Logo', 'Special', 'ID'], 1)
# Convert the numerical values from string type to integer type. 

def str_to_int(string):

    try:

        integer = float(string)

        return integer

    except ValueError:

        return np.nan

    

processed_df['Acceleration'] = processed_df['Acceleration'].apply(str_to_int)

processed_df['Aggression'] = processed_df['Aggression'].apply(str_to_int)

processed_df['Agility'] = processed_df['Agility'].apply(str_to_int)

processed_df['Balance'] = processed_df['Balance'].apply(str_to_int)

processed_df['Ball control'] = processed_df['Ball control'].apply(str_to_int)

processed_df['Composure'] = processed_df['Composure'].apply(str_to_int)

processed_df['Crossing'] = processed_df['Crossing'].apply(str_to_int)

processed_df['Curve'] = processed_df['Curve'].apply(str_to_int)

processed_df['Dribbling'] = processed_df['Dribbling'].apply(str_to_int)

processed_df['Finishing'] = processed_df['Finishing'].apply(str_to_int)

processed_df['Free kick accuracy'] = processed_df['Free kick accuracy'].apply(str_to_int)

processed_df['GK diving'] = processed_df['GK diving'].apply(str_to_int)

processed_df['GK handling'] = processed_df['GK handling'].apply(str_to_int)

processed_df['GK kicking'] = processed_df['GK kicking'].apply(str_to_int)

processed_df['GK positioning'] = processed_df['GK positioning'].apply(str_to_int)

processed_df['GK reflexes'] = processed_df['GK reflexes'].apply(str_to_int)

processed_df['Heading accuracy'] = processed_df['Heading accuracy'].apply(str_to_int)

processed_df['Interceptions'] = processed_df['Interceptions'].apply(str_to_int)

processed_df['Jumping'] = processed_df['Jumping'].apply(str_to_int)

processed_df['Long passing'] = processed_df['Long passing'].apply(str_to_int)

processed_df['Long shots'] = processed_df['Long shots'].apply(str_to_int)

processed_df['Marking'] = processed_df['Marking'].apply(str_to_int)

processed_df['Penalties'] = processed_df['Penalties'].apply(str_to_int)

processed_df['Positioning'] = processed_df['Positioning'].apply(str_to_int)

processed_df['Reactions'] = processed_df['Reactions'].apply(str_to_int)

processed_df['Short passing'] = processed_df['Short passing'].apply(str_to_int)

processed_df['Shot power'] = processed_df['Shot power'].apply(str_to_int)

processed_df['Sliding tackle'] = processed_df['Sliding tackle'].apply(str_to_int)

processed_df['Sprint speed'] = processed_df['Sprint speed'].apply(str_to_int)

processed_df['Stamina'] = processed_df['Stamina'].apply(str_to_int)

processed_df['Standing tackle'] = processed_df['Standing tackle'].apply(str_to_int)

processed_df['Strength'] = processed_df['Strength'].apply(str_to_int)

processed_df['Vision'] = processed_df['Vision'].apply(str_to_int)

processed_df['Volleys'] = processed_df['Volleys'].apply(str_to_int)

processed_df['CAM'] = processed_df['CAM'].apply(str_to_int)

processed_df['CB'] = processed_df['CB'].apply(str_to_int)

processed_df['CDM'] = processed_df['CDM'].apply(str_to_int)

processed_df['CF'] = processed_df['CF'].apply(str_to_int)

processed_df['CM'] = processed_df['CM'].apply(str_to_int)

processed_df['LAM'] = processed_df['LAM'].apply(str_to_int)

processed_df['LB'] = processed_df['LB'].apply(str_to_int)

processed_df['LCM'] = processed_df['LCM'].apply(str_to_int)

processed_df['LDM'] = processed_df['LDM'].apply(str_to_int)

processed_df['LF'] = processed_df['LF'].apply(str_to_int)

processed_df['LM'] = processed_df['LM'].apply(str_to_int)

processed_df['LS'] = processed_df['LS'].apply(str_to_int)

processed_df['LW'] = processed_df['LW'].apply(str_to_int)

processed_df['LWB'] = processed_df['LWB'].apply(str_to_int)

processed_df['RAM'] = processed_df['RAM'].apply(str_to_int)

processed_df['RB'] = processed_df['RB'].apply(str_to_int)

processed_df['RCB'] = processed_df['RCB'].apply(str_to_int)

processed_df['RCM'] = processed_df['RCM'].apply(str_to_int)

processed_df['RDM'] = processed_df['RDM'].apply(str_to_int)

processed_df['RF'] = processed_df['RF'].apply(str_to_int)

processed_df['RM'] = processed_df['RM'].apply(str_to_int)

processed_df['RS'] = processed_df['RS'].apply(str_to_int)

processed_df['RW'] = processed_df['RW'].apply(str_to_int)

processed_df['RWB'] = processed_df['RWB'].apply(str_to_int)

processed_df['ST'] = processed_df['ST'].apply(str_to_int)
# Convert wages and values to floats

def str_to_float(string):

    if string[-1] == 'M':

        value = float(string[1:-1])*1000000

    elif string[-1] == 'K':

        value = float(string[1:-1])*1000

    else:

        value = float(string[1:])

    return value

        

processed_df['Value'] = processed_df['Value'].apply(str_to_float)

processed_df['Wage'] = processed_df['Wage'].apply(str_to_float)
# Number of player per country

nations = {'Nation': 0}

def count_nations(nation):

    #Count the players per country

    if (nation in nations):

        nations[nation] = nations[nation] + 1

    else:

        nations[nation] = 1



processed_df['Nationality'].apply(count_nations)

nations.pop('Nation', None)

# Sort and take the top 10%

sorted_nations = OrderedDict(sorted(nations.items(), key=lambda x: x[1], reverse=True))

sorted_nations = dict((k, sorted_nations[k]) for k in list(sorted_nations.keys())[0:int(len(sorted_nations)/10)])



fig = plt.figure(figsize=(16,4))

ax = fig.add_subplot(111)

ax.bar(range(len(sorted_nations)), list(sorted_nations.values()))

plt.xticks(range(len(sorted_nations)), list(sorted_nations.keys()), rotation=45,  ha='left')

plt.xlabel('Nations')

plt.ylabel('Number of players')

plt.show()
# Number of player per continent

continent_list = {

    'Africa' : ['Algeria','Angola','Benin','Botswana','Burkina','Burundi','Cameroon','Cape Verde','Central African Republic','Chad','Comoros','Congo','DR Congo','Djibouti','Egypt','Equatorial Guinea','Eritrea','Ethiopia','Gabon','Gambia','Ghana','Guinea','Guinea Bissau','Ivory Coast','Kenya','Lesotho','Liberia','Libya','Madagascar','Malawi','Mali','Mauritania','Mauritius','Morocco','Mozambique','Namibia','Niger','Nigeria','Rwanda','Sao Tome and Principe','Senegal','Seychelles','Sierra Leone','Somalia','South Africa','South Sudan','Sudan','Swaziland','Tanzania','Togo','Tunisia','Uganda','Zambia','Zimbabwe','Burkina Faso'],

    'Antarctica' : ['Fiji','Kiribati','Marshall Islands','Micronesia','Nauru','New Zealand','Palau','Papua New Guinea','Samoa','Solomon Islands','Tonga','Tuvalu','Vanuatu'],

    'Asia' : ['Afghanistan','Bahrain','Bangladesh','Bhutan','Brunei','Burma (Myanmar)','Cambodia','China','China PR','East Timor','India','Indonesia','Iran','Iraq','Israel','Japan','Jordan','Kazakhstan','North Korea','South Korea','Korea Republic','Korea DPR','Kuwait','Kyrgyzstan','Laos','Lebanon','Malaysia','Maldives','Mongolia','Nepal','Oman','Pakistan','Palestine','Philippines','Qatar','Russian Federation','Saudi Arabia','Singapore','Sri Lanka','Syria','Tajikistan','Thailand','Turkey','Turkmenistan','United Arab Emirates','Uzbekistan','Vietnam','Yemen','Russia'],

    'Australia Oceania' : ['Australia','New Caledonia'],

    'Europe' : ['Albania','Andorra','Armenia','Austria','Azerbaijan','Belarus','Belgium','Bosnia Herzegovina','Bulgaria','Croatia','Cyprus','Czech Republic','Denmark','Estonia','Finland','France','FYR Macedonia','Georgia','Germany','Greece','Hungary','Iceland','Ireland','Italy','Kosovo','Latvia','Liechtenstein','Lithuania','Luxembourg','Macedonia','Malta','Moldova','Monaco','Montenegro','Netherlands','Northern Ireland','Norway','Poland','Portugal','Romania','San Marino','Scotland','Serbia','Slovakia','Slovenia','Spain','Sweden','Switzerland','Ukraine','England','Vatican City','Republic of Ireland','Wales'],

    'North America' : ['Antigua and Barbuda','Bahamas','Barbados','Belize','Canada','Costa Rica','Cuba','Dominica','Dominican Republic','El Salvador','Grenada','Guatemala','Haiti','Honduras','Jamaica','Mexico','Nicaragua','Panama','Saint Kitts and Nevis','Saint Lucia','Saint Vincent and the Grenadines','Trinidad and Tobago','United States'],

    'South America' : ['Argentina','Bolivia','Brazil','Chile','Colombia','Curacao','Ecuador','Guyana','Paraguay','Peru','Suriname','Trinidad & Tobago','Uruguay','Venezuela']

}

continents = {'Africa' : 0, 'Antarctica' : 0,'Asia' : 0, 'Australia Oceania' : 0, 'Europe' : 0, 'North America' : 0, 'South America' : 0}



def count_continent(nation):

    # Count the players per continent

    for continent in continent_list:

        if nation in continent_list[continent]:

            continents[continent] = continents[continent] + 1

            return continent

    return np.NaN



processed_df['Nationality'].apply(count_continent)

sorted_continents = OrderedDict(sorted(continents.items(), key=lambda x: x[1], reverse=True))



fig = plt.figure(figsize=(10,4))

ax = fig.add_subplot(111)

ax.bar(range(len(sorted_continents)), list(sorted_continents.values()))

plt.xticks(range(len(sorted_continents)), list(sorted_continents.keys()), rotation = 45)

plt.xlabel('Continent')

plt.ylabel('Number of players')

plt.show()
club_list_per_league = {

    'England' : ['Manchester United', 'Chelsea', 'Manchester City', 'Arsenal', 'Tottenham Hotspur', 'Liverpool', 'Leicester City', 'Southampton', 'Everton', 'Swansea City', 'Stoke City', 'West Ham United', 'Watford', 'Bournemouth', 'Crystal Palace', 'West Bromwich Albion', 'Burnley', 'Newcastle United', 'Brighton & Hove Albion', 'Huddersfield Town'],

    'France' : ['Paris Saint-Germain', 'AS Monaco', 'AS Saint-Étienne', 'Olympique de Marseille', 'Olympique Lyonnais', 'OGC Nice', 'Girondins de Bordeaux', 'Stade Rennais FC', 'Montpellier Hérault SC', 'FC Nantes', 'En Avant de Guingamp', 'Amiens SC Football', 'Dijon FCO', 'Angers SCO', 'LOSC Lille', 'RC Strasbourg', 'SM Caen', 'Toulouse FC', 'ES Troyes AC', 'FC Metz'],

    'Germany' : ['FC Bayern Munich', 'Borussia Dortmund', 'Bayer 04 Leverkusen', 'FC Schalke 04', 'RB Leipzig', 'Borussia Mönchengladbach', 'TSG 1899 Hoffenheim', 'SV Werder Bremen', 'VfL Wolfsburg', 'Hertha BSC Berlin', 'Eintracht Frankfurt', 'VfB Stuttgart', 'Eintracht Braunschweig', 'FC Augsburg', '1. FSV Mainz 05', 'Hamburger SV', 'Hannover 96', 'SC Freiburg'],

    'Italy' : ['Juventus', 'Milan', 'Napoli', 'Inter', 'Roma', 'Torino', 'Lazio', 'Genoa', 'Sassuolo', 'Sampdoria', 'Fiorentina', 'Atalanta', 'Sampdoria', 'Bologna', 'Chievo Verona', 'Udinese', 'Cagliari', 'Ferrara (SPAL)', 'Crotone', 'Hellas Verona', 'Benevento Calcio'],

    'Spain' : ['Real Madrid CF', 'FC Barcelona', 'Atletico Madrid', 'Athletic Club de Bilbao','1. FC Köln', 'Villarreal CF', 'Real Sociedad', 'UD Las Palmas', 'RC Celta de Vigo', 'RCD Espanyol', 'SD Eibar', 'Valencia CF', 'Sevilla FC', 'RC Deportivo de La Coruña', 'Real Betis Balompié', 'Getafe CF', 'Girona CF', 'Deportivo Alav√©s', 'Levante UD', 'Deportivo Alavés', 'Málaga CF'],

}



two_columns = processed_df[['Nationality', 'Club']]



nationality_proportion = {'England' : [0, 0], 'France' : [0, 0], 'Germany' : [0, 0], 'Italy' : [0, 0], 'Spain' : [0, 0]}

def count_foreign_players():

    index = 0

    while (index < len(two_columns)):

        for country in club_list_per_league:

            if two_columns['Club'][index] in club_list_per_league[country]:

                if two_columns['Nationality'][index] == country:

                    nationality_proportion[country][0] = nationality_proportion[country][0] + 1

                else:

                    nationality_proportion[country][1] = nationality_proportion[country][1] + 1

        index += 1



count_foreign_players()

N = 5

home = (nationality_proportion['England'][0], nationality_proportion['France'][0], nationality_proportion['Germany'][0], nationality_proportion['Italy'][0], nationality_proportion['Spain'][0])

foreign = (nationality_proportion['England'][1], nationality_proportion['France'][1], nationality_proportion['Germany'][1], nationality_proportion['Italy'][1], nationality_proportion['Spain'][1])

ind = np.arange(N)    # the x locations for the groups

width = 0.35       # the width of the bars: can also be len(x) sequence



fig = plt.figure(figsize=(6,4))

ax = fig.add_subplot(111)

p1 = ax.bar(ind, home, width)

p2 = ax.bar(ind, foreign, width, bottom = home)

plt.ylabel('Number of Players')

plt.title('Proportion of foreign player per league')

plt.xticks(ind, ('England', 'France', 'Germany', 'Italy', 'Spain'))

plt.yticks(np.arange(0, 700, 100))

plt.legend((p1[0], p2[0]), ('Home', 'Foreign'))

plt.show()
# Maximum player value per country

# Selecting players with age smaller then 40 

dataset = processed_df.loc[processed_df['Age'] <= 40]

country_value = dataset.groupby("Nationality")["Value"].max()

country_value = country_value.reset_index()



data = [ dict(

        type = 'choropleth',

        locationmode = "country names",

        locations = country_value['Nationality'],

        z = country_value['Value'],

        autocolorscale = False,

        reversescale = True,

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) ),

        colorbar = dict(

            autotick = False,

            title = 'Maximum<br>Player Value'),

      ) ]



layout = dict(

    title = 'Maximum Player Value Per Country',

    geo = dict(

        showframe = False,

        showcoastlines = False,

        projection = dict(

            type = 'Mercator'

        )

    )

)



fig = dict( data=data, layout=layout )

py.iplot( fig, validate=False )
# Maximim player wage per country

country_wage = dataset.groupby("Nationality")["Wage"].max()

country_wage = country_wage.reset_index()



data = [ dict(

        type = 'choropleth',

        locationmode = "country names",

        locations = country_wage['Nationality'],

        z = country_wage['Wage'],

        autocolorscale = False,

        reversescale = True,

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) ),

        colorbar = dict(

            autotick = False,

            title = 'Maximum<br>Player Value'),

      ) ]



layout = dict(

    title = 'Maximum Player Wage Per Country',

    geo = dict(

        showframe = False,

        showcoastlines = False,

        projection = dict(

            type = 'Mercator'

        )

    )

)



fig = dict( data=data, layout=layout )

py.iplot( fig, validate=False )
# Player value per league 

value_per_league = {'League': 0}

league_value = {'England' : 0, 'France' : 0, 'Germany' : 0, 'Italy' : 0, 'Spain' : 0}



def sum_values(series):

    value = series[1]



    # Count the players value per league

    for league in club_list_per_league:

        if series[0] in club_list_per_league[league]:

            league_value[league] += value

            return league

    return np.NaN



club_value = processed_df[['Club', 'Value']]

club_value.apply(sum_values, 1)

value_per_league.pop('League', None)



# Average the players value per league

for league in league_value:

    league_value[league] = league_value[league] / (float(nationality_proportion[league][0])+float(nationality_proportion[league][1]))



sorted_league_value = OrderedDict(sorted(league_value.items(), key=lambda x: x[1], reverse=True))



fig = plt.figure(figsize=(6,4))

ax = fig.add_subplot(111)

ax.bar(range(len(sorted_league_value)), list(sorted_league_value.values()))

plt.xticks(range(len(sorted_league_value)), list(sorted_league_value.keys()))

plt.xlabel('League')

plt.ylabel('Avergage value of players')

ax.set(ylim=(0.5e7, 1e7))

plt.show()
# Player value per position

position_list = {

    'Forward' : ['CF', 'LF', 'LS', 'LW', 'RF', 'RS', 'RW', 'ST'],

    'Midfield' : ['CAM', 'CDM', 'CM', 'LAM', 'LCM', 'LDM', 'LM', 'RAM', 'RCB', 'RCM', 'RDM', 'RM'],

    'Backward' : ['CB', 'LB', 'LCB', 'LWB', 'RB', 'RCB', 'RWB'],

    'Goalkeeper' : ['GK'],

}

sum_per_position = {'Forward': 0, 'Midfield': 0, 'Backward': 0, 'Goalkeeper': 0}

player_per_position = {'Forward': 0, 'Midfield': 0, 'Backward': 0, 'Goalkeeper': 0}

average_per_position = {'Forward': 0, 'Midfield': 0, 'Backward': 0, 'Goalkeeper': 0}



def sum_values(series):

    already_counted = [0, 0, 0, 0] # Array to avoid having repeated values in the average

    preferred_position_list = re.sub("[^\w]", " ",  series[0]).split() #Split the preferred positions

    value = series[1]



    #Count the players per position category

    index = 0

    for preferred_position in preferred_position_list:

        for position in position_list:

            if preferred_position in position_list[position]:

                if(not already_counted[index]):

                    sum_per_position[position] += value

                    player_per_position[position] += 1                    

                    already_counted[index] = 1

                    

position_value = processed_df[['Preferred Positions', 'Value']]

position_value.apply(sum_values, 1)



# Average the players value per position category

for position in sum_per_position:

    average_per_position[position] = sum_per_position[position] / player_per_position[position]



fig = plt.figure(figsize=(6,4))

ax = fig.add_subplot(111)

ax.bar(range(len(average_per_position)), list(average_per_position.values()))

plt.xticks(range(len(average_per_position)), list(average_per_position.keys()))

plt.xlabel('Position')

plt.ylabel('Average value of players')

plt.show()
# Versatility and value

skill_versatility = {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0]}

average = {1: 0, 2: 0, 3: 0, 4: 0}



def sum_values(series):

    preferred_position_list = re.sub("[^\w]", " ",  series[0]).split() # Split the preferred positions

    skill_count = len(preferred_position_list)

    value = series[1]



    # Count the players per versatility

    if (skill_count in skill_versatility):

        skill_versatility[skill_count][0] += 1

        skill_versatility[skill_count][1] += value  

                    

position_value = processed_df[['Preferred Positions', 'Value']]

position_value.apply(sum_values, 1)



# Average the players value per position count

for count in skill_versatility:

    average[count] = skill_versatility[count][1] / skill_versatility[count][0]



fig = plt.figure(figsize=(6,4))

ax = fig.add_subplot(111)

ax.bar(range(len(average)), list(average.values()))

plt.xticks(range(len(average)), list(average.keys()))

plt.xlabel('Number of possible positions')

plt.ylabel('Average value of players')

ax.set(ylim=(2000000, 3500000))

plt.show()
# Player value per skill

fig = plt.figure(figsize=(6,4))

sns.set(style="ticks")

test = pd.DataFrame(np.c_[processed_df['Value'], processed_df['Acceleration'], processed_df['Aggression'], processed_df['Agility'], processed_df['Balance'],

                          processed_df['Ball control'], processed_df['Composure'],processed_df["Crossing"],processed_df['Curve'], processed_df['Dribbling'],

                          processed_df['Finishing'], processed_df['Free kick accuracy'], processed_df['GK diving'], processed_df['GK handling'], processed_df['GK kicking'],

                          processed_df['GK positioning'], processed_df['GK reflexes'], processed_df['Heading accuracy'], processed_df['Interceptions'], processed_df['Jumping'],

                          processed_df['Long passing'], processed_df['Long shots'], processed_df['Marking'], processed_df['Penalties'], processed_df['Positioning'],

                          processed_df['Reactions'], processed_df['Short passing'], processed_df['Shot power'], processed_df['Sliding tackle'], processed_df['Sprint speed'], 

                          processed_df['Stamina'], processed_df['Standing tackle'], processed_df['Strength'], processed_df['Vision'], processed_df['Volleys'] ], 

                    columns=["Value", "Acceleration", "Aggression", "Agility", "Balance", "Ball control", "Composure", "Crossing", "Curve", "Dribbling", "Finishing", "Free kick accuracy", "GK diving",

                             "GK handling", "GK kicking", "GK positioning", "GK reflexes", "Heading accuracy", "Interceptions", "Jumping", "Long passing", "Long shots", "Marking", "Penalties", "Positioning",

                             "Reactions", "Short passing", "Shot power", "Sliding tackle", "Sprint speed", "Stamina", "Standing tackle", "Strength", "Vision","Volleys"])

test = test.melt('Value', var_name='Skills',  value_name='Score')

g = sns.lmplot(y="Value", x="Score", hue='Skills', col="Skills", data=test, col_wrap=2)

plt.show()
# Potential and Value

# Selecting unique Age from DataFrame

potential = dataset.sort_values("Potential")['Potential'].unique()

# Selecting mean Overall from DataFrame

overall = dataset.sort_values("Overall")['Overall'].unique()

# Selecting mean value from DataFrame

value = dataset.groupby("Potential")["Value"].mean().values



plt.figure(figsize=(8,6))

plt.title('Mean Value vs Potential Rating', fontsize=20, fontweight='bold')

plt.xlabel('Potential', fontsize=15)

plt.ylabel('Player Value', fontsize=15)

sns.set_style("whitegrid")

plt.plot(potential, value, label="Value")

plt.legend(loc=4, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")

plt.show()
#Overall and Value

# Selecting unique Age from DataFrame

overall = dataset.sort_values("Overall")['Overall'].unique()

# Selecting mean Overall from DataFrame

value = dataset.groupby("Overall")["Value"].mean().values



plt.figure(figsize=(8,6))

plt.title('Mean Value vs Overall Rating', fontsize=20, fontweight='bold')

plt.xlabel('Overall', fontsize=15)

plt.ylabel('Player Value', fontsize=15)

sns.set_style("whitegrid")

plt.plot(overall, value, label="Value")

plt.legend(loc=4, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")

plt.show()
# Age and value

# Selecting unique Age from DataFrame

age = dataset.sort_values("Age")['Age'].unique()

# Selecting mean Overall from DataFrame

value = dataset.groupby("Age")["Value"].mean().values



plt.figure(figsize=(8,6))

plt.title('Mean Value vs Age', fontsize=20, fontweight='bold')

plt.xlabel('Age', fontsize=15)

plt.ylabel('Player Value', fontsize=15)

sns.set_style("whitegrid")

plt.plot(age, value, label="Value")

plt.legend(loc=4, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")

plt.show()
# Age and quality

# Age and potential

# Selecting players with age smaller then 40 

dataset = processed_df.loc[processed_df['Age'] <= 40]

# Selecting unique Age from DataFrame

age = dataset.sort_values("Age")['Age'].unique()

# Selecting mean Overall from DataFrame

overall = dataset.groupby("Age")["Overall"].mean().values

# Selecting mean Potential from DataFrame

potential = dataset.groupby("Age")["Potential"].mean().values



plt.figure(figsize=(10,6))

plt.title('Mean Overall/Potential Rating vs Age', fontsize=20, fontweight='bold')

plt.xlabel('Player Age', fontsize=15)

plt.ylabel('Player Rating', fontsize=15)

sns.set_style("whitegrid")

plt.plot(age, overall, label="Overall")

plt.plot(age, potential, label="Potential")

plt.legend(loc=4, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")

plt.show()
# Maximum player overall score per country

country_overall = dataset.groupby("Nationality")["Overall"].max()

country_overall = country_overall.reset_index()



data = [ dict(

        type = 'choropleth',

        locationmode = "country names",

        locations = country_overall['Nationality'],

        z = country_overall['Overall'],

        colorscale = [[55,"rgb(5, 10, 172)"],[59,"rgb(40, 60, 190)"],[63,"rgb(70, 100, 245)"],\

            [67,"rgb(90, 120, 245)"],[71,"rgb(106, 137, 247)"],[75,"rgb(220, 220, 220)"]],

        autocolorscale = False,

        reversescale = True,

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) ),

        colorbar = dict(

            autotick = False,

            title = 'Maximum<br>Overall Score'),

      ) ]



layout = dict(

    title = 'Player Quality',

    geo = dict(

        showframe = False,

        showcoastlines = False,

        projection = dict(

            type = 'Mercator'

        )

    )

)



fig = dict( data=data, layout=layout )

py.iplot( fig, validate=False )
# Maximum player potential score per country

country_potential = dataset.groupby("Nationality")["Potential"].max()

country_potential = country_potential.reset_index()



data = [ dict(

        type = 'choropleth',

        locationmode = "country names",

        locations = country_potential['Nationality'],

        z = country_potential['Potential'],

        colorscale = [[65,"rgb(5, 10, 172)"],[67,"rgb(40, 60, 190)"],[69,"rgb(70, 100, 245)"],\

            [71,"rgb(90, 120, 245)"],[73,"rgb(106, 137, 247)"],[75,"rgb(220, 220, 220)"]],

        autocolorscale = False,

        reversescale = True,

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) ),

        colorbar = dict(

            autotick = False,

            title = 'Maximum<br>Potential Score'),

      ) ]



layout = dict(

    title = 'Player Potential',

    geo = dict(

        showframe = False,

        showcoastlines = False,

        projection = dict(

            type = 'Mercator'

        )

    )

)



fig = dict( data=data, layout=layout )

py.iplot( fig, validate=False )
# Team value

club_value = dataset.groupby("Club")["Value"].sum()

club_value = club_value.reset_index()



sns.set(style="whitegrid")

# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(8, 10))



# Plot the total crashes

sns.set_color_codes("pastel")

sns.barplot(y=club_value.nlargest(20, "Value")["Club"], x=club_value.nlargest(20, "Value")["Value"], data=club_value,

            label="Total", color="b")

ax.set(xlim=(3e8, 9e8))



sns.despine(left=True, bottom=True)
# Team quality

club_overall = dataset.groupby("Club")["Overall"].mean()

club_overall = club_overall.reset_index()



sns.set(style="whitegrid")

# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(8, 10))



# Plot the total crashes

sns.set_color_codes("pastel")

sns.barplot(y=club_overall.nlargest(20, "Overall")["Club"], x=club_overall.nlargest(20, "Overall")["Overall"], data=club_overall,

            label="Total", color="b")

ax.set(xlim=(75, 85))

sns.despine(left=True, bottom=True)
# Team potential

club_potential = dataset.groupby("Club")["Potential"].mean()

club_potential = club_potential.reset_index()



sns.set(style="whitegrid")

# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(8, 10))



# Plot the total crashes

sns.set_color_codes("muted")

sns.barplot(y=club_potential.nlargest(20, "Potential")["Club"], x=club_potential.nlargest(20, "Potential")["Potential"], data=club_potential,

            label="Total", color="b")

ax.set(xlim=(75, 85))

sns.despine(left=True, bottom=True)
# Heatmap: show the correlation between similar features.

# Select a subset of the df

used_variables = ['Overall', 'Potential',

       'Value', 'Wage', 'Acceleration', 'Aggression', 'Agility', 'Balance',

       'Ball control', 'Composure', 'Crossing', 'Curve', 'Dribbling',

       'Finishing', 'Free kick accuracy', 'GK diving', 'GK handling',

       'GK kicking', 'GK positioning', 'GK reflexes', 'Heading accuracy',

       'Interceptions', 'Jumping', 'Long passing', 'Long shots', 'Marking',

       'Penalties', 'Positioning', 'Reactions', 'Short passing', 'Shot power',

       'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle',

       'Strength', 'Vision', 'Volleys']

used_columns = (processed_df.columns.get_level_values(None).isin(used_variables))

df = processed_df.loc[:, used_columns]



# Draw the full plot

plt.figure(figsize=(13,13))

sns.clustermap(df.corr(), center=0, cmap="vlag", linewidths=.75)

# Heatmap

# Select a subset of the df

used_variables = ['Overall', 'Potential', 'Value', 'Wage', 'CAM',

                  'CB', 'CDM', 'CF', 'CM', 'LAM', 'LB', 'LCB', 'LCM',

                  'LDM', 'LF', 'LM', 'LS', 'LW', 'LWB','RAM', 'RB',

                  'RCB', 'RCM', 'RDM', 'RF', 'RM', 'RS', 'RW', 'RWB', 'ST']

used_columns = (processed_df.columns.get_level_values(None).isin(used_variables))

df = processed_df.loc[:, used_columns]



# Draw the full plot

plt.figure(figsize=(13,13))

sns.clustermap(df.corr(), center=0, cmap="vlag", linewidths=.75)
preprocessed_df = processed_df.copy()

preprocessed_df = preprocessed_df.drop(['Name', 'Nationality', 'Club'], 1)

preprocessed_df = preprocessed_df.drop(preprocessed_df.columns[0], 1)

#Separate x and y

y_value = preprocessed_df['Value'].copy()

y_wage = preprocessed_df['Wage'].copy()

preprocessed_df = preprocessed_df.drop(['Value', 'Wage'], 1)
# Add versatility to the model

def count_positions(position):

    return len(re.sub("[^\w]", " ",  position).split()) #Split the preferred positions



preprocessed_df['Versatility'] = preprocessed_df['Preferred Positions'].apply(count_positions)

preprocessed_df.head()
# Remove the initial one hot encoded column and remove space in columns to enable graphviz tree plotting

preprocessed_df = preprocessed_df.drop(['Preferred Positions'], 1)

preprocessed_df.columns = preprocessed_df.columns.str.replace('\s+', '_')
# Train-Test split

X_train_value, X_test_value, y_train_value, y_test_value = train_test_split(preprocessed_df, y_value, test_size=0.2)

X_train_wage, X_test_wage, y_train_wage, y_test_wage = train_test_split(preprocessed_df, y_wage, test_size=0.2)
# Create permutations of different hyperparameter combinations

num_boost_round = 999

early_stopping_rounds = 10



xgb_params = {

                'n_estimators':[500],

                # Other parameters

                'objective':['reg:linear'],

                'eval_metric':[ ["rmse" ,"mae"]],

                'min_child_weight': [1],

                'eta': [0.01, 0.1], #Learning rate

                'gamma': [0.05, 0.1], #Minimum acceptable entropy improvement to perform a split

                #Bias\variance tradeoff knobs

                'subsample': [0.8, 1.0],

                'colsample_bytree': [0.8, 1.0],

                'max_depth': [6, 9]

              }



params_perms = [dict(zip(xgb_params, v)) for v in product(*xgb_params.values())]

print(len(params_perms))

print(params_perms[0])
dtrain_value = xgb.DMatrix(X_train_value, label=y_train_value)

dtest_value = xgb.DMatrix(X_test_value, label=y_test_value)

evaluation_set_value = [(dtest_value, "Test_set")]
mean_train_value = np.mean(y_train_value)



# Get predictions on the test set

baseline_predictions_value = np.ones(y_test_value.shape) * mean_train_value



# Compute MAE

mae_baseline_value = mean_absolute_error(y_test_value, baseline_predictions_value)



# Compute RMSE

rmse_baseline_value = sqrt(mean_squared_error(y_test_value, baseline_predictions_value))



print("Baseline MAE is {:.2f}".format(mae_baseline_value))

print("Baseline RMSE is {:.2f}".format(rmse_baseline_value))
best_score_value = mae_baseline_value

best_params_value = {}

for i, grid_search_value in enumerate(params_perms):

    print("Permutation number:", i+1)

    cv_results_value = xgb.cv(grid_search_value,

                        dtrain_value,

                        num_boost_round=num_boost_round,

                        seed=42,

                        nfold=2,

                        metrics=['mae', 'rmse'],

                        early_stopping_rounds=early_stopping_rounds

                        )

    

    mean_mae_value = cv_results_value['test-mae-mean'].min()

    boost_rounds_value = cv_results_value['test-mae-mean'].argmin()

    print("\tMAE {} for {} rounds".format(mean_mae_value, boost_rounds_value))

    if mean_mae_value < best_score_value:

        best_score_value = mean_mae_value

        best_params_value = grid_search_value
print("Best Score:", best_score_value)

print("Best Parameters:", best_params_value)
#Retrain a model with new best params

model_value = xgb.train(

    best_params_value,

    dtrain_value,

    num_boost_round=num_boost_round,

    evals=evaluation_set_value,

    early_stopping_rounds=early_stopping_rounds

)
print("Average guess error", mae_baseline_value)

print("vs Mean Absolute Error", mean_absolute_error(model_value.predict(dtest_value), y_test_value))
#Feature Importance

plt.figure(figsize=(20,15))

xgb.plot_importance(model_value, ax=plt.gca())
#Tree Diagram

plt.figure()

plt.savefig('tree_value.png')
#Predict the value based on the model.

dpred_value = xgb.DMatrix(preprocessed_df, label = y_value)

y_pred = pd.Series(model_value.predict(dpred_value))
#Top 10% players with overpriced transfer values.

sub_value = y_value.subtract(y_pred)

exp_value = processed_df.loc[sub_value.nlargest(int(len(sub_value)/1000)).index]

exp_value
#Top 10% players with underpriced transfer values.

cheap_value = sub_value.nsmallest(int(len(sub_value)/1000))

cheap_value = processed_df.loc[cheap_value.index]

cheap_value
model_value.save_model("model_value.model")
dtrain_wage = xgb.DMatrix(X_train_wage, label=y_train_wage)

dtest_wage = xgb.DMatrix(X_test_wage, label=y_test_wage)

evaluation_set_wage = [(dtest_wage, "Test_set")]
mean_train_wage = np.mean(y_train_wage)



# Get predictions on the test set

baseline_predictions_wage = np.ones(y_test_wage.shape) * mean_train_wage



# Compute MAE

mae_baseline_wage = mean_absolute_error(y_test_wage, baseline_predictions_wage)



# Compute RMSE

rmse_baseline_wage = sqrt(mean_squared_error(y_test_wage, baseline_predictions_wage))



print("Baseline MAE is {:.2f}".format(mae_baseline_wage))

print("Baseline RMSE is {:.2f}".format(rmse_baseline_wage))
best_score_wage = mae_baseline_wage

best_params_wage = {}

for i, grid_search_wage in enumerate(params_perms):

    print("Permutation number:", i+1)

    cv_results_wage = xgb.cv(grid_search_wage,

                        dtrain_wage,

                        num_boost_round=num_boost_round,

                        seed=42,

                        nfold=2,

                        metrics=['mae', 'rmse'],

                        early_stopping_rounds=early_stopping_rounds

                        )

    

    mean_mae_wage = cv_results_wage['test-mae-mean'].min()

    boost_rounds_wage = cv_results_wage['test-mae-mean'].argmin()

    print("\tMAE {} for {} rounds".format(mean_mae_wage, boost_rounds_wage))

    if mean_mae_wage < best_score_wage:

        best_score_wage = mean_mae_wage

        best_params_wage = grid_search_wage
#Retrain a model with new best params

model_wage = xgb.train(

    best_params_wage,

    dtrain_wage,

    num_boost_round=num_boost_round,

    evals=evaluation_set_wage,

    early_stopping_rounds=early_stopping_rounds

)
print("Average guess error", mae_baseline_wage)

print("vs Mean Absolute Error", mean_absolute_error(model_wage.predict(dtest_wage), y_test_wage))
#Feature Importance

plt.figure(figsize=(20,15))

xgb.plot_importance(model_wage, ax=plt.gca())
#Predict the value based on the model.

dpred_wage = xgb.DMatrix(preprocessed_df, label = y_wage)

y_pred_wage = pd.Series(model_wage.predict(dpred_wage))
#Top 10% players with overpriced wages.

sub_wage = y_value.subtract(y_pred_wage)

exp_wage = processed_df.loc[sub_wage.nlargest(int(len(sub_wage)/1000)).index]

exp_wage
#Top 10% players with underpriced transfer values.

cheap_wage = sub_wage.nsmallest(int(len(sub_wage)/1000))

cheap_wage = processed_df.loc[cheap_wage.index]

cheap_wage
model_wage.save_model("model_wage.model")