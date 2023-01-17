import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns
#importing data set

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
fifa_dataset = pd.read_csv('/kaggle/input/fifa19/data.csv')

fifa_dataset.drop('Unnamed: 0', axis =1, inplace=True)

fifa_dataset.head(10)
#checking for null values

fifa_dataset.isnull().any()
fifa_dataset['ShortPassing'].fillna(fifa_dataset['ShortPassing'].mean(), inplace = True)

fifa_dataset['Volleys'].fillna(fifa_dataset['Volleys'].mean(), inplace = True)

fifa_dataset['Dribbling'].fillna(fifa_dataset['Dribbling'].mean(), inplace = True)

fifa_dataset['Curve'].fillna(fifa_dataset['Curve'].mean(), inplace = True)

fifa_dataset['FKAccuracy'].fillna(fifa_dataset['FKAccuracy'], inplace = True)

fifa_dataset['LongPassing'].fillna(fifa_dataset['LongPassing'].mean(), inplace = True)

fifa_dataset['BallControl'].fillna(fifa_dataset['BallControl'].mean(), inplace = True)

fifa_dataset['HeadingAccuracy'].fillna(fifa_dataset['HeadingAccuracy'].mean(), inplace = True)

fifa_dataset['Finishing'].fillna(fifa_dataset['Finishing'].mean(), inplace = True)

fifa_dataset['Crossing'].fillna(fifa_dataset['Crossing'].mean(), inplace = True)

fifa_dataset['Weight'].fillna('200lbs', inplace = True)

fifa_dataset['Contract Valid Until'].fillna(2019, inplace = True)

fifa_dataset['Height'].fillna("5'11", inplace = True)

fifa_dataset['Loaned From'].fillna('None', inplace = True)

fifa_dataset['Joined'].fillna('Jul 1, 2018', inplace = True)

fifa_dataset['Jersey Number'].fillna(8, inplace = True)

fifa_dataset['Body Type'].fillna('Normal', inplace = True)

fifa_dataset['Position'].fillna('ST', inplace = True)

fifa_dataset['Club'].fillna('No Club', inplace = True)

fifa_dataset['Work Rate'].fillna('Medium/ Medium', inplace = True)

fifa_dataset['Skill Moves'].fillna(fifa_dataset['Skill Moves'].median(), inplace = True)

fifa_dataset['Weak Foot'].fillna(3, inplace = True)

fifa_dataset['Preferred Foot'].fillna('Right', inplace = True)

fifa_dataset['International Reputation'].fillna(1, inplace = True)

fifa_dataset['Wage'].fillna('€300K', inplace = True)
#position of players on field

fifa_dataset['Position'].unique()
#top 10 countries producing players in FIFA-19

data = fifa_dataset.groupby('Nationality')['ID'].count().sort_values(ascending = False).reset_index().head(10)

countries = data['Nationality']

players = data['ID']

plt.figure(figsize =(11, 11))

colors=['powderblue', 'lavender', 'palegreen', 'navajowhite', 'pink', 'khaki', 'bisque', 'lightsalmon', 'thistle', 'springgreen']

plt.pie(players, colors = colors, autopct = '%.2f%%', shadow = True)

plt.legend(labels = countries)

plt.title('Top 10 countries represented by footballers in FIFA 19',fontsize = 20)

plt.show()
#as all column names ain't visible, just printing them out

for col in fifa_dataset.columns:

    print(col)
_5star_skills = fifa_dataset[fifa_dataset['Skill Moves']==5.0][['Name', 'Age', 'Nationality', 'Overall', 'Club', 'Value']].head(5).reset_index()

_5star_skills = _5star_skills.drop('index', axis =1)

_5star_skills
respective_positions = fifa_dataset.groupby('Position')['ID'].count().reset_index()

no_of_players = respective_positions['ID']

positions = respective_positions['Position']

plt.figure(figsize = (10,8))

plt.bar(positions , no_of_players, color = 'crimson')

plt.xticks(rotation = 90)

plt.title('Players and their respective positions', fontsize = 20)

plt.xlabel('Positions')

plt.ylabel('Number of Players')

plt.show()
#players and their preferred foot

preferred_foot = fifa_dataset.groupby('Preferred Foot')['ID'].count().reset_index()

plt.figure( figsize = (9, 8))

plt.pie(preferred_foot['ID'], labels= preferred_foot['Preferred Foot'], colors =['springgreen', 'teal'], autopct = '%.2f%%')

plt.title('Players and their preferred foot', fontsize = 20)

plt.show()
fifa_dataset['Value'] = fifa_dataset['Value'].str.replace('€','')

values= []

for i in fifa_dataset['Value']:

    if 'M' in i:

        i = i.replace('M', '')

        i = float(i)*1000000

    elif 'K' in i:

        i = i.replace('K', '')

        i = float(i)*1000

    values.append(float(i))

fifa_dataset['Value'] =values
most_valuable = fifa_dataset.sort_values('Value', ascending= False)[['Name', 'Age', 'Club', 'Nationality', 'Value']].head(10).reset_index()

most_valuable = most_valuable.drop('index', axis = 1)

most_valuable
oldest_players = fifa_dataset.sort_values('Age', ascending= False)[['Name', 'Age', 'Club', 'Nationality', 'Position']].head(10).reset_index()

oldest_players = oldest_players.drop('index', axis = 1)

oldest_players
data = fifa_dataset.groupby('Club')['Overall'].mean().reset_index().sort_values('Overall', ascending = False).head(10).reset_index()

data = data.drop('index', axis = 1)

plt.figure(figsize = (10,8))

plt.barh(data['Club'], data['Overall'], color = 'blue')

plt.title('Top 10 Clubs by Overall Stats', fontsize = 20)

plt.xlabel('Average Overall Stats')

plt.ylabel('Clubs')

plt.show()
club_value = fifa_dataset.groupby('Club')['Value'].sum().sort_values(ascending = False).reset_index().head(10)

plt.figure(figsize = (10,8))

plt.bar('Club', 'Value', data = club_value, color = 'm')

plt.xticks(rotation = 90)

plt.title('Top 10 most valuable clubs',fontsize = 20)

plt.xlabel('Club name')

plt.ylabel('Club Value (x 10^8)')

plt.show()
fifa_dataset['Wage'] = fifa_dataset['Wage'].str.replace('€','')

wages= []

for i in fifa_dataset['Wage']:

    i = i.replace('K', '')

    i = float(i)*1000

    wages.append(float(i))

fifa_dataset['Wage'] = wages
#average wage per age category

age_wage = fifa_dataset.groupby('Age')['Wage'].mean().reset_index()

age = age_wage['Age']

value= age_wage['Wage']

plt.figure(figsize = (10,8))

plt.barh(age, value, color = 'cyan', edgecolor = 'black')

plt.title('Average Wage per Age Category', fontsize = 20)

plt.xlabel('Wages')

plt.ylabel('Age')

plt.show()
age_overall  = fifa_dataset.groupby('Age')['Overall'].mean().reset_index().sort_values('Age')

age_potential  = fifa_dataset.groupby('Age')['Potential'].mean().reset_index().sort_values('Age')

plt.figure(figsize = (10,8))

plt.plot('Age', 'Overall', data = age_overall, color = 'darkcyan')

plt.plot('Age', 'Potential', linestyle = 'dashed', data = age_potential, color = 'crimson')

plt.title('Average Overall/Potential Ratings VS Age', fontsize = 20)

plt.xlabel('Age')

plt.ylabel('Rating')

plt.legend()

plt.show()
data = fifa_dataset.groupby('Work Rate')['ID'].count().reset_index()

print(data)

work_rate = data['Work Rate']

players = data['ID']

plt.figure(figsize = (9,9))

plt.pie(players, autopct = '%.1f%%', colors = colors,shadow = True, explode = (0,0,0,0.6,0.4,0.2,0,0,0), pctdistance = 0.8 )

circle = plt.Circle((0,0), 0.65, fc = 'white')

donut = plt.gcf()

donut.gca().add_artist(circle)

plt.axis('equal')

plt.title('Work Rate Stats',fontsize = 20)

plt.legend(labels = work_rate)

plt.show()
#top 20 most wage paying clubs in the world

data = fifa_dataset.groupby('Club')['Wage'].mean().reset_index().sort_values('Wage', ascending = False).head(20)

plt.figure(figsize = (10,8))

plt.barh(data['Club'], data['Wage'], color = 'lightsalmon')

plt.title('Top 20 clubs by average player wage',fontsize = 20)

plt.xlabel('Wage')

plt.ylabel('Club')

plt.show()
penalties = fifa_dataset[['Name', 'Club', 'Overall', 'Penalties']].sort_values('Penalties',ascending = False).head(10).reset_index()

penalties = penalties.drop('index', axis =1)

penalties

#Top 10 penalty takers of the game
finishing = fifa_dataset[['Name', 'Club', 'Finishing']].sort_values('Finishing', ascending = False).head(10).reset_index()

finishing = finishing.drop('index', axis =1)

finishing

#top 10 finishers of the game
freekicks = fifa_dataset[['Name', 'Club', 'FKAccuracy']].sort_values('FKAccuracy', ascending = False).head(10).reset_index()

freekicks = freekicks.drop('index', axis =1)

freekicks

#top 10 free kick takers
sprint_speed = fifa_dataset[['Name', 'Club', 'SprintSpeed']].sort_values('SprintSpeed', ascending = False).head(10).reset_index()

sprint_speed = sprint_speed.drop('index', axis = 1)

sprint_speed
most_potential = fifa_dataset[['Name', 'Club', 'Age', 'Position', 'Potential']].sort_values('Potential', ascending = False).head(10).reset_index()

most_potential = most_potential.drop('index', axis = 1)

most_potential
clubs = ('Borussia Dortmund', 'FC Barcelona', 'Real Madrid', 'Juventus', 'Paris Saint-Germain', 'Arsenal', 'Manchester City', 'Manchester United', 'Chelsea', 'Liverpool', 'Roma', 'Atlético Madrid', 'FC Bayern München', 'RB Leipzig', 'Ajax', 'FC Porto', 'AS Monaco', 'Inter', 'Atalanta')

club_age = fifa_dataset[fifa_dataset['Club'].isin(clubs)][['Club', 'Age']].reset_index()

club_pl_age = club_age.drop('index', axis = 1)

plt.figure(figsize = (12, 8))

sns.violinplot(x = 'Club', y = 'Age' , data = club_pl_age)

plt.xticks(rotation = 70)

plt.title('Age Distribution among Top Football Clubs', fontsize = 20)

plt.show()
country_ages= fifa_dataset[['Nationality','Age']]

countries = ['Argentina', 'Brazil', 'Portugal', 'Netherlands', 'Spain', 'Italy', 'Germany', 'England', 'France', 'Mexico', 'Iceland']

country_ages = country_ages[country_ages['Nationality'].isin(countries)]

plt.figure( figsize = (11,9))

sns.boxplot('Nationality', 'Age', data = country_ages, palette = 'Reds')

plt.title('Age distribution by Country', fontsize = 20)

plt.show()
plt.figure(figsize = (10,8))

sns.boxplot('Work Rate', 'Age', data = fifa_dataset, palette = 'GnBu_d')

plt.title('Age Vs Work Rate', fontsize = 20)

plt.xticks(rotation = 70)
strikers = fifa_dataset[fifa_dataset['Position'] == 'ST'][['Name', 'Club', 'Age', 'Value', 'Overall']].sort_values('Overall', ascending = False).head(10).reset_index()

strikers = strikers.drop('index', axis = 1)

strikers
centrebacks = fifa_dataset[fifa_dataset['Position'] == 'CB'][['Name', 'Club', 'Age', 'Value', 'Overall']].sort_values('Value', ascending = False).head(10).reset_index()

centrebacks = centrebacks.drop('index', axis = 1)

centrebacks

#top 10 centre backs by value
plt.figure( figsize = (19,15))

sns.heatmap(fifa_dataset[['Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value',

                    'Wage', 'Special', 'Preferred Foot', 'International Reputation', 'Weak Foot',

                    'Skill Moves', 'Work Rate', 'Body Type', 'Position', 'Height', 'Weight',

                    'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

                    'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

                    'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

                    'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

                    'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

                    'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

                    'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause']].corr(), annot = True, cmap = 'BuPu')

plt.title('Correlation among player features',fontsize = 20)

plt.show()
teenagers = fifa_dataset[fifa_dataset['Age'] < 20]

teenagers_value = fifa_dataset[fifa_dataset['Age'] < 20][['Name', 'Club', 'Age', 'Position', 'Value']].sort_values('Value', ascending = False).head(10).reset_index()

teenagers_value = teenagers_value.drop('index', axis = 1)

teenagers_value
data = teenagers.groupby('Club')['Value'].sum().sort_values(ascending = False).head(10).reset_index()

plt.figure(figsize = (10,8))

plt.bar('Club', 'Value', data = data, color = 'lightseagreen')

plt.xticks( rotation  = 70)

plt.title('Most Valuable Teen Club Squads')

plt.xlabel('Club')

plt.ylabel('Value (x 10 ^ 7)')

plt.show()
teen_club_overall = teenagers[['Club', 'Overall']]

teen_clubs_req = list(clubs)

plt.figure(figsize = (12,8))

teen_club_overall = teen_club_overall[teen_club_overall['Club'].isin(teen_clubs_req)]

sns.boxplot('Club', 'Overall', data = teen_club_overall , palette = 'RdBu')

plt.xticks( rotation = 70)

plt.title('Overall rating distribution among youth squads of major clubs')

plt.show()
teen_speed = teenagers[['Name', 'Age', 'Club', 'Overall', 'Position', 'SprintSpeed']].sort_values('SprintSpeed', ascending = False).head(10).reset_index()

teen_speed = teen_speed.drop('index', axis = 1)

teen_speed
teen_potential = teenagers[['Name', 'Age', 'Club', 'Potential', 'Position']].sort_values('Potential', ascending = False).head(10).reset_index()

teen_potential = teen_potential.drop('index', axis = 1)

teen_potential
barcelona = fifa_dataset[fifa_dataset['Club'] == 'FC Barcelona']

barcelona
data = barcelona[['Name', 'Value']]

plt.figure(figsize = (12,9))

plt.barh('Name', 'Value', data = data, color = 'r')

plt.title('FC Barcelona Player Values',fontsize = 20)

plt.xlabel('Value * 10^8')

plt.ylabel('PLayer Name')

plt.show()
nationality_value = fifa_dataset.groupby('Nationality')['Value'].sum().sort_values(ascending = False).head(20).reset_index()

plt.figure(figsize = (12,10))

plt.bar('Nationality', 'Value', data = nationality_value, color = 'springgreen')

plt.xticks( rotation = 60)

plt.title('Top 20 countries with highest team value',fontsize = 20)

plt.xlabel('Countries')

plt.ylabel('Value * 10^9')

plt.show()
clubs_player_rep = fifa_dataset[['Club','International Reputation']]

clubs_player_rep = clubs_player_rep[clubs_player_rep['Club'].isin(clubs)]

plt.figure(figsize= (18,10))

sns.violinplot('Club', 'International Reputation', data = clubs_player_rep)

plt.xticks(rotation = 60)

plt.title('International Reputation distribution among major clubs', fontsize = 20)

plt.show()
best_player_per_position = fifa_dataset.iloc[fifa_dataset.groupby(fifa_dataset['Position'])['Overall'].idxmax()][['Name', 'Age', 'Club', 'Position']].reset_index()

best_player_per_position = best_player_per_position.drop('index', axis = 1)

best_player_per_position

#best player by position on Fifa 19
teenagers = fifa_dataset[fifa_dataset['Age'] < 20].reset_index()

best_potential_teen_player_per_position = teenagers.iloc[teenagers.groupby(teenagers['Position'])['Potential'].idxmax()][['Name', 'Age', 'Club', 'Position', 'Value', 'Potential']].reset_index()

best_potential_teen_player_per_position = best_potential_teen_player_per_position.drop('index', axis = 1)

best_potential_teen_player_per_position

#best potential teen player by position on fifa 19
potential_by_country = fifa_dataset[['Nationality', 'Potential']]

potential_by_country = fifa_dataset[fifa_dataset['Nationality'].isin(countries)]

plt.figure(figsize = (11, 10))

sns.boxenplot('Nationality', 'Potential', data = potential_by_country)

plt.title('Potential distribution among countries', fontsize = 20)

plt.show()
import datetime

date_time = str(datetime.datetime.now())

date_time = int(date_time.strip().split(' ')[0].split('-')[0])

names_joined = fifa_dataset[['Name', 'Club', 'Joined']]

dates_joined = names_joined['Joined'].apply(lambda x: str(x).strip().split(',')[-1])

for i in range(len(dates_joined)):

    dates_joined[i] = date_time - float(dates_joined[i])

names_joined['Years at Club'] = dates_joined

years_at_club = names_joined[['Name', 'Club', 'Years at Club']].sort_values('Years at Club', ascending = False).head(10).reset_index()

years_at_club = years_at_club.drop('index', axis = 1)

years_at_club