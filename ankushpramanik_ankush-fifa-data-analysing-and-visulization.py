# basic operations

import numpy as np

import pandas as pd 



# for visualizations

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')



# file path

import os

print(os.listdir("../input"))
# reading the data and also checking the computation time



%time data = pd.read_csv('../input/data.csv')



print(data.shape)
# checking the first 5 rows and columns



data.head()
# checking if the data contains any NULL value



data.isnull().sum()
# decsribing the data



data.describe()
# taking out the information from the given data



data.info()
# filling the missing value for the continous variables for proper data visualization



data['ShortPassing'].fillna(data['ShortPassing'].mean(), inplace = True)

data['Volleys'].fillna(data['Volleys'].mean(), inplace = True)

data['Dribbling'].fillna(data['Dribbling'].mean(), inplace = True)

data['Curve'].fillna(data['Curve'].mean(), inplace = True)

data['FKAccuracy'].fillna(data['FKAccuracy'], inplace = True)

data['LongPassing'].fillna(data['LongPassing'].mean(), inplace = True)

data['BallControl'].fillna(data['BallControl'].mean(), inplace = True)

data['HeadingAccuracy'].fillna(data['HeadingAccuracy'].mean(), inplace = True)

data['Finishing'].fillna(data['Finishing'].mean(), inplace = True)

data['Crossing'].fillna(data['Crossing'].mean(), inplace = True)

data['Weight'].fillna('200lbs', inplace = True)

data['Contract Valid Until'].fillna(2019, inplace = True)

data['Height'].fillna("5'11", inplace = True)

data['Loaned From'].fillna('None', inplace = True)

data['Joined'].fillna('Jul 1, 2018', inplace = True)

data['Jersey Number'].fillna(8, inplace = True)

data['Body Type'].fillna('Normal', inplace = True)

data['Position'].fillna('ST', inplace = True)

data['Club'].fillna('No Club', inplace = True)

data['Work Rate'].fillna('Medium/ Medium', inplace = True)

data['Skill Moves'].fillna(data['Skill Moves'].median(), inplace = True)

data['Weak Foot'].fillna(3, inplace = True)

data['Preferred Foot'].fillna('Right', inplace = True)

data['International Reputation'].fillna(1, inplace = True)

data['Wage'].fillna('€200K', inplace = True)

data.fillna(0, inplace = True)
def defending(data):

    return int(round((data[['Marking', 'StandingTackle', 

                               'SlidingTackle']].mean()).mean()))



def general(data):

    return int(round((data[['HeadingAccuracy', 'Dribbling', 'Curve', 

                               'BallControl']].mean()).mean()))



def mental(data):

    return int(round((data[['Aggression', 'Interceptions', 'Positioning', 

                               'Vision','Composure']].mean()).mean()))



def passing(data):

    return int(round((data[['Crossing', 'ShortPassing', 

                               'LongPassing']].mean()).mean()))



def mobility(data):

    return int(round((data[['Acceleration', 'SprintSpeed', 

                               'Agility','Reactions']].mean()).mean()))

def power(data):

    return int(round((data[['Balance', 'Jumping', 'Stamina', 

                               'Strength']].mean()).mean()))



def rating(data):

    return int(round((data[['Potential', 'Overall']].mean()).mean()))



def shooting(data):

    return int(round((data[['Finishing', 'Volleys', 'FKAccuracy', 

                               'ShotPower','LongShots', 'Penalties']].mean()).mean()))
# renaming a column

data.rename(columns={'Club Logo':'Club_Logo'}, inplace=True)



# adding these categories to the data



data['Defending'] = data.apply(defending, axis = 1)

data['General'] = data.apply(general, axis = 1)

data['Mental'] = data.apply(mental, axis = 1)

data['Passing'] = data.apply(passing, axis = 1)

data['Mobility'] = data.apply(mobility, axis = 1)

data['Power'] = data.apply(power, axis = 1)

data['Rating'] = data.apply(rating, axis = 1)

data['Shooting'] = data.apply(shooting, axis = 1)
players = data[['Name','Defending','General','Mental','Passing',

                'Mobility','Power','Rating','Shooting','Flag','Age',

                'Nationality', 'Photo', 'Club_Logo', 'Club']]



players.head()
# comparison of preferred foot over the different players



data['Preferred Foot'].value_counts().head(50).plot.bar(color = 'purple')

plt.title('Most Preferred Foot of the Players', fontsize = 20)
#  comparison of international reputation among the players



data['International Reputation'].value_counts()
# plotting a pie chart to represent share of international repuatation



labels = ['1', '2', '3', '4', '5']

sizes = [16532, 1261, 309, 51, 6]

colors = ['red', 'yellow', 'green', 'pink', 'blue']

explode = [0.1, 0.1, 0.2, 0.5, 0.9]



plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True)

plt.title('A Pie Chart for International Repuatation for the Football Players', fontsize = 25)

plt.legend()

plt.show()

# A player's foot (left or right) that is weaker than their preferred foot. A player's attribute rated between 1 to 5 

# which specifies the shot power and ball control for the other foot of that player than his preferred foot's.The higher

# rate defines the higher shot power and ball control.



data['Weak Foot'].value_counts()
# plotting a pie chart to represent the share of week foot players



labels = ['5', '4', '3', '2', '1'] 

size = [229, 2662, 11349, 3761, 158]

colors = ['red', 'yellow', 'green', 'pink', 'blue']

explode = [0.1, 0.1, 0.1, 0.1, 0.1]



plt.pie(size, labels = labels, colors = colors, explode = explode, shadow = True)

plt.title('Pie Chart for Representing Week Foot of the Players', fontsize = 25)

plt.legend()

plt.show()
# different positions acquired by the players 



plt.figure(figsize = (12, 8))

sns.set(style = 'dark', palette = 'colorblind', color_codes = True)

ax = sns.countplot('Position', data = data, color = 'orange')

ax.set_xlabel(xlabel = 'Different Positions in Football', fontsize = 16)

ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)

ax.set_title(label = 'Comparison of Positions and Players', fontsize = 20)

plt.show()
# defining a function for cleaning the Weight data



def extract_value_from(value):

  out = value.replace('lbs', '')

  return float(out)



# applying the function to weight column

#data['value'] = data['value'].apply(lambda x: extract_value_from(x))

data['Weight'] = data['Weight'].apply(lambda x : extract_value_from(x))



data['Weight'].head()
# defining a function for cleaning the wage column



def extract_value_from(Value):

    out = Value.replace('€', '')

    if 'M' in out:

        out = float(out.replace('M', ''))*1000000

    elif 'K' in Value:

        out = float(out.replace('K', ''))*1000

    return float(out)
# applying the function to the wage column



data['Value'] = data['Value'].apply(lambda x: extract_value_from(x))

data['Wage'] = data['Wage'].apply(lambda x: extract_value_from(x))



data['Wage'].head()
# Comparing the players' Wages



sns.set(style = 'dark', palette = 'bright', color_codes = True)

x = data.Wage

plt.rcParams['figure.figsize'] = (18, 7)

sns.countplot(x, data = data, color = 'y')

plt.xlabel('Wage Range for Players', fontsize = 16)

plt.ylabel('Count of the Players', fontsize = 16)

plt.title('Comparing the wages of players', fontsize = 20)

plt.xticks(rotation = 90)

plt.show()
# Skill Moves of Players



plt.figure(figsize = (7, 8))

ax = sns.countplot(x = 'Skill Moves', data = data, palette = 'pastel')

ax.set_title(label = 'Count of players on Basis of their skill moves', fontsize = 20)

ax.set_xlabel(xlabel = 'Number of Skill Moves', fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

plt.show()
# Height of Players



plt.figure(figsize = (13, 8))

ax = sns.countplot(x = 'Height', data = data, palette = 'dark')

ax.set_title(label = 'Count of players on Basis of Height', fontsize = 20)

ax.set_xlabel(xlabel = 'Height in Foot per inch', fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

plt.show()
# To show Different body weight of the players participating in the FIFA 2019



plt.figure(figsize = (30, 8))

sns.countplot(x = 'Weight', data = data, palette = 'rainbow')

plt.title('Different Weights of the Players Participating in FIFA 2019', fontsize = 20)

plt.xlabel('Heights associated with the players', fontsize = 16)

plt.ylabel('count of Players', fontsize = 16)

plt.show()
# To show Different Work rate of the players participating in the FIFA 2019



plt.figure(figsize = (15, 8))

sns.countplot(x = 'Work Rate', data = data, palette = 'hls')

plt.title('Different work rates of the Players Participating in the FIFA 2019', fontsize = 20)

plt.xlabel('Work rates associated with the players', fontsize = 16)

plt.ylabel('count of Players', fontsize = 16)

plt.show()
# To show Different Speciality Score of the players participating in the FIFA 2019



x = data.Special

plt.figure(figsize = (12, 8))

ax = sns.distplot(x, bins = 58, kde = False, color = 'm')

ax.set_xlabel(xlabel = 'Special score range', fontsize = 16)

ax.set_ylabel(ylabel = 'Count of the Players',fontsize = 16)

ax.set_title(label = 'Histogram for the Speciality Scores of the Players', fontsize = 20)

plt.show()
# To show Different potential scores of the players participating in the FIFA 2019



x = data.Potential

plt.figure(figsize=(12,8))

ax = sns.distplot(x, bins = 58, kde = False, color = 'y')

ax.set_xlabel(xlabel = "Player\'s Potential Scores", fontsize = 16)

ax.set_ylabel(ylabel = 'Number of players', fontsize = 16)

ax.set_title(label = 'Histogram of players Potential Scores', fontsize = 20)

plt.show()
# To show Different overall scores of the players participating in the FIFA 2019



sns.set(style = "dark", palette = "deep", color_codes = True)

x = data.Overall

plt.figure(figsize = (12,8))

ax = sns.distplot(x, bins = 52, kde = False, color = 'r')

ax.set_xlabel(xlabel = "Player\'s Scores", fontsize = 16)

ax.set_ylabel(ylabel = 'Number of players', fontsize = 16)

ax.set_title(label = 'Histogram of players Overall Scores', fontsize = 20)

plt.show()
# To show Different nations participating in the FIFA 2019



data['Nationality'].value_counts().head(80).plot.bar(color = 'orange', figsize = (20, 7))

plt.title('Different Nations Participating in FIFA 2019')

plt.xlabel('Name of The Country')

plt.ylabel('count')

plt.show()
# To show that there are people having same age

# Histogram: number of players's age



sns.set(style = "dark", palette = "colorblind", color_codes = True)

x = data.Age

plt.figure(figsize = (12,8))

ax = sns.distplot(x, bins = 58, kde = False, color = 'g')

ax.set_xlabel(xlabel = "Player\'s age", fontsize = 16)

ax.set_ylabel(ylabel = 'Number of players', fontsize = 16)

ax.set_title(label = 'Histogram of players age', fontsize = 20)

plt.show()
#comparing the different body types of the players participating in FIFA 2019



data['Body Type'].value_counts().plot.bar(color = 'green', figsize = (7, 5))

plt.title('Different Body Types')

plt.xlabel('Body Types')

plt.ylabel('count')

plt.show()
# selecting some of the interesting and important columns from the set of columns in the given dataset



selected_columns = ['Name', 'Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value',

                    'Wage', 'Special', 'Preferred Foot', 'International Reputation', 'Weak Foot',

                    'Skill Moves', 'Work Rate', 'Body Type', 'Position', 'Height', 'Weight',

                    'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

                    'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

                    'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

                    'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

                    'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

                    'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

                    'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause']

# violin plot 



plt.rcParams['figure.figsize'] = (18, 7)

sns.violinplot(data['Overall'], data['Age'], hue = data['Preferred Foot'], palette = 'Set1')

plt.title('Comparison of Overall Scores and age wrt Preferred foot', fontsize = 20)
# bubble plot



plt.scatter(data['Overall'], data['International Reputation'], s = data['Age']*1000, c = 'pink')

plt.xlabel('Overall Ratings', fontsize = 20)

plt.ylabel('International Reputation', fontsize = 20)

plt.legend('Age')

plt.show()
# having a look at the sample of selected data



data_selected.sample(5)

# plotting a correlation heatmap



plt.rcParams['figure.figsize'] = (30, 20)

sns.heatmap(data_selected[['Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value',

                    'Wage', 'Special', 'Preferred Foot', 'International Reputation', 'Weak Foot',

                    'Skill Moves', 'Work Rate', 'Body Type', 'Position', 'Height', 'Weight',

                    'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

                    'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

                    'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

                    'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

                    'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

                    'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

                    'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause']].corr(), annot = True)



plt.title('Histogram of the Dataset', fontsize = 30)

plt.show()
# best players per each position with their age, club, and nationality based on their overall scores



data.iloc[data.groupby(data['Position'])['Overall'].idxmax()][['Position', 'Name', 'Age', 'Club', 'Nationality']]
# best players from each positions with their age, nationality, club based on their potential scores



data.iloc[data.groupby(data['Position'])['Potential'].idxmax()][['Position', 'Name', 'Age', 'Club', 'Nationality']]
# picking up the countries with highest number of players to compare their overall scores



data['Nationality'].value_counts().head(8)
# Every Nations' Player and their Weights



some_countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Columbia')

data_countries = data.loc[data['Nationality'].isin(some_countries) & data['Weight']]



plt.rcParams['figure.figsize'] = (12, 7)

ax = sns.violinplot(x = data_countries['Nationality'], y = data_countries['Weight'], palette = 'colorblind')

ax.set_xlabel(xlabel = 'Countries', fontsize = 9)

ax.set_ylabel(ylabel = 'Weight in lbs', fontsize = 9)

ax.set_title(label = 'Distribution of Weight of players from different countries', fontsize = 20)

plt.show()
# Every Nations' Player and their overall scores



some_countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Columbia')

data_countries = data.loc[data['Nationality'].isin(some_countries) & data['Overall']]



plt.rcParams['figure.figsize'] = (12, 7)

ax = sns.barplot(x = data_countries['Nationality'], y = data_countries['Overall'], palette = 'spring')

ax.set_xlabel(xlabel = 'Countries', fontsize = 9)

ax.set_ylabel(ylabel = 'Overall Scores', fontsize = 9)

ax.set_title(label = 'Distribution of overall scores of players from different countries', fontsize = 20)

plt.show()
# Every Nations' Player and their wages



some_countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Columbia')

data_countries = data.loc[data['Nationality'].isin(some_countries) & data['Wage']]



plt.rcParams['figure.figsize'] = (12, 7)

ax = sns.barplot(x = data_countries['Nationality'], y = data_countries['Wage'], palette = 'Purples')

ax.set_xlabel(xlabel = 'Countries', fontsize = 9)

ax.set_ylabel(ylabel = 'Wage', fontsize = 9)

ax.set_title(label = 'Distribution of Wages of players from different countries', fontsize = 15)

plt.show()
# Every Nations' Player and their International Reputation



some_countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Columbia')

data_countries = data.loc[data['Nationality'].isin(some_countries) & data['International Reputation']]



plt.rcParams['figure.figsize'] = (12, 7)

ax = sns.violinplot(x = data_countries['Nationality'], y = data_countries['International Reputation'], palette = 'autumn')

ax.set_xlabel(xlabel = 'Countries', fontsize = 9)

ax.set_ylabel(ylabel = 'Distribution of reputation', fontsize = 9)

ax.set_title(label = 'Distribution of International Repuatation of players from different countries', fontsize = 15)

plt.show()
# finding the the popular clubs around the globe



data['Club'].value_counts().head(10)
some_clubs = ('CD Leganés', 'Southampton', 'RC Celta', 'Empoli', 'Fortuna Düsseldorf', 'Manchestar City',

             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid')



data_clubs = data.loc[data['Club'].isin(some_clubs) & data['Overall']]



plt.rcParams['figure.figsize'] = (14, 8)

ax = sns.barplot(x = data_clubs['Club'], y = data_clubs['Overall'], palette = 'inferno')

ax.set_xlabel(xlabel = 'Some Popular Clubs', fontsize = 9)

ax.set_ylabel(ylabel = 'Overall Score', fontsize = 9)

ax.set_title(label = 'Distribution of Overall Score in Different popular Clubs', fontsize = 20)

plt.show()
# Distribution of Ages in some Popular clubs



some_clubs = ('CD Leganés', 'Southampton', 'RC Celta', 'Empoli', 'Fortuna Düsseldorf', 'Manchestar City',

             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid')



data_club = data.loc[data['Club'].isin(some_clubs) & data['Wage']]



plt.rcParams['figure.figsize'] = (14, 8)

ax = sns.violinplot(x = 'Club', y = 'Age', data = data_club, palette = 'magma')

ax.set_xlabel(xlabel = 'Names of some popular Clubs', fontsize = 10)

ax.set_ylabel(ylabel = 'Distribution', fontsize = 10)

ax.set_title(label = 'Disstribution of Ages in some Popular Clubs', fontsize = 20)
# Distribution of Wages in some Popular clubs



some_clubs = ('CD Leganés', 'Southampton', 'RC Celta', 'Empoli', 'Fortuna Düsseldorf', 'Manchestar City',

             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid')



data_club = data.loc[data['Club'].isin(some_clubs) & data['Wage']]



plt.rcParams['figure.figsize'] = (16, 8)

ax = sns.violinplot(x = 'Club', y = 'Wage', data = data_club, palette = 'Reds')

ax.set_xlabel(xlabel = 'Names of some popular Clubs', fontsize = 10)

ax.set_ylabel(ylabel = 'Distribution', fontsize = 10)

ax.set_title(label = 'Disstribution of Wages in some Popular Clubs', fontsize = 20)

plt.show()
# Distribution of Wages in some Popular clubs



some_clubs = ('CD Leganés', 'Southampton', 'RC Celta', 'Empoli', 'Fortuna Düsseldorf', 'Manchestar City',

             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid')



data_club = data.loc[data['Club'].isin(some_clubs) & data['International Reputation']]



plt.rcParams['figure.figsize'] = (16, 8)

ax = sns.violinplot(x = 'Club', y = 'International Reputation', data = data_club, palette = 'bright')

ax.set_xlabel(xlabel = 'Names of some popular Clubs', fontsize = 10)

ax.set_ylabel(ylabel = 'Distribution of Reputation', fontsize = 10)

ax.set_title(label = 'Disstribution of International Reputation in some Popular Clubs', fontsize = 20)

plt.show()
some_clubs = ('CD Leganés', 'Southampton', 'RC Celta', 'Empoli', 'Fortuna Düsseldorf', 'Manchestar City',

             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid')



data_clubs = data.loc[data['Club'].isin(some_clubs) & data['Weight']]



plt.rcParams['figure.figsize'] = (14, 8)

ax = sns.violinplot(x = 'Club', y = 'Weight', data = data_clubs, palette = 'Oranges')

ax.set_xlabel(xlabel = 'Some Popular Clubs', fontsize = 9)

ax.set_ylabel(ylabel = 'Weight in lbs', fontsize = 9)

ax.set_title(label = 'Distribution of Weight in Different popular Clubs', fontsize = 20)

plt.show()
# finding 15 youngest Players from the dataset



youngest = data.sort_values('Age', ascending = True)[['Name', 'Age', 'Club', 'Nationality']].head(15)

print(youngest)
# finding 15 eldest players from the dataset



eldest = data.sort_values('Age', ascending = False)[['Name', 'Age', 'Club', 'Nationality']].head(15)

print(eldest)
# checking the head of the joined column



data['Joined'].head()
# The longest membership in the club



import datetime



now = datetime.datetime.now()

data['Join_year'] = data.Joined.dropna().map(lambda x: x.split(',')[1].split(' ')[1])

data['Years_of_member'] = (data.Join_year.dropna().map(lambda x: now.year - int(x))).astype('int')

membership = data[['Name', 'Club', 'Years_of_member']].sort_values(by = 'Years_of_member', ascending = False).head(10)

membership.set_index('Name', inplace=True)

membership
# defining the features of players



player_features = ('Acceleration', 'Aggression', 'Agility', 

                   'Balance', 'BallControl', 'Composure', 

                   'Crossing', 'Dribbling', 'FKAccuracy', 

                   'Finishing', 'GKDiving', 'GKHandling', 

                   'GKKicking', 'GKPositioning', 'GKReflexes', 

                   'HeadingAccuracy', 'Interceptions', 'Jumping', 

                   'LongPassing', 'LongShots', 'Marking', 'Penalties')



# Top four features for every position in football



for i, val in data.groupby(data['Position'])[player_features].mean().iterrows():

    print('Position {}: {}, {}, {}'.format(i, *tuple(val.nlargest(4).index)))
from math import pi



idx = 1

plt.figure(figsize=(15,45))

for position_name, features in data.groupby(data['Position'])[player_features].mean().iterrows():

    top_features = dict(features.nlargest(5))

    

    # number of variable

    categories=top_features.keys()

    N = len(categories)



    # We are going to plot the first line of the data frame.

    # But we need to repeat the first value to close the circular graph:

    values = list(top_features.values())

    values += values[:1]



    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)

    angles = [n / float(N) * 2 * pi for n in range(N)]

    angles += angles[:1]



    # Initialise the spider plot

    ax = plt.subplot(9, 3, idx, polar=True)



    # Draw one axe per variable + add labels labels yet

    plt.xticks(angles[:-1], categories, color='grey', size=8)



    # Draw ylabels

    ax.set_rlabel_position(0)

    plt.yticks([25,50,75], ["25","50","75"], color="grey", size=7)

    plt.ylim(0,100)

    

    plt.subplots_adjust(hspace = 0.5)

    

    # Plot data

    ax.plot(angles, values, linewidth=1, linestyle='solid')



    # Fill area

    ax.fill(angles, values, 'b', alpha=0.1)

    

    plt.title(position_name, size=11, y=1.1)

    

    idx += 1 
# Top 10 left footed footballers



data[data['Preferred Foot'] == 'Left'][['Name', 'Age', 'Club', 'Nationality']].head(10)
# Top 10 Right footed footballers



data[data['Preferred Foot'] == 'Right'][['Name', 'Age', 'Club', 'Nationality']].head(10)
# comparing the performance of left-footed and right-footed footballers

# ballcontrol vs dribbing



sns.lmplot(x = 'BallControl', y = 'Dribbling', data = data, col = 'Preferred Foot')
# visualizing clubs with highest number of different countries



data.groupby(data['Club'])['Nationality'].nunique().sort_values(ascending = False).head(10)
# visualizing clubs with highest number of different countries



data.groupby(data['Club'])['Nationality'].nunique().sort_values(ascending = True).head(10)