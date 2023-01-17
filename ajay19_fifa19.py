import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))
data = pd.read_csv("../input/data.csv")

print(data.shape)

data.sample(5)
data.isnull().sum()
#describe the data

data.describe()
data.info()
#let's fill the missing values for continuous variables.

data['Dribbling'].fillna(data['Dribbling'].mean(), inplace = True)

data['ShortPassing'].fillna(data['ShortPassing'].mean(), inplace = True)

data['Volleys'].fillna(data['Volleys'].mean(), inplace = True)

data['Curve'].fillna(data['Curve'].mean(), inplace = True)

data['FKAccuracy'].fillna(data['FKAccuracy'].mean(), inplace = True)

data['LongPassing'].fillna(data['LongPassing'].mean(), inplace = True)

data['BallControl'].fillna(data['BallControl'].mean(), inplace = True)

data['Acceleration'].fillna(data['Acceleration'].mean(), inplace = True)

data['SprintSpeed'].fillna(data['SprintSpeed'].mean(), inplace = True)

data['Crossing'].fillna(data['Crossing'].mean(), inplace = True)

data['Finishing'].fillna(data['Finishing'].mean(), inplace = True)

data['HeadingAccuracy'].fillna(data['HeadingAccuracy'].mean(), inplace = True)



data['Weight'].fillna('200lbs', inplace = True)

data['Contract Valid Until'].fillna(2019, inplace = True)

data['Height'].fillna("5'10", inplace = True)

data['Joined'].fillna('Jul 1, 2018', inplace = True)

data['Loaned From'].fillna('None', inplace = True)

data['Jersey Number'].fillna(0, inplace = True)











data.info()
data['Body Type'].value_counts(dropna=False)
data['Body Type'].fillna('Normal', inplace = True)

data['Body Type'].replace('Messi', 'Stocky', inplace = True)

data['Body Type'].replace('C. Ronaldo', 'Stocky', inplace = True)

data['Body Type'].replace('Courtois', 'Stocky', inplace = True)

data['Body Type'].replace('PLAYER_BODY_TYPE_25', 'Stocky', inplace = True)

data['Body Type'].replace('Shaqiri', 'Stocky', inplace = True)

data['Body Type'].replace('Neymar', 'Stocky', inplace = True)

data['Body Type'].replace('Akinfenwa', 'Stocky',inplace = True)

data['Body Type'].value_counts()
data['Position'].fillna('CM', inplace = True)

data['Club'].fillna('No Club', inplace = True)

data['Work Rate'].fillna('Medium/ Medium', inplace = True)

data['Skill Moves'].fillna(data['Skill Moves'].median(), inplace = True)

data['Weak Foot'].fillna(3, inplace = True)

data['Preferred Foot'].fillna('Right', inplace = True)

data['International Reputation'].fillna(3, inplace = True)

data.info()
data['Preferred Foot'].value_counts().plot.bar()
data['International Reputation'].value_counts().plot.bar()
labels = ['5', '4', '3', '2', '1'] 

size = [229, 2662, 11349, 3761, 158]

colors = ['red', 'yellow', 'green', 'black', 'blue']

explode = [0.1, 0.1, 0.1, 0.1, 0.1]



plt.pie(size, labels = labels, colors = colors, explode = explode, shadow = True)

plt.title('Pie Chart for Representing Week Foot of the Players', fontsize = 14)

plt.legend()

plt.show()
plt.figure(figsize = (12, 8))

sns.set(style = 'dark', palette = 'colorblind', color_codes = True)

ax = sns.countplot('Position', data = data, color = 'yellow')

ax.set_xlabel(xlabel = 'Different Positions in Football', fontsize = 16)

ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)

ax.set_title(label = 'Comparison of Positions and Players', fontsize = 20)

plt.show()
data['Weight'].head()
def extract_value_from(value):

  out = value.replace('lbs', '')

  return float(out)



# applying the function to weight column

#data['value'] = data['value'].apply(lambda x: extract_value_from(x))

data['Weight'] = data['Weight'].apply(lambda x : extract_value_from(x))



data['Weight'].head()
'''

data['Wage'].fillna('€100K', inplace = True)



data['Weight1'] = 0



for x in range(len(data['Weight'])):

    data['Weight1'][x] = data['Weight'][x].replace('lbs', '')



data['Weight1'].head()

'''
def extract_value_from(Value):

    out = Value.replace('€', '')

    if 'M' in out:

        out = float(out.replace('M', ''))*1000000

    elif 'K' in Value:

        out = float(out.replace('K', ''))*1000

    return float(out)



data['Value'] = data['Value'].apply(lambda x: extract_value_from(x))

data['Wage'] = data['Wage'].apply(lambda x: extract_value_from(x))



data['Wage'].head()
data['Skill Moves'].value_counts().plot.bar()
data['Height'].value_counts().plot.bar()
plt.figure(figsize = (60, 15))

sns.countplot(x = 'Weight', data = data, palette = 'dark')

plt.title('Different Weights of the Players Participating in FIFA 2019', fontsize = 20)

plt.xlabel('Heights associated with the players', fontsize = 16)

plt.ylabel('count of Players', fontsize = 16)

plt.show()
data['Work Rate'].value_counts().plot.bar()

# To show Different Speciality Score of the players participating in the FIFA 2019



sns.set(style = 'dark', palette = 'colorblind', color_codes = True)

x = data.Special

plt.figure(figsize = (12, 8))

ax = sns.distplot(x, bins = 58, kde = False, color = 'm')

ax.set_xlabel(xlabel = 'Special score range', fontsize = 16)

ax.set_ylabel(ylabel = 'Count of the Players',fontsize = 16)

ax.set_title(label = 'Histogram for the Speciality Scores of the Players', fontsize = 20)

plt.show()
# To show Different potential scores of the players participating in the FIFA 2019



sns.set(style = "dark", palette = "muted", color_codes = True)

x = data.Potential

plt.figure(figsize=(12,8))

ax = sns.distplot(x, bins = 58, kde = False, color = 'y')

ax.set_xlabel(xlabel = "Player\'s Potential Scores", fontsize = 16)

ax.set_ylabel(ylabel = 'Number of players', fontsize = 16)

ax.set_title(label = 'Histogram of players Potential Scores', fontsize = 20)

plt.show()
data['Nationality'].value_counts().plot.bar(color = 'orange', figsize = (55, 15 ))
data['Nationality'].value_counts()
data['Age'].value_counts().plot.bar(color = 'blue', figsize = (55, 15 ))
data['Body Type'].value_counts().plot.bar(color = 'green', figsize = (7, 5))

plt.title('Different Body Types')

plt.xlabel('Body Types')

plt.ylabel('count')

plt.show()
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



data_selected = pd.DataFrame(data, columns = selected_columns)
data.iloc[data.groupby(data['Position'])['Overall'].idxmax()][['Position', 'Name', 'Age', 'Club', 'Nationality']]
# Every Nations' Player and their Weights



some_countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Columbia')

data_countries = data.loc[data['Nationality'].isin(some_countries) & data['Weight']]



plt.rcParams['figure.figsize'] = (12, 7)

ax = sns.violinplot(x = data_countries['Nationality'], y = data_countries['Weight'], palette = 'colorblind')

ax.set_xlabel(xlabel = 'Countries', fontsize = 9)

ax.set_ylabel(ylabel = 'Weight in lbs', fontsize = 9)

ax.set_title(label = 'Distribution of Weight of players from different countries', fontsize = 20)

plt.show()
# Every Nations' Player and their wages



some_countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Columbia')

data_countries = data.loc[data['Nationality'].isin(some_countries) & data['Wage']]



plt.rcParams['figure.figsize'] = (12, 7)

ax = sns.barplot(x = data_countries['Nationality'], y = data_countries['Wage'],)

ax.set_xlabel(xlabel = 'Countries', fontsize = 9)

ax.set_ylabel(ylabel = 'Wage', fontsize = 9)

ax.set_title(label = 'Distribution of Wages of players from different countries', fontsize = 15)

plt.show()

some_clubs = ('CD Leganés', 'Southampton', 'RC Celta', 'Empoli', 'Fortuna Düsseldorf', 'Manchestar City',

             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid')



data_clubs = data.loc[data['Club'].isin(some_clubs) & data['Overall']]



plt.rcParams['figure.figsize'] = (14, 8)

ax = sns.barplot(x = data_clubs['Club'], y = data_clubs['Overall'], palette = 'deep')

ax.set_xlabel(xlabel = 'Some Popular Clubs', fontsize = 9)

ax.set_ylabel(ylabel = 'Overall Score', fontsize = 9)

ax.set_title(label = 'Distribution of Overall Score in Different popular Clubs', fontsize = 20)

plt.show()
some_clubs = ('CD Leganés', 'Southampton', 'RC Celta', 'Empoli', 'Fortuna Düsseldorf', 'Manchestar City',

             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid')



data_clubs = data.loc[data['Club'].isin(some_clubs) & data['Wage']]



plt.rcParams['figure.figsize'] = (14, 8)

ax = sns.barplot(x = data_clubs['Club'], y = data_clubs['Wage'], palette = 'deep')

ax.set_xlabel(xlabel = 'Some Popular Clubs', fontsize = 9)

ax.set_ylabel(ylabel = 'Wage', fontsize = 9)

ax.set_title(label = 'Distribution of Wage in Different popular Clubs', fontsize = 20)

plt.show()
some_clubs = ('CD Leganés', 'Southampton', 'RC Celta', 'Empoli', 'Fortuna Düsseldorf', 'Manchestar City',

             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid')



data_clubs = data.loc[data['Club'].isin(some_clubs) & data['Weight']]



plt.rcParams['figure.figsize'] = (14, 8)

ax = sns.violinplot(x = 'Club', y = 'Weight', data = data_clubs, palette = 'dark')

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

# Top 10 left footed footballers



data[data['Preferred Foot'] == 'Left'][['Name', 'Age', 'Club', 'Nationality']].head(10)

# Top 10 Right footed footballers



data[data['Preferred Foot'] == 'Right'][['Name', 'Age', 'Club', 'Nationality']].head(10)

# comparing the performance of left-footed and right-footed footballers

# ballcontrol vs dribbing



sns.lmplot(x = 'BallControl', y = 'Dribbling', data = data, col = 'Preferred Foot')
# comparing the performance of left-footed and right-footed footballers

# ballcontrol vs dribbing



sns.lmplot(x = 'ShortPassing', y = 'LongPassing', data = data, col = 'Preferred Foot')