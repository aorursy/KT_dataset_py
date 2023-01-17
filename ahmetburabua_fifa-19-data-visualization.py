# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/fifa19/data.csv")

df = data.copy()

df.head(10)
df.describe()
def country(x):

    return df[df["Nationality"] == x][["Name","Age","Overall","Club"]]
country("Turkey")
def club(x):

    return df[df["Club"] == x][["Name","Age","Overall","Nationality","Value","Contract Valid Until"]]
club("Fenerbahçe SK")
data.isnull().sum()
df['ShortPassing'].fillna(data['ShortPassing'].mean(), inplace = True)

df['Volleys'].fillna(data['Volleys'].mean(), inplace = True)

df['Dribbling'].fillna(data['Dribbling'].mean(), inplace = True)

df['Curve'].fillna(data['Curve'].mean(), inplace = True)

df['FKAccuracy'].fillna(data['FKAccuracy'], inplace = True)

df['LongPassing'].fillna(data['LongPassing'].mean(), inplace = True)

df['BallControl'].fillna(data['BallControl'].mean(), inplace = True)

df['HeadingAccuracy'].fillna(data['HeadingAccuracy'].mean(), inplace = True)

df['Finishing'].fillna(data['Finishing'].mean(), inplace = True)

df['Crossing'].fillna(data['Crossing'].mean(), inplace = True)

df['Weight'].fillna('200lbs', inplace = True)

df['Contract Valid Until'].fillna(2019, inplace = True)

df['Height'].fillna("5'11", inplace = True)

df['Loaned From'].fillna('None', inplace = True)

df['Joined'].fillna('Jul 1, 2018', inplace = True)

df['Jersey Number'].fillna(8, inplace = True)

df['Body Type'].fillna('Normal', inplace = True)

df['Position'].fillna('ST', inplace = True)

df['Club'].fillna('No Club', inplace = True)

df['Work Rate'].fillna('Medium/ Medium', inplace = True)

df['Skill Moves'].fillna(data['Skill Moves'].median(), inplace = True)

df['Weak Foot'].fillna(3, inplace = True)

df['Preferred Foot'].fillna('Right', inplace = True)

df['International Reputation'].fillna(1, inplace = True)

df['Wage'].fillna('€200K', inplace = True)
df.fillna(0,inplace=True)
def defending(data):

    return int(round((df[['Marking', 'StandingTackle', 

                               'SlidingTackle']].mean()).mean()))



def general(data):

    return int(round((df[['HeadingAccuracy', 'Dribbling', 'Curve', 

                               'BallControl']].mean()).mean()))



def mental(data):

    return int(round((df[['Aggression', 'Interceptions', 'Positioning', 

                               'Vision','Composure']].mean()).mean()))



def passing(data):

    return int(round((df[['Crossing', 'ShortPassing', 

                               'LongPassing']].mean()).mean()))



def mobility(data):

    return int(round((df[['Acceleration', 'SprintSpeed', 

                               'Agility','Reactions']].mean()).mean()))

def power(data):

    return int(round((df[['Balance', 'Jumping', 'Stamina', 

                               'Strength']].mean()).mean()))



def rating(data):

    return int(round((df[['Potential', 'Overall']].mean()).mean()))



def shooting(data):

    return int(round((df[['Finishing', 'Volleys', 'FKAccuracy', 

                               'ShotPower','LongShots', 'Penalties']].mean()).mean()))
df.rename(columns={'Club Logo':'Club_Logo'}, inplace=True)

##

df['Defending'] = df.apply(defending, axis = 1)

df['General'] = df.apply(general, axis = 1)

df['Mental'] = df.apply(mental, axis = 1)

df['Passing'] = df.apply(passing, axis = 1)

df['Mobility'] = df.apply(mobility, axis = 1)

df['Power'] = df.apply(power, axis = 1)

df['Rating'] = df.apply(rating, axis = 1)

df['Shooting'] = df.apply(shooting, axis = 1)
players = df[["Name","Age","Defending","Overall","Nationality","Club_Logo","Value","Power","Flag","Club","Position"]]

players.head()
## Comparison of preferred foot over different players

plt.rcParams['figure.figsize'] = (10,5)

sns.countplot(data['Preferred Foot'], palette = 'pink')

plt.title('Most Preferred Foot of the Players', fontsize = 20)

plt.show()
## Plotting a pie chart to represent share of international requatation



labels = ['1', '2', '3', '4', '5']

size = df['International Reputation'].value_counts()

colors = plt.cm.copper(np.linspace(0, 1, 5))

explode = [0.1, 0.3, 0.5, 0.7, 0.9]



plt.rcParams['figure.figsize'] = (9,9)

plt.pie(size , labels = labels, colors = colors, explode = explode , shadow = True)

plt.title('International Repuatation for the Football Players', fontsize = 30)

plt.legend()

plt.show()
## Different Position Acquired by The Players



plt.figure(figsize = (18,8))

plt.style.use('fivethirtyeight')

ax = sns.countplot('Position', data = data , palette = 'bone')

ax.set_xlabel(xlabel = 'Different Positions in Football',fontsize = 20)

ax.set_ylabel(ylabel = 'Count of Players', fontsize = 20)

ax.set_title(label = 'Comparison of Positions and Players', fontsize = 20)

plt.show()
# Histogram = number of players age



sns.set(style = "dark", palette = "colorblind", color_codes = True)

x = data.Age

plt.figure(figsize = (15,8))

ax = sns.distplot(x, bins = 58, kde = False, color = "g")

ax.set_xlabel(xlabel = "Player's age", fontsize = 18)

ax.set_ylabel(ylabel = "Number of Players", fontsize = 18)

ax.set_title(label = "Histogram of players age", fontsize = 20)

plt.show()
# Best Players Overall Scores



df.iloc[df.groupby(df["Position"])["Overall"].idxmax()][["Position","Name","Age","Club","Nationality"]]
# Best Players Potential Scores



df.iloc[df.groupby(df["Position"])["Potential"].idxmax()][["Position","Name","Age","Club","Nationality"]]
df["Nationality"].value_counts().head(10)
# Every Nations' Player and their International Reputation



some_countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Turkey')

df_countries = df.loc[df['Nationality'].isin(some_countries) & df['International Reputation']]



plt.rcParams['figure.figsize'] = (15, 7)

ax = sns.boxenplot(x = df_countries['Nationality'], y = df_countries['International Reputation'], palette = 'autumn')

ax.set_xlabel(xlabel = 'Countries', fontsize = 9)

ax.set_ylabel(ylabel = 'Distribution of reputation', fontsize = 9)

ax.set_title(label = 'Distribution of International Repuatation of players from different countries', fontsize = 15)

plt.show()
# Distribution of Ages in some Popular clubs



some_clubs = ('Fenerbahçe SK', 'Southampton', 'Juventus', 'Empoli', 'Manchester United', 'Manchestar City',

             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid')



df_club = df.loc[df['Club'].isin(some_clubs) & df['Wage']]



plt.rcParams['figure.figsize'] = (15, 8)

ax = sns.boxenplot(x = 'Club', y = 'Age', data = df_club, palette = 'magma')

ax.set_xlabel(xlabel = 'Names of some popular Clubs', fontsize = 10)

ax.set_ylabel(ylabel = 'Distribution', fontsize = 10)

ax.set_title(label = 'Disstribution of Ages in some Popular Clubs', fontsize = 20)

plt.xticks(rotation = 90)

plt.show()
youngest = df.sort_values("Age", ascending = True)[["Name","Age","Club","Nationality"]].head(10)

print(youngest)
eldest = df.sort_values("Age",ascending = False)[["Name","Age","Club","Nationality"]].head(10)

print(eldest)
# The longest membership in the club



import datetime



now = datetime.datetime.now()

df["Join_year"] = df.Joined.dropna().map(lambda x : x.split(",")[1].split(" ")[1])

df["Years_of_member"] = (df.Join_year.dropna().map(lambda x : now.year - int(x))).astype("int")

membership = df[["Name","Club","Years_of_member"]].sort_values(by = "Years_of_member",ascending = False).head(10)

membership.set_index("Name",inplace = True)

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



for i, val in df.groupby(df['Position'])[player_features].mean().iterrows():

    print('Position {}: {}, {}, {}'.format(i, *tuple(val.nlargest(4).index)))
from math import pi



idx = 1

plt.figure(figsize=(15,45))

for position_name, features in df.groupby(df['Position'])[player_features].mean().iterrows():

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



df[df['Preferred Foot'] == 'Left'][['Name', 'Age', 'Club', 'Nationality']].head(10)
# Top 10 Right footed footballers



df[df['Preferred Foot'] == 'Right'][['Name', 'Age', 'Club', 'Nationality']].head(10)