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

        

        

# We dont Probably need the Gridlines. Do we? If yes comment this line

sns.set(style="ticks")



flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"] # defining the colour palette

flatui = sns.color_palette(flatui)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
img1 = '/kaggle/input/image/img_club_logo.jpg'

img2 = '/kaggle/input/image/img_player.jpg'

img3 = '/kaggle/input/image/graph.png'

img4 = '/kaggle/input/image/img_flag.jpg'

img = '/kaggle/input/image/players.png'
path = '/kaggle/input/fifa2019/FIFA_data.csv'

df=pd.read_csv(path) # reading the dataset

df.head()
df.shape # checking the number of rows and columns in the dataset
df.info() #Printing a concise summary of the DataFrame.
df.isnull().sum() # checking the count of the missing values in each column
df.columns # listing the columns
# Plotting the Heatmap of the columns using correlation matrix

f,ax = plt.subplots(figsize=(25, 15))

sns.heatmap(df.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.show()
#Nationality Text Size = Nationality Player Count
!pip install WordCloud
from wordcloud import WordCloud  # importing the worldcloud module. Wordcloud uses the text sizes to represent the frequency of the text
# Plotiing the wordcloud for the Nationalit column

plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='white',

                          width=1920,

                          height=1080

                         ).generate(" ".join(df.Nationality))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')

plt.show()
#Imputing the missing values for the columns Club and Position

df['Club'].fillna('No Club', inplace = True)

df['Position'].fillna('ST', inplace = True)
# selecting columns to impute the missing values by mean

to_impute_by_mean = df.loc[:, ['Crossing', 'Finishing', 'HeadingAccuracy',

                                 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy',

                                 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed',

                                 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping',

                                 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions',

                                 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking',

                                 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

                                 'GKKicking', 'GKPositioning', 'GKReflexes']]
# replacing the missing values with mean

for i in to_impute_by_mean.columns:

    df[i].fillna(df[i].mean(), inplace = True)
'''These are categorical variables and will be imputed by mode.'''

to_impute_by_mode = df.loc[:, ['Body Type','International Reputation', 'Height', 'Weight', 'Preferred Foot','Jersey Number']]

for i in to_impute_by_mode.columns:

    df[i].fillna(df[i].mode()[0], inplace = True)
'''The following variables are either discrete numerical or continuous numerical variables.So the will be imputed by median.'''

to_impute_by_median = df.loc[:, ['Weak Foot', 'Skill Moves', ]]

for i in to_impute_by_median.columns:

    df[i].fillna(df[i].median(), inplace = True)
df.head(20)
'''Columns remaining to be imputed'''

df.columns[df.isna().any()]
df.fillna(0, inplace = True) # Filling the remaining  missing values with zero

df.head(20)
# functions to get the rounded values from different columns

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

df.rename(columns={'Club Logo':'Club_Logo'}, inplace=True)
df.columns

# adding these categories to the data



df['Defending'] = df.apply(defending, axis = 1)

df['General'] = df.apply(general, axis = 1)

df['Mental'] = df.apply(mental, axis = 1)

df['Passing'] = df.apply(passing, axis = 1)

df['Mobility'] = df.apply(mobility, axis = 1)

df['Power'] = df.apply(power, axis = 1)

df['Rating'] = df.apply(rating, axis = 1)

df['Shooting'] = df.apply(shooting, axis = 1)
# dataset after transformation

df.head(10)
# creating the players dataset

players = df[['Name','Defending','General','Mental','Passing',

                'Mobility','Power','Rating','Shooting','Flag','Age',

                'Nationality', 'Photo', 'Club_Logo', 'Club']]



players.head(10)
# different positions acquired by the players 



plt.figure(figsize = (18, 8))

plt.style.use('fivethirtyeight')

ax = sns.countplot('Position', data = df, palette = 'dark')

ax.set_xlabel(xlabel = 'Different Positions in Football', fontsize = 16)

ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)

ax.set_title(label = 'Comparison of Positions and Players', fontsize = 20)

plt.show()
# plotting coun of players based on their heights

plt.figure(figsize = (13, 8))

ax = sns.countplot(x = 'Height', data = df, palette = 'bone')

ax.set_title(label = 'Count of players on Basis of Height', fontsize = 20)

ax.set_xlabel(xlabel = 'Height in Foot per inch', fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

plt.show()

# To show Different Work rate of the players participating in the FIFA 2019



plt.figure(figsize = (15, 7))

# plt.style.use('_classic_test')



sns.countplot(x = 'Work Rate', data = df, palette = 'hls')

plt.title('Different work rates of the Players Participating in the FIFA 2019', fontsize = 20)

plt.xlabel('Work rates associated with the players', fontsize = 16)

plt.ylabel('count of Players', fontsize = 16)

plt.show()



x = df.Special

plt.figure(figsize = (12, 8))

plt.style.use('tableau-colorblind10')



ax = sns.distplot(x, bins = 58, kde = False, color = 'cyan')

ax.set_xlabel(xlabel = 'Special score range', fontsize = 16)

ax.set_ylabel(ylabel = 'Count of the Players',fontsize = 16)

ax.set_title(label = 'Histogram for the Speciality Scores of the Players', fontsize = 20)

plt.show()
# Every Nations' Player and their overall scores



some_countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Columbia') # defining a tuple consisting of country names

data_countries = df.loc[df['Nationality'].isin(some_countries) & df['Overall']] # extracting the overall data of the countries selected in the line above

data_countries.head()



plt.rcParams['figure.figsize'] = (15, 7)

ax = sns.barplot(x = data_countries['Nationality'], y = data_countries['Overall'], palette = 'spring') # creating a bargraph

ax.set_xlabel(xlabel = 'Countries', fontsize = 9)

ax.set_ylabel(ylabel = 'Overall Scores', fontsize = 9)

ax.set_title(label = 'Distribution of overall scores of players from different countries', fontsize = 20)

plt.show()
df['Club'].value_counts().head(10) # finding the number of players in each club
data = df.copy() # creating a copy dataset
# for visualizations

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

sns.set(style="ticks")
some_clubs = ('CD Leganés', 'Southampton', 'RC Celta', 'Empoli', 'Fortuna Düsseldorf', 'Manchestar City',

             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid') # creating a tuple of club names



data_clubs = data.loc[data['Club'].isin(some_clubs) & data['Overall']] # extracting the overall data of the clubs selected in the line above



data_clubs.head()
plt.rcParams['figure.figsize'] = (15, 8)

ax = sns.boxplot(x = data_clubs['Club'], y = data_clubs['Overall'], palette = 'inferno') # creating a boxplot

ax.set_xlabel(xlabel = 'Some Popular Clubs', fontsize = 9)

ax.set_ylabel(ylabel = 'Overall Score', fontsize = 9)

ax.set_title(label = 'Distribution of Overall Score in Different popular Clubs', fontsize = 20)

plt.xticks(rotation = 90)

plt.show()
# finding out the top 10 left footed footballers



left = data[data['Preferred Foot'] == 'Left'][['Name', 'Age', 'Club', 'Nationality']].head(10)

left
# finding out the top 10 Right footed footballers



right = data[data['Preferred Foot'] == 'Right'][['Name', 'Age', 'Club', 'Nationality']].head(10)

right
# comparing the performance of left-footed and right-footed footballers

# ballcontrol vs dribbing



sns.lmplot(x = 'BallControl', y = 'Dribbling', data = data, col = 'Preferred Foot')

plt.show()
data.groupby(data['Club'])['Nationality'].nunique().sort_values(ascending = False).head(10) # checking the clubs where players from the most number of nations play
data.groupby(data['Club'])['Nationality'].nunique().sort_values(ascending = True).head(10) # checking the clubs where players from the least number of nations play
df.head()
df.drop(['Unnamed: 0'],axis=1,inplace=True) # dropping the unnamed column
df.head() # dataset after dropping column
#Player with maximum Potential and Overall Performance

player = str(df.loc[df['Potential'].idxmax()][1])

print('Maximum Potential : '+str(df.loc[df['Potential'].idxmax()][1]))

print('Maximum Overall Perforamnce : '+str(df.loc[df['Overall'].idxmax()][1]))
# finding the best players for each performance criteria



pr_cols=['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',

       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',

       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',

       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',

       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',

       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',

       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']

i=0

while i < len(pr_cols):

    print('Best {0} : {1}'.format(pr_cols[i],df.loc[df[pr_cols[i]].idxmax()][1]))

    i += 1
# creating a list of best players in each of the pr_cols criteria

i=0

best = []

while i < len(pr_cols):

    best.append(df.loc[df[pr_cols[i]].idxmax()][1])

    i +=1
best
# Plot to show the preferred foot choice of different players

f, ax = plt.subplots(figsize=(8, 6))

sns.countplot(x="Preferred Foot", hue="Real Face", data=df)

plt.show()
df.loc[df['Potential'].idxmax()][1] # Finding the player with the maximum potential
# showing the name of the players which occurs the most number of times from the first 20 names

plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='black',

                          width=1920,

                          height=1080

                         ).generate(" ".join(df.Name[0:20]))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('players.png')

plt.show()
df.columns # all the columns in the dataset
# checking which clubs have been mentioned the most

plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='black',

                          width=1920,

                          height=1080

                         ).generate(" ".join(df.Club))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('players.png')

plt.show()
# showing the name of the players which occurs the most number of times(left join)

plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='black',

                          width=1920,

                          height=1080

                         ).generate(" ".join(left.Name))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('players.png')

plt.show()
# showing the name of the players which occurs the most number of times(right join)

plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='white',

                          width=1920,

                          height=1080

                         ).generate(" ".join(right.Name))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('players.png')

plt.show()
# Checking which player has been mentioned the most in the 'best' list that we have prepared

plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='white',

                          width=1920,

                          height=1080

                         ).generate(" ".join(best))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('players.png')

plt.show()