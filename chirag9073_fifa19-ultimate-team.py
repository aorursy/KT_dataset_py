import numpy as np 

import pandas as pd

import os

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set(style="ticks")
from wordcloud import WordCloud
df=pd.read_csv("../input/fifa19/data.csv")
df.head(10)
df.shape
df.info()
df.isnull().sum()
df.columns
f,ax = plt.subplots(figsize=(25, 15))

sns.heatmap(df.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.show()
plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='white',

                          width=1920,

                          height=1080

                         ).generate(" ".join(df.Nationality))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('nationality.png')

plt.show()
df['Club'].fillna('No Club', inplace = True)

df['Position'].fillna('ST', inplace = True)
impute_mean = df.loc[:, ['Crossing', 'Finishing', 'HeadingAccuracy',

                                 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy',

                                 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed',

                                 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping',

                                 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions',

                                 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking',

                                 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

                                 'GKKicking', 'GKPositioning', 'GKReflexes']]
for i in impute_mean.columns:

    df[i].fillna(df[i].mean(), inplace = True)
impute_mode = df.loc[:, ['Body Type','International Reputation', 'Height', 'Weight', 'Preferred Foot','Jersey Number']]

for i in impute_mode.columns:

    df[i].fillna(df[i].mode()[0], inplace = True)
impute_median = df.loc[:, ['Weak Foot', 'Skill Moves', ]]

for i in impute_median.columns:

    df[i].fillna(df[i].median(), inplace = True)
df.columns[df.isna().any()]
df.fillna(0, inplace = True)
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
df['Defending'] = df.apply(defending, axis = 1)

df['General'] = df.apply(general, axis = 1)

df['Mental'] = df.apply(mental, axis = 1)

df['Passing'] = df.apply(passing, axis = 1)

df['Mobility'] = df.apply(mobility, axis = 1)

df['Power'] = df.apply(power, axis = 1)

df['Rating'] = df.apply(rating, axis = 1)

df['Shooting'] = df.apply(shooting, axis = 1)
df.rename(columns={'Club Logo':'Club_Logo'}, inplace=True)
players = df[['Name','Defending','General','Mental','Passing',

                'Mobility','Power','Rating','Shooting','Flag','Age',

                'Nationality', 'Photo', 'Club_Logo', 'Club']]



players.head(20)
plt.figure(figsize = (18, 8))

sns.set(style="ticks")

plt.style.use('fivethirtyeight')

ax = sns.countplot('Position', data = df, palette = 'dark')

ax.set_xlabel(xlabel = 'Different Positions in Football', fontsize = 16)

ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)

ax.set_title(label = 'Comparison of Positions and Players', fontsize = 20)

plt.show()
plt.figure(figsize = (13, 8))

sns.set(style="ticks")

ax = sns.countplot(x = 'Height', data = df, palette = 'bone')

ax.set_title(label = 'Count of players on Basis of Height', fontsize = 20)

ax.set_xlabel(xlabel = 'Height in Foot per inch', fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

plt.show()

plt.figure(figsize = (15, 7))

sns.set(style="ticks")

sns.countplot(x = 'Work Rate', data = df, palette = 'plasma')

plt.title('Different work rates of the Players Participating in the FIFA 2019', fontsize = 20)

plt.xlabel('Work rates associated with the players', fontsize = 16)

plt.ylabel('count of Players', fontsize = 16)

plt.show()

x = df.Special

plt.figure(figsize = (12, 8))

sns.set(style="ticks")

ax = sns.distplot(x, bins = 58, kde = False, color = 'cyan')

ax.set_xlabel(xlabel = 'Special score range', fontsize = 16)

ax.set_ylabel(ylabel = 'Count of the Players',fontsize = 16)

ax.set_title(label = 'Histogram for the Speciality Scores of the Players', fontsize = 20)

plt.show()
some_countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Columbia')

data_countries = df.loc[df['Nationality'].isin(some_countries) & df['Overall']]

plt.rcParams['figure.figsize'] = (15, 7)

sns.set(style="ticks")

ax = sns.barplot(x = data_countries['Nationality'], y = data_countries['Overall'], palette = 'spring')

ax.set_xlabel(xlabel = 'Countries', fontsize = 9)

ax.set_ylabel(ylabel = 'Overall Scores', fontsize = 9)

ax.set_title(label = 'Distribution of overall scores of players from different countries', fontsize = 20)

plt.show()
df['Club'].value_counts().head(10)
data = df.copy()
# for visualizations

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

sns.set(style="ticks")
some_clubs = ('CD Leganés', 'Southampton', 'RC Celta', 'Empoli', 'Fortuna Düsseldorf', 'Manchestar City',

             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid')



data_clubs = data.loc[data['Club'].isin(some_clubs) & data['Overall']]



plt.rcParams['figure.figsize'] = (15, 8)

ax = sns.boxplot(x = data_clubs['Club'], y = data_clubs['Overall'], palette = 'inferno')

ax.set_xlabel(xlabel = 'Some Popular Clubs', fontsize = 9)

ax.set_ylabel(ylabel = 'Overall Score', fontsize = 9)

ax.set_title(label = 'Distribution of Overall Score in Different popular Clubs', fontsize = 20)

plt.xticks(rotation = 90)

plt.show()
left = data[data['Preferred Foot'] == 'Left'][['Name', 'Age', 'Club', 'Nationality']].head(10)

left
plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='white',

                          width=1920,

                          height=1080

                         ).generate(" ".join(left.Name))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('left.png')

plt.show()
right = data[data['Preferred Foot'] == 'Right'][['Name', 'Age', 'Club', 'Nationality']].head(10)

right
plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='white',

                          width=1920,

                          height=1080

                         ).generate(" ".join(right.Name))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('right.png')

plt.show()
sns.lmplot(x = 'BallControl', y = 'Dribbling', data = data, col = 'Preferred Foot')

plt.show()
data.groupby(data['Club'])['Nationality'].nunique().sort_values(ascending = False).head(10)
data.groupby(data['Club'])['Nationality'].nunique().sort_values(ascending = True).head(10)
df.head()
df.drop(['Unnamed: 0'],axis=1,inplace=True)
player = str(df.loc[df['Potential'].idxmax()][1])

print('Maximum Potential : '+str(df.loc[df['Potential'].idxmax()][1]))

print('Maximum Overall Perforamnce : '+str(df.loc[df['Overall'].idxmax()][1]))
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
i=0

best = []

while i < len(pr_cols):

    best.append(df.loc[df[pr_cols[i]].idxmax()][1])

    i +=1
best
plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='white',

                          width=1920,

                          height=1080

                         ).generate(" ".join(best))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('best.png')

plt.show()