from IPython.display import Image

Image(filename='../input/fifa-19-word/1_YOeQS.jpeg', width="800", height='50')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/fifa19/data.csv")

df.head()
### Check the rows and columns of the Dataset

df.shape
df.columns
# drop redundant columns

df.drop('Unnamed: 0', axis=1, inplace=True)
# view dataframe summary

df.info()
df.isnull().sum()
import missingno as msno

msno.matrix(df)
msno.bar(df, color = 'y', figsize = (15,8))
#Imputing Club and Position

df['Club'].fillna('No Club', inplace = True)

df['Position'].fillna('ST', inplace = True)
impute_by_mean = df.loc[:, ['Crossing', 'Finishing', 'HeadingAccuracy',

                                 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy',

                                 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed',

                                 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping',

                                 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions',

                                 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking',

                                 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

                                 'GKKicking', 'GKPositioning', 'GKReflexes']]
for i in impute_by_mean.columns:

    df[i].fillna(df[i].mean(), inplace = True)

### These are categorical variables and will be imputed by mode.

impute_by_mode = df.loc[:, ['Body Type','International Reputation', 'Height', 'Weight', 'Preferred Foot','Jersey Number']]

for i in impute_by_mode.columns:

    df[i].fillna(df[i].mode()[0], inplace = True)
### The following variables are either discrete numerical or continuous numerical variables.So the will be imputed by median.'''

impute_by_median = df.loc[:, ['Weak Foot', 'Skill Moves', ]]

for i in impute_by_median.columns:

    df[i].fillna(df[i].median(), inplace = True)
df.fillna(0, inplace = True)
##### Columns remaining to be imputed'''

df.columns[df.isna().any()]
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
df.isnull().sum()
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

plt.savefig('national.png')

plt.show()
df['Preferred Foot'].value_counts()/len(df)
plt.style.use('fivethirtyeight')

fig, ax = plt.subplots(figsize=(8,6))

graph = sns.countplot(ax=ax,x=df['Preferred Foot'], data=df, palette = 'bone')

graph.set_title('Preferred Foot of the Players', fontsize = 20)

graph.set_xticklabels(graph.get_xticklabels(),rotation=30)

for p in graph.patches:

    height = p.get_height()

    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
# plotting a pie chart to represent share of international repuatation



labels = ['1', '2', '3', '4', '5']

sizes = df['International Reputation'].value_counts()

colors = plt.cm.copper(np.linspace(0, 1, 5))

explode = [0.1, 0.1, 0.2, 0.5, 0.9]



plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True)

plt.title('International Repuatation for the Football Players', fontsize = 20)

plt.legend()

plt.show()
# different positions acquired by the players 



plt.figure(figsize = (18, 8))

plt.style.use('fivethirtyeight')

ax = sns.countplot('Position', data = df, palette = 'bone')

ax.set_xlabel(xlabel = 'Different Positions in Football', fontsize = 16)

ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)

ax.set_title(label = 'Comparison of Positions and Players', fontsize = 20)

plt.show()
# Skill Moves of Players



plt.figure(figsize = (10, 8))

ax = sns.countplot(x = 'Skill Moves', data = df, palette = 'pastel')

ax.set_title(label = 'Count of players on Basis of their skill moves', fontsize = 20)

ax.set_xlabel(xlabel = 'Number of Skill Moves', fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

plt.show()


plt.figure(figsize = (13, 8))

ax = sns.countplot(x = 'Height', data = df, palette = 'dark')

ax.set_title(label = 'Count of players on Basis of Height', fontsize = 20)

ax.set_xlabel(xlabel = 'Height in Foot per inch', fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

plt.show()
# To show Different Work rate of the players participating in the FIFA 2019



plt.figure(figsize = (15, 7))

plt.style.use('tableau-colorblind10')



sns.countplot(x = 'Work Rate', data = df, palette = 'hls')

plt.title('Different work rates of the Players Participating in the FIFA 2019', fontsize = 20)

plt.xlabel('Work rates associated with the players', fontsize = 16)

plt.ylabel('count of Players', fontsize = 16)

plt.show()
x = df.Special

plt.figure(figsize = (12, 8))

plt.style.use('tableau-colorblind10')



ax = sns.distplot(x, bins = 58, kde = False, color = 'm')

ax.set_xlabel(xlabel = 'Special score range', fontsize = 16)

ax.set_ylabel(ylabel = 'Count of the Players',fontsize = 16)

ax.set_title(label = 'Histogram for the Speciality Scores of the Players', fontsize = 20)

plt.show()
# To show Different potential scores of the players participating in the FIFA 2019



x = df.Potential

plt.figure(figsize=(12,8))

plt.style.use('seaborn-paper')



ax = sns.distplot(x, bins = 58, kde = False, color = 'y')

ax.set_xlabel(xlabel = "Player\'s Potential Scores", fontsize = 16)

ax.set_ylabel(ylabel = 'Number of players', fontsize = 16)

ax.set_title(label = 'Histogram of players Potential Scores', fontsize = 20)

plt.show()
# To show that there are people having same age

# Histogram: number of players's age



sns.set(style = "dark", palette = "colorblind", color_codes = True)

x = df.Age

plt.figure(figsize = (15,8))

plt.style.use('ggplot')

ax = sns.distplot(x, bins = 58, kde = False, color = 'g')

ax.set_xlabel(xlabel = "Player\'s age", fontsize = 16)

ax.set_ylabel(ylabel = 'Number of players', fontsize = 16)

ax.set_title(label = 'Histogram of players age', fontsize = 20)

plt.show()
# best players per each position with their age, club, and nationality based on their overall scores



df.iloc[df.groupby(df['Position'])['Overall'].idxmax()][['Position', 'Name', 'Age', 'Club', 'Nationality']]
df['Skill Moves'].value_counts()
fig, ax = plt.subplots(figsize=(12,8))

plt.style.use('fivethirtyeight')

graph = sns.countplot(ax=ax,x=df['Skill Moves'], data=df, palette = 'PuBuGn_d')

graph.set_title('Skill Moves of the Players', fontsize = 20)

graph.set_xticklabels(graph.get_xticklabels(), rotation=30)

for p in graph.patches:

    height = p.get_height()

    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
fig, ax = plt.subplots(figsize=(12,8))

plt.style.use('fivethirtyeight')

graph = sns.countplot(ax=ax,x=df['Skill Moves'], data=df, hue='Preferred Foot', palette = 'PuBuGn_d')

graph.set_title('Skill Moves of Players segregated by Preferred Foot'  , fontsize = 20)

graph.set_xticklabels(graph.get_xticklabels(), rotation=30)

for p in graph.patches:

    height = p.get_height()

    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
# To show Different overall scores of the players participating in the FIFA 2019



sns.set(style = "dark", palette = "deep", color_codes = True)

x = df.Overall

plt.figure(figsize = (12,8))

plt.style.use('ggplot')



ax = sns.distplot(x, bins = 52, kde = False, color = 'b')

ax.set_xlabel(xlabel = "Player\'s Potential Scores", fontsize = 16)

ax.set_ylabel(ylabel = 'Number of players', fontsize = 16)

ax.set_title(label = 'Histogram of players Overall Scores', fontsize = 20)

plt.show()
df['Nationality'].nunique()
df['Nationality'].unique()
top_country = df['Nationality'].value_counts().head(15)



top_country
fig, ax = plt.subplots(figsize=(12,8))

x = top_country.values

y = top_country.index

ax.barh(y, x, align='center', color='dimgray')

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Number of Players')

ax.set_ylabel('Name of Countries', rotation=0)

ax.set_title('Top 10 Countries with most number of players')

plt.show()
# Every Nations' Player and their overall scores



countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Columbia', 'Japan', 'Mexico', 'China PR')

data_country= df.loc[df['Nationality'].isin(countries) & df['Overall']]



plt.rcParams['figure.figsize'] = (15, 7)

ax = sns.barplot(x = data_country['Nationality'], y = data_country['Overall'], palette = 'PuBuGn_d')

ax.set_xlabel(xlabel = 'Countries', fontsize = 12)

ax.set_ylabel(ylabel = 'Overall Scores', fontsize = 12)

ax.set_title(label = 'Distribution of overall scores of players from different countries', fontsize = 20)

plt.show()
df['Overall'].value_counts().head()
df_best_players = pd.DataFrame.copy(df.sort_values(by ='Overall',ascending = False ).head(10))

plt.figure(1,figsize = (12,6))

sns.barplot(x ='Name' , y = 'Overall' , data = df_best_players, palette='Reds')

plt.ylim(85,95)

plt.show()
### top 10 eldest players

df.sort_values(by = 'Age' , ascending = False)[['Name','Club','Nationality','Overall', 'Age' ]].head()
### top 10 youngest players

df.sort_values(by = 'Age' , ascending = True)[['Name','Club','Nationality','Overall', 'Age' ]].head()
#### Best Finisher



df.sort_values(by = 'Finishing' , ascending = False)[['Name','Club','Nationality','Overall', 'Age','Finishing']].head()
### fastest players



df.sort_values(by = 'SprintSpeed' , ascending = False)[['Name','Club','Nationality','Overall', 'Age','SprintSpeed']].head()
#### Best dribbler

df.sort_values(by = 'Dribbling' , ascending = False)[['Name','Club','Nationality','Overall', 'Age','Dribbling']].head()
#Player with maximum Potential and Overall Performance

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
f, ax = plt.subplots(figsize=(8, 6))

sns.countplot(x="Preferred Foot", hue="Real Face", data=df)

plt.show()
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