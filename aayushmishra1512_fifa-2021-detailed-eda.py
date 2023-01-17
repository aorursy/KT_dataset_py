import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #importing our visualization libraries

import seaborn as sns
df = pd.read_csv('/kaggle/input/fifa-2021-complete-player-data/FIFA-21 Complete.csv',sep = ';') #reading our data which is ';' seperated
df.head()
df.isnull().sum() #checking null values in our data
df.info() #checkinng thee column details in our data
df.dtypes #checking the type of data present in our data frame
df['nationality'].value_counts().nunique() #checking total number of countries available in the data set
plt.style.use('dark_background') #top 50 nations that the players represent in FIFA 2021

plt.figure(figsize = (20,7))

df['nationality'].value_counts().head(10).plot.bar(color = 'orangered')

plt.title('Players from different countries present in FIFA-2021')

plt.xlabel('Country')

plt.ylabel('Count')

plt.show()
plt.style.use('dark_background') #checking the age distribution of the players in FIFA-2021

age  = df.age

plt.figure(figsize = (12,8))

ax = sns.distplot(age,bins = 60,kde = False,color ='orangered')

ax.set_ylabel('Number of Players')

ax.set_xlabel('Age')

ax.set_title('Distribution of Age of Players')

plt.show()
plt.style.use('dark_background') #popular clubs in FIFA-2021

plt.figure(figsize = (20,7))

df['team'].value_counts().head(20).plot.bar(color = 'orangered')

plt.title('Most Popular Clubs in FIFA-2021')

plt.xlabel('Clubs')

plt.ylabel('Count')

plt.show()
df.sort_values('age', ascending = True)[['name', 'age', 'team', 'nationality']].head(10) #The 10 youngest players present in the game
df.sort_values('age', ascending = False)[['name', 'age', 'team', 'nationality']].head(10) #the top 10 oldest players present in the game
df.groupby(['team'])['age'].mean().sort_values(ascending = True).head(10) #top 10 team with the youngest squad
df.groupby(['team'])['age'].mean().sort_values(ascending = False).head(10) #top 10 teams with the oldest squad
df['position'].value_counts() #checking the different positions and  the number of players playing them
df[df['position'] == 'CB'][['name', 'age', 'team', 'nationality']].head(10) #Top players that only play CB position
df[df['position'] == 'ST'][['name', 'age', 'team', 'nationality']].head(10) #Top players that only play ST position
df[df['position'] == 'CF'][['name', 'age', 'team', 'nationality']].head(10) #Top players that only play CF position
df[df['position'] == 'LW'][['name', 'age', 'team', 'nationality']].head(10) #Top players that only play LW position
df[df['position'] == 'RW'][['name', 'age', 'team', 'nationality']].head(10) #Top players that only play RW position
df[df['position'] == 'GK'][['name', 'age', 'team', 'nationality']].head(10) #Top players that only play GK position
df[['name', 'age', 'team', 'nationality']].head(10) #Top 10 Players present in the game
plt.figure(figsize=(12,8))

sns.heatmap(df[['age', 'nationality', 'overall', 'potential', 'team', 'hits', 'position']].corr(), annot = True) #overall correlation between the various columns present in our data

plt.title('Overall relation between columns of the Dataset', fontsize = 20)

plt.show()
def player(x): #method to check the  individual player information

    return df.loc[df['name']==x]



def country(x): #method to check the information of any country's football team

    return df[df['nationality'] == x][['name','overall','potential','position','hits','age','team']]



def club(x): #method to check the the club's player details

    return df[df['team'] == x][['name','overall','potential','position','hits','age']]

def overall(x): #method to get players with similar overall ratings

        return df[df['overall'] == x][['name','overall','potential','position','hits','age','team']]
player('Cristiano Ronaldo') #checking out Cristiano Ronaldo's stats
country('India') #checking the indian Football teams players and their stats
club('Real Madrid ') #the club stats for Real Madrid
overall(86).head(10) #gives us top 10 players with overall rating 86
plt.figure(figsize=(12,8)) #comparing overall score of a person versus their age

sns.lineplot(df['overall'], df['age'], palette = "Set1")

plt.title('Overall vs Age', fontsize = 20)

plt.show()
plt.figure(figsize=(12,8)) #comparing potential of a player vs their age

sns.lineplot(df['potential'], df['age'], palette = "Set1")

plt.title('Potential vs Age', fontsize = 20)

plt.show()
plt.style.use('dark_background') #checking the potential distribution of the players in FIFA-2021

potential  = df.potential

plt.figure(figsize = (12,8))

ax = sns.distplot(potential,bins = 60,kde = False,color ='orangered')

ax.set_ylabel('Number of Players')

ax.set_xlabel('Potential')

ax.set_title('Distribution of Potentials of Players')

plt.show()
plt.style.use('dark_background') #checking the overall scores distribution of the players in FIFA-2021

overall  = df.overall

plt.figure(figsize = (12,8))

ax = sns.distplot(overall,bins = 60,kde = False,color ='orangered')

ax.set_ylabel('Number of Players')

ax.set_xlabel('Overall')

ax.set_title('Distribution of Overall scores of Players')

plt.show()
plt.figure(figsize=(12,8)) #checking for the top 10 teams with highest overall scores in the game

ax = df.groupby(['team'])['overall'].max().sort_values(ascending = False).head(10).plot.bar(color='orangered')

ax.set_xlabel('Clubs')

ax.set_ylabel('Overall')

ax.set_title("Clubs with highest Overall scores in FIFA-2021")

plt.show()
plt.figure(figsize=(12,8)) #checking for the top 10 teams with highest potential scores of players in the game

ax = df.groupby(['team'])['potential'].max().sort_values(ascending = False).head(10).plot.bar(color='orangered')

ax.set_xlabel('Clubs')

ax.set_ylabel('Potential')

ax.set_title("Clubs with highest Potential scores of players in FIFA-2021")

plt.show()
#Top clubs from Europe's Top-5 Leagues with high overall player scores

plt.figure(figsize=(12,8))

top = ('FC Barcelona ', 'Juventus ', 'Paris Saint-Germain ', 'Real Madrid ', 'FC Bayern München ')

df2 = df.loc[df['team'].isin(top)  & df['potential'] ]



ax = sns.barplot(x=df2['team'], y=df2['potential'], palette="Set1");

ax.set_title(label='Distribution of Potential Scores in 5 Teams', fontsize=20);
#Top clubs from Europe's Top-5 Leagues with high overall player scores

plt.figure(figsize=(12,8))

top = ('FC Barcelona ', 'Juventus ', 'Paris Saint-Germain ', 'Real Madrid ', 'FC Bayern München ')

df2 = df.loc[df['team'].isin(top)  & df['overall'] ]



ax = sns.barplot(x=df2['team'], y=df2['overall'], palette="Set1");

ax.set_title(label='Distribution of Overall Scores in 5 Teams', fontsize=20);
import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

rating = pd.DataFrame(df.groupby(['nationality'])['overall'].sum().reset_index())

count = pd.DataFrame(rating.groupby('nationality')['overall'].sum().reset_index())



plot = [go.Choropleth(

            colorscale = 'inferno',

            locationmode = 'country names',

            locations = count['nationality'],

            text = count['nationality'],

            z = count['overall'],

)]



layout = go.Layout(title = 'Country vs Overall Ratings of players belonging to them')



fig = go.Figure(data = plot, layout = layout)

py.iplot(fig)
rating = pd.DataFrame(df.groupby(['nationality'])['potential'].sum().reset_index())

count = pd.DataFrame(rating.groupby('nationality')['potential'].sum().reset_index())



plot = [go.Choropleth(

            colorscale = 'inferno',

            locationmode = 'country names',

            locations = count['nationality'],

            text = count['nationality'],

            z = count['potential'],

)]



layout = go.Layout(title = 'Country vs Potential Ratings of players belonging to them')



fig = go.Figure(data = plot, layout = layout)

py.iplot(fig)
from wordcloud import WordCloud

plt.subplots(figsize=(12,8))

wordcloud = WordCloud(

                          background_color='Black',

                          width=1920,

                          height=1080

                         ).generate(" ".join(df.team))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()