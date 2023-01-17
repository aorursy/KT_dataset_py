# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
%matplotlib inline
#importing data
df = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')
df.info()
df.isnull().sum()
#use missingno to graphicaly observe the missing data(optional)
msno.matrix(df)
#since the % of empty data is small we drop them
empty_data = np.where(pd.isnull(df))
df = df.drop(empty_data[0])
df.isnull().sum()
df = df.drop(['Rank'], axis=1).reset_index()
df = df.drop(['index'], axis=1)
df['Year']=df['Year'].astype(int)
#Saving changes in df into df1 in order to be able to restore the changes when needed
df1=df.copy()
df1
genre_sum =df1.groupby(['Genre']).agg({'sum'}).reset_index()
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(genre_sum['Genre'],
        genre_sum['NA_Sales'],
        color='r',
        label='NA_Sales');
ax.plot(genre_sum['Genre'],
        genre_sum['EU_Sales'],
        color='b',
        label='EU_Sales');
ax.plot(genre_sum['Genre'],
        genre_sum['JP_Sales'],
        color='gainsboro',
        label='JP_Sales');
ax.plot(genre_sum['Genre'],
        genre_sum['Other_Sales'],
        color='k',
        label='Other_Sales');

ax.set_title('Sales per Genre except JP')
plt.xlabel('Genre')
plt.ylabel('Sales (M)')
ax.legend();
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(genre_sum['Genre'],
        genre_sum['NA_Sales'],
        color='grey',
        label='NA_Sales');
ax.plot(genre_sum['Genre'],
        genre_sum['EU_Sales'],
        color='gainsboro',
        label='EU_Sales');
ax.plot(genre_sum['Genre'],
        genre_sum['JP_Sales'],
        color='r',
        label='JP_Sales');
ax.plot(genre_sum['Genre'],
        genre_sum['Other_Sales'],
        color='gainsboro',
        label='Other_Sales');

ax.set_title('Sales per Genre JP')
plt.xlabel('Genre')
plt.ylabel('Sales (M)')
ax.legend();
year_sum =df1.groupby(['Year']).agg({'sum'}).reset_index()
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(year_sum['Year'],
        year_sum['NA_Sales'],
        color='r',
        label='NA_Sales');
ax.plot(year_sum['Year'],
        year_sum['EU_Sales'],
        color='b',
        label='EU_Sales');
ax.plot(year_sum['Year'],
        year_sum['JP_Sales'],
        color='g',
        label='JP_Sales');
ax.plot(year_sum['Year'],
        year_sum['Other_Sales'],
        color='k',
        label='Other_Sales');

ax.set_title('Sales per Realese Year')
plt.xlabel('Release Year')
plt.ylabel('Sales (M)')
ax.legend();

data_games = {'Year':list(range(1980, 2021))}
games_count = pd.DataFrame(data_games, columns = ['Year'])
games_published=[119,154,280,382,324,306, 381,439,397,483,590,637,668,370,735,793,672,645,655,695,700,691,709,718,679,792,857,899,983,939,814,798,779,660,606,586,570,515,390,342,83]
gc=games_count['Year']
for i in gc:
    games_count['Games_published']=games_published

games_count['Sales']=year_sum['Global_Sales']
games_count = games_count.dropna()
games_count
games_count["Games_published"] = games_count["Games_published"] / games_count["Games_published"].max()
games_count["Sales"] = games_count["Sales"] / games_count["Sales"].max()
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(games_count['Year'],
        games_count['Sales'],
        color='g',
        label='Sales');
ax.plot(games_count['Year'],
        games_count['Games_published'],
        color='r',
        label='Published Games');
ax.set_title('Sales vs Realese per Year')
plt.xlabel('Release Year')
plt.ylabel('Q')
ax.legend();
df2 = df1.copy()
df2=df2.sort_values(by='Global_Sales', ascending = False)

df3 = df2.copy()

short_names=df3['Name'].apply(lambda x: ' '.join(x.split()[:3]))
df3['Name']=short_names
for i in df3['Name']:
    df3['count'] = 1
    
df3=df3.groupby(['Name']).agg({'sum'}).copy()

df4=df3.copy()
columns=['Platform', 'Year','Genre','Publisher','NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales','count']
df4.columns=columns
df4['Game Tilte']=df4.index
df4['Global_Sales']=df4['Global_Sales'].round(decimals=2)
large=df4.nlargest(10,['Global_Sales'],keep='first') 
fig = plt.figure(figsize=(15,6))
ax = fig.add_axes([0,0,1,1])
game_title = large['Game Tilte']
sales = large['Global_Sales']
ax.bar(game_title,sales)
xlabel=('Games Title'),
ylabel=('Sales (M)')
ax.set_title('Top 10 sold Game Series')
# Make some labels.
rects = ax.patches
labels = sales

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 2, label,
            ha='center', va='bottom')

plt.show()
legend_cols = [col for col in df['Name'] if 'The Legend of' in col]
print(legend_cols)
large=large.drop(large.index[2])
large = large.rename(index={'Call of Duty:': 'Call of Duty', 'Grand Theft Auto:':'Grand Theft Auto', 'Super Mario Bros.':'Super Mario', \
                            'New Super Mario':'Super Mario', 'Need for Speed:':'Need for Speed' })
large2 =large.groupby(['Name']).agg({'sum'})
large3 = large2.copy()
large3.columns=['Platform', 'Year','Genre','Publisher','NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales','count', 'Game_Title']
large3 = large3.sort_values(by='Global_Sales', ascending=False)
large3['Game_Title'] = large3.index
fig = plt.figure(figsize=(15,6))
ax = fig.add_axes([0,0,1,1])
game_title = large3['Game_Title']
sales = large3['Global_Sales']
ax.bar(game_title,sales)
xlabel=('Games Title'),
ylabel=('Sales (M)')
ax.set_title('Top 6 sold Game Series')
# Make some labels.
rects = ax.patches
labels = sales

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 2, label,
            ha='center', va='bottom')

plt.show()
fig, ax = plt.subplots(figsize=(18, 9))
ax.plot(large3['Game_Title'],
        large3['NA_Sales'],
        color='r',
        label='NA_Sales');
ax.plot(large3['Game_Title'],
        large3['EU_Sales'],
        color='c',
        label='EU_Sales');
ax.plot(large3['Game_Title'],
        large3['JP_Sales'],
        color='b',
        label='JP_Sales');
ax.plot(large3['Game_Title'],
        large3['Other_Sales'],
        color='k',
        label='Other_Sales');

ax.set_title('Game Series Sales per Market')
plt.xlabel('Game Series Title')
plt.ylabel('Sales (M)')
ax.legend();
large3['Sales_per1'] = large3['Global_Sales']/large3['count']
large4= large3.sort_values(by='Sales_per1', ascending=False).copy()
fig = plt.figure(figsize=(15,6))
ax = fig.add_axes([0,0,1,1])
game_title = large4['Game_Title']
sales = large4['Sales_per1']
ax.bar(game_title,sales)
xlabel=('Games Title'),
ylabel=('Sales (M)')
ax.set_title('Top Average Sales per 1 Game from series')
# Make some labels.
rects = ax.patches
labels = sales

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 1, label,
            ha='center', va='bottom')

plt.show()
Shooter = df1[df1.Genre == 'Shooter']
Shooter =Shooter.groupby(['Year']).agg({'sum'}).reset_index()
Shooter=Shooter[['Year','Name' ,'Global_Sales']]

Misc = df1[df1.Genre == 'Misc']
Misc =Misc.groupby(['Year']).agg({'sum'}).reset_index()
Misc=Misc[['Year','Name' , 'Global_Sales']]

Action = df1[df1.Genre == 'Action']
Action =Action.groupby(['Year']).agg({'sum'}).reset_index()
Action=Action[['Year','Name' , 'Global_Sales']]

Sports = df1[df1.Genre == 'Sports']
Sports =Sports.groupby(['Year']).agg({'sum'}).reset_index()
Sports=Sports[['Year','Name' , 'Global_Sales']]

Fighting = df1[df1.Genre == 'Fighting']
Fighting =Fighting.groupby(['Year']).agg({'sum'}).reset_index()
Fighting=Fighting[['Year','Name' , 'Global_Sales']]

Puzzle = df1[df1.Genre == 'Puzzle']
Puzzle =Puzzle.groupby(['Year']).agg({'sum'}).reset_index()
Puzzle=Puzzle[['Year','Name' , 'Global_Sales']]

Racing = df1[df1.Genre == 'Racing']
Racing =Racing.groupby(['Year']).agg({'sum'}).reset_index()
Racing=Racing[['Year','Name' , 'Global_Sales']]

Platform = df1[df1.Genre == 'Platform']
Platform =Platform.groupby(['Year']).agg({'sum'}).reset_index()
Platform=Platform[['Year','Name' , 'Global_Sales']]

Simulation = df1[df1.Genre == 'Simulation']
Simulation =Simulation.groupby(['Year']).agg({'sum'}).reset_index()
Simulation=Simulation[['Year','Name' , 'Global_Sales']]

Adventure = df1[df1.Genre == 'Adventure']
Adventure =Adventure.groupby(['Year']).agg({'sum'}).reset_index()
Adventure=Adventure[['Year','Name' , 'Global_Sales']]

Role_Playing = df1[df1.Genre == 'Role-Playing']
Role_Playing =Role_Playing.groupby(['Year']).agg({'sum'}).reset_index()
Role_Playing=Role_Playing[['Year','Name' , 'Global_Sales']]

Strategy = df1[df1.Genre == 'Strategy']
Strategy =Strategy.groupby(['Year']).agg({'sum'}).reset_index()
Strategy=Strategy[['Year','Name' , 'Global_Sales']]
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(Shooter['Year'],
        Shooter['Global_Sales'],
        color='r',
        label='Shooter');
ax.plot(Misc['Year'],
        Misc['Global_Sales'],
        color='b',
        label='Misc');
ax.plot(Action['Year'],
        Action['Global_Sales'],
        color='g',
        label='Action');
ax.plot(Sports['Year'],
        Sports['Global_Sales'],
        color='c',
        label='Sports');
ax.plot(Strategy['Year'],
        Strategy['Global_Sales'],
        color='m',
        label='Strategy');
ax.plot(Fighting['Year'],
        Fighting['Global_Sales'],
        color='y',
        label='Fighting');
ax.plot(Puzzle['Year'],
        Puzzle['Global_Sales'],
        color='k',
        label='Puzzle');
ax.plot(Racing['Year'],
        Racing['Global_Sales'],
        color='tab:orange',
        label='Racing');
ax.plot(Platform['Year'],
        Platform['Global_Sales'],
        color='tab:green',
        label='Platform');
ax.plot(Simulation['Year'],
        Simulation['Global_Sales'],
        color='tab:brown',
        label='Simulation');
ax.plot(Adventure['Year'],
        Adventure['Global_Sales'],
        color='tab:purple',
        label='Adventure');
ax.plot(Role_Playing['Year'],
        Role_Playing['Global_Sales'],
        color='tab:olive',
        label='Role_Playing');
ax.set_title('Sales per Genre by Realese Year')
plt.xlabel('Release Year')
plt.ylabel('Sales (M)')
ax.legend();
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(Shooter['Year'],
        Shooter['Global_Sales'],
        color='whitesmoke',
        label='Shooter');
ax.plot(Misc['Year'],
        Misc['Global_Sales'],
        color='whitesmoke',
        label='Misc');
ax.plot(Action['Year'],
        Action['Global_Sales'],
        color='whitesmoke',
        label='Action');
ax.plot(Sports['Year'],
        Sports['Global_Sales'],
        color='c',
        label='Sports');
ax.plot(Strategy['Year'],
        Strategy['Global_Sales'],
        color='whitesmoke',
        label='Strategy');
ax.plot(Fighting['Year'],
        Fighting['Global_Sales'],
        color='whitesmoke',
        label='Fighting');
ax.plot(Puzzle['Year'],
        Puzzle['Global_Sales'],
        color='whitesmoke',
        label='Puzzle');
ax.plot(Racing['Year'],
        Racing['Global_Sales'],
        color='whitesmoke',
        label='Racing');
ax.plot(Platform['Year'],
        Platform['Global_Sales'],
        color='whitesmoke',
        label='Platform');
ax.plot(Simulation['Year'],
        Simulation['Global_Sales'],
        color='whitesmoke',
        label='Simulation');
ax.plot(Adventure['Year'],
        Adventure['Global_Sales'],
        color='whitesmoke',
        label='Adventure');
ax.plot(Role_Playing['Year'],
        Role_Playing['Global_Sales'],
        color='whitesmoke',
        label='Role_Playing');
ax.set_title('Sales per Genre')
plt.xlabel('Release Year')
plt.ylabel('Sales (M)')
ax.legend();

Sports1=Sports.copy()
Sports1.columns=['Year', 'Name', 'Global_Sales']
Sports1=Sports1.sort_values(by='Global_Sales', ascending=False)
Sports1.nlargest(5,'Global_Sales')
df5=df1.copy()

#top 5 sold games in Sprts genre in 2006.
df5=df5[df5.Genre == 'Sports']
y1 = df5.loc[df['Year'] == 2006]
y2006=y1.nlargest(5,['Global_Sales'],keep='first') 
y2006
y2 = df5.loc[df['Year'] == 2009]
y2009=y2.nlargest(5,['Global_Sales'],keep='first') 
y2009
wii1 = y1.loc[df['Platform'] == 'Wii']
wii1_sum = wii1['Global_Sales'].sum()
wii1_sum
wii2 = y2.loc[df['Platform'] == 'Wii']
wii2_sum = wii2['Global_Sales'].sum()
wii2_sum
wii2.info()
wii2.head(10)
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(Shooter['Year'],
        Shooter['Global_Sales'],
        color='r',
        label='Shooter');
ax.plot(Misc['Year'],
        Misc['Global_Sales'],
        color='b',
        label='Misc');
ax.plot(Action['Year'],
        Action['Global_Sales'],
        color='indigo',
        label='Action');
ax.plot(Sports['Year'],
        Sports['Global_Sales'],
        color='c',
        label='Sports');
ax.plot(Strategy['Year'],
        Strategy['Global_Sales'],
        color='whitesmoke',
        label='Strategy');
ax.plot(Fighting['Year'],
        Fighting['Global_Sales'],
        color='whitesmoke',
        label='Fighting');
ax.plot(Puzzle['Year'],
        Puzzle['Global_Sales'],
        color='whitesmoke',
        label='Puzzle');
ax.plot(Racing['Year'],
        Racing['Global_Sales'],
        color='lightgrey',
        label='Racing');
ax.plot(Platform['Year'],
        Platform['Global_Sales'],
        color='whitesmoke',
        label='Platform');
ax.plot(Simulation['Year'],
        Simulation['Global_Sales'],
        color='lightgrey',
        label='Simulation');
ax.plot(Adventure['Year'],
        Adventure['Global_Sales'],
        color='whitesmoke',
        label='Adventure');
ax.plot(Role_Playing['Year'],
        Role_Playing['Global_Sales'],
        color='whitesmoke',
        label='Role_Playing');
ax.set_title('Sales per Genre per Realese Year')
plt.xlabel('Release Year')
plt.ylabel('Sales (M)')
ax.legend();