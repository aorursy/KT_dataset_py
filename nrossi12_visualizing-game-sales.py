# Import Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
vg_data = pd.read_csv("../input/videogamesales/vgsales.csv")
vg_data.head()
# Any record with NA in Year column will be set to the average value within Year column

vg_data.Year = vg_data.Year.fillna(vg_data.Year.mean())
# Change year column to integer data type

# Check column data types

vg_data['Year'] = vg_data['Year'].astype(int)

vg_data.dtypes
# Sum of null values within data set and to see what column they belong to.

vg_data.isnull().sum()
# Gathering Nintendo records

nintendo = vg_data.loc[vg_data['Publisher'] == 'Nintendo']

nintendo.head()
# Visualising value counts of Nintendo consoles

nintendo.Platform.value_counts().plot(kind='pie', figsize=(15, 7), autopct='%.2f')



plt.xlabel('Console')

plt.ylabel('Count of games in dataset')

plt.title('Popularity of Nintendo consoles by games in data')

plt.show()
# Total game sales per console

nintendo.groupby('Platform').sum()
nintendo.drop(columns=['Rank', 'Year']).groupby('Platform').sum().plot(kind='bar', figsize=(15, 7))



plt.title('Nintendo total game sales per console')

plt.xlabel('Console')

plt.ylabel('Sales to $1m')

plt.show()
nintendo.Genre.value_counts().plot(kind='pie', figsize=(15, 7), autopct='%.2f')



plt.title('Popularity of game genres with Nintendo')

plt.xlabel('Genre')

plt.ylabel('Amount of video games')

plt.show()
nin_rpy = nintendo.loc[nintendo['Genre']=='Role-Playing'].drop(columns=['Year', 'Publisher', 'Rank'])

nin_plt = nintendo.loc[nintendo['Genre']=='Platform'].drop(columns=['Year', 'Publisher', 'Rank'])

nin_ftg = nintendo.loc[nintendo['Genre']=='Fighting'].drop(columns=['Year', 'Publisher', 'Rank'])

nin_sht = nintendo.loc[nintendo['Genre']=='Shooter'].drop(columns=['Year', 'Publisher', 'Rank'])

nin_sim = nintendo.loc[nintendo['Genre']=='Simulation'].drop(columns=['Year', 'Publisher', 'Rank'])

nin_str = nintendo.loc[nintendo['Genre']=='Strategy'].drop(columns=['Year', 'Publisher', 'Rank'])

nin_adv = nintendo.loc[nintendo['Genre']=='Adventure'].drop(columns=['Year', 'Publisher', 'Rank'])

nin_rac = nintendo.loc[nintendo['Genre']=='Racing'].drop(columns=['Year', 'Publisher', 'Rank'])

nin_spt = nintendo.loc[nintendo['Genre']=='Sports'].drop(columns=['Year', 'Publisher', 'Rank'])

nin_puz = nintendo.loc[nintendo['Genre']=='Puzzle'].drop(columns=['Year', 'Publisher', 'Rank'])

nin_act = nintendo.loc[nintendo['Genre']=='Action'].drop(columns=['Year', 'Publisher', 'Rank'])

nin_msc = nintendo.loc[nintendo['Genre']=='Misc'].drop(columns=['Year', 'Publisher', 'Rank'])



nin_genre_list = [nin_rpy, nin_plt, nin_ftg, nin_sht, nin_sim, nin_str, nin_adv, nin_rac, nin_spt, nin_puz, nin_act, nin_msc]
nin_genre_list = [nin_rpy, nin_plt, nin_ftg, nin_sht, nin_sim, nin_sim, nin_adv, nin_rac, nin_spt, nin_spt, nin_puz, nin_act, nin_msc]

nin_gnr = ["Role-play", "Platform", "Fighting", "Shooting", "Simulation", "Strategy", "Adventure", "Racing", "Sports", "Puzzle", "Action", "Misc"]



# Plot for genre sales per Nintendo console

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15,20))



for df, ax, i in zip(nin_genre_list, axes.ravel(), nin_gnr):

    df.groupby('Platform').sum().plot(kind='bar', ax=ax)

    ax.set_title(f"Plot for Nintendo {i} genre sales per console")

    plt.tight_layout()
# Top 20 Nintendo game sales

nintendo.drop(columns=['Year', 'Rank'])[:20].set_index('Name').plot(kind='bar', figsize=(15, 7))



plt.title('Nintendo top 20 video game sales data')

plt.xlabel('Video Game')

plt.ylabel('Sales in millions')

plt.show()
vg_data.head()
# Gathering Sony records

sony = vg_data.loc[vg_data['Publisher'] == 'Sony Computer Entertainment']

sony.head()
# Visualising value counts of Sony consoles

sony.Platform.value_counts().plot(kind='pie', figsize=(15, 7), autopct='%.2f')



plt.xlabel('Console')

plt.ylabel('Count of games in dataset')

plt.title('Popularity of Sony consoles by game release')

plt.show()
# Total game sales per console

sony.groupby('Platform').sum()
sony.drop(columns=['Year', 'Rank']).groupby('Platform').sum().plot(kind='bar', figsize=(15, 7))



plt.title('Playstation total game sales per console')

plt.xlabel('Console')

plt.ylabel('Sales to $1m')

plt.show()
pd.set_option('display.max_columns', 12)

vg_data.head()
sony = vg_data.loc[vg_data['Publisher'] == 'Sony Computer Entertainment']

sony.Genre.value_counts().plot(kind='pie', figsize=(15, 7), autopct='%.2f')



plt.title('Popularity of game genres with Playstation')

plt.xlabel('Genre')

plt.ylabel('Amount of video games')

plt.show()
sony_spt = sony.loc[sony['Genre']=='Sports'].drop(columns=['Year', 'Publisher', 'Rank'])

sony_msc = sony.loc[sony['Genre']=='Misc'].drop(columns=['Year', 'Publisher', 'Rank'])

sony_puz = sony.loc[sony['Genre']=='Puzzle'].drop(columns=['Year', 'Publisher', 'Rank'])

sony_sim = sony.loc[sony['Genre']=='Simulation'].drop(columns=['Year', 'Publisher', 'Rank'])

sony_sgy = sony.loc[sony['Genre']=='Strategy'].drop(columns=['Year', 'Publisher', 'Rank'])

sony_ftg = sony.loc[sony['Genre']=='Fighting'].drop(columns=['Year', 'Publisher', 'Rank'])

sony_adv = sony.loc[sony['Genre']=='Adventure'].drop(columns=['Year', 'Publisher', 'Rank'])

sony_rpy = sony.loc[sony['Genre']=='Role-Playing'].drop(columns=['Year', 'Publisher', 'Rank'])

sony_sht = sony.loc[sony['Genre']=='Shooter'].drop(columns=['Year', 'Publisher', 'Rank'])

sony_rac = sony.loc[sony['Genre']=='Racing'].drop(columns=['Year', 'Publisher', 'Rank'])

sony_plt = sony.loc[sony['Genre']=='Platform'].drop(columns=['Year', 'Publisher', 'Rank'])

sony_act = sony.loc[sony['Genre']=='Action'].drop(columns=['Year', 'Publisher', 'Rank'])



sony_genre_list = [sony_spt, sony_msc, sony_puz, sony_sim, sony_sgy, sony_ftg, sony_adv, sony_rpy, sony_sht, sony_rac, sony_plt, sony_act]
sony_genre_list = [sony_spt, sony_msc, sony_puz, sony_sim, sony_sgy, sony_ftg, sony_adv, sony_rpy, sony_sht, sony_rac, sony_plt, sony_act]

sony_gnr = ["Sports", "Misc", "Puzzles", "Simulation", "Strategy", "Fighting", "Adventure", "Role-playing", "Shooting", "Racing", "Platform", "Action"]



# Plot for genre sales per Sony console

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15,20))



for df, ax, i in zip(sony_genre_list, axes.ravel(), sony_gnr):

    df.groupby('Platform').sum().plot(kind='bar', ax=ax)

    ax.set_title(f"Plot for Sony {i} genre sales per console")

    plt.tight_layout()
# Top 10 ps game sales

sony.drop(columns=['Year', 'Rank'])[:20].set_index('Name').plot(kind='bar', figsize=(15, 7))



plt.title('Sony Playstation top 20 video game sales data')

plt.xlabel('Video Game')

plt.ylabel('Sales in millions')

plt.show()
sony.loc[sony['Name']== 'Gran Turismo 4']
# Frequency of year values in data

vg_data.Year.value_counts().sort_index()
vg_data.Year.value_counts().sort_index().plot(kind='bar', figsize=(15, 7), grid=True, color=['orange', 'blue', 'cyan', 'green', 'yellow'])



plt.title('Dataset value counts of games per year')

plt.ylabel('Value Counts')

plt.xlabel('Year')

plt.show()
all_years = vg_data.set_index('Year').sort_values(by='Year', ascending=True)

all_years.index = pd.to_datetime(all_years.index, format='%Y')

all_years.head()
import dateutil.parser
b4_2000 = all_years[all_years.index < dateutil.parser.parse("2000-01-01")]
b4_2000.Platform.value_counts().plot(kind='bar', figsize=(15, 7))



plt.title('Most popular gaming colsoles from 1980 to 1999')

plt.xlabel('Console')

plt.ylabel('Popularity by record count')

plt.show()
aft_2000 = all_years[all_years.index >dateutil.parser.parse("2000-01-01")]
aft_2000.Platform.value_counts().plot(kind='bar', figsize=(15, 7))



plt.title('Most popular gaming colsoles from 2000 to 2020')

plt.xlabel('Console')

plt.ylabel('Popularity by record count')

plt.show()
b4_2000.drop(columns=['Rank']).groupby('Publisher').sum().sort_values(by='Global_Sales', ascending=False)[:20]
b4_2000.drop(columns=['Rank']).groupby('Publisher').sum().sort_values(by='Global_Sales', ascending=False)[:20].plot(kind='bar', figsize=(15, 7))



plt.title('Publishing company sales in total, between 2000 and 2020')

plt.xlabel('Publisher')

plt.ylabel('Sales in $1m\'s')

plt.show()
aft_2000.head()
aft_2000.groupby('Publisher').sum().sort_values(by='Global_Sales', ascending=False)[:20]
aft_2000.drop(columns=['Rank']).groupby('Publisher').sum().sort_values(by='Global_Sales', ascending=False)[:20].plot(kind='bar', figsize=(15, 7))



plt.title('Publishing company sales in total, between 2000 and 2020')

plt.xlabel('Publisher')

plt.ylabel('Sales in $1m\'s')

plt.show()
vg_data.head()
# Top 20 sales in NA compared to EU by publisher

na_eu = vg_data.groupby(['Publisher'])[['NA_Sales', 'EU_Sales']].sum().sort_values(by=['NA_Sales'], ascending=True)[-20:]

na_eu
na_eu.plot(kind='bar', figsize=(15, 7))



plt.xlabel('Publisher')

plt.ylabel('Sales in millions')

plt.title('North America compared to Europe sales in data')

plt.show()
vg_data.head()
# Top 20 sales in JAPAN compared to OTHER by publisher

jp_ot = vg_data.groupby(['Publisher'])[['JP_Sales', 'Other_Sales']].sum().sort_values(by=['JP_Sales'], ascending=True)[-20:]

jp_ot
jp_ot.plot(kind='bar', figsize=(15, 7))



plt.xlabel('Publisher')

plt.ylabel('Sales in millions')

plt.title('Japan compared to Other sales in data')

plt.show()