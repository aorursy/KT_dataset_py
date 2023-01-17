import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

!pip install pywaffle --quiet

from pywaffle import Waffle

from wordcloud import WordCloud
df= pd.read_csv("../input/videogamesales/vgsales.csv")
df
df.info()
df.describe()
print("A detailed description of the dataset ")

d = df.describe().T

d
print('Insights obtained from the dataset are as follows :')

print("1. MEAN NORTH AMERICA SALES =",d.iloc[2,1])

print("1. MEAN EUROPE SALES =",d.iloc[3,1])

print("1. MEAN JAPAN SALES =",d.iloc[4,1])

print("1. MEAN OTHER SALES =",d.iloc[5,1])

print("1. MEAN GLOBAL SALES =",d.iloc[6,1])
print("Number of games: ", len(df))

publishers = df['Publisher'].unique()

print("Number of publishers: ", len(publishers))

platforms = df['Platform'].unique()

print("Number of platforms: ", len(platforms))

genres =df['Genre'].unique()

print("Number of genres: ", len(genres))
df.isnull().sum()
df=df.dropna()

data=df
data_genre = df.groupby(by=['Genre'])['Global_Sales'].sum().reset_index().sort_values(by=['Global_Sales'], ascending=False)
plt.figure(figsize=(15, 10))

ax=sns.barplot(x="Genre", y="Global_Sales", data=data_genre)

plt.xticks(rotation=90)

for p in ax.patches:

    ax.annotate(int(p.get_height()), (p.get_x()+0.25, p.get_height()+1), va='bottom',

                    color= 'black')

plt.show()
publisher = df.loc[:,['Publisher','Global_Sales']]

publisher['total_sales'] = publisher.groupby('Publisher')['Global_Sales'].transform('sum')

publisher.drop('Global_Sales', axis=1, inplace=True)

publisher = publisher.drop_duplicates()

publisher = publisher.head(10)



fig = px.pie(publisher, names='Publisher', values='total_sales', template='seaborn')

fig.update_traces(rotation=90, pull=[0.2,0.1,0.1,0.1,0.1], textinfo="percent+label")

fig.show()
GP=df[["Platform", "Global_Sales"]].groupby(['Platform'], as_index=False).mean().sort_values(by='Global_Sales', ascending=False)

plt.figure(figsize=(20,8))



sns.barplot(x='Platform', y='Global_Sales', data=GP)
game = data.loc[data['Name']!='Wii Sports',['Name','EU_Sales']]

game = game.sort_values('EU_Sales', ascending=False)

game = game.head()



fig = px.pie(game, names='Name', values='EU_Sales', template='seaborn')

fig.update_traces(rotation=90, pull=0.06, textinfo="percent+label")

fig.show()
game = data.loc[data['Name']!='Wii Sports',['Name','NA_Sales']]

game = game.sort_values('NA_Sales', ascending=False)

game = game.head()



fig = px.pie(game, names='Name', values='NA_Sales', template='seaborn')

fig.update_traces(rotation=90, pull=0.06, textinfo="percent+label")

fig.show()
game = data.loc[data['Name']!='Wii Sports',['Name','JP_Sales']]

game = game.sort_values('JP_Sales', ascending=False)

game = game.head()



fig = px.pie(game, names='Name', values='JP_Sales', template='seaborn')

fig.update_traces(rotation=90, pull=0.06, textinfo="percent+label")

fig.show()
game = data.loc[data['Name']!='Wii Sports',['Name','Global_Sales']]

game = game.sort_values('Global_Sales', ascending=False)

game = game.head()



fig = px.pie(game, names='Name', values='Global_Sales', template='seaborn')

fig.update_traces(rotation=90, pull=0.06, textinfo="percent+label")

fig.show()
platform = data.loc[data['Name']!='Wii Sports',['Platform','NA_Sales']]

platform['total_sales'] = platform.groupby('Platform')['NA_Sales'].transform('sum')

platform.drop('NA_Sales', axis=1, inplace=True)

platform = platform.drop_duplicates()

platform = platform.sort_values('total_sales', ascending=False)

platform = platform.head()



fig = px.pie(platform, names='Platform', values='total_sales', template='seaborn')

fig.update_traces(rotation=90, pull=0.06, textinfo="percent+label")

fig.show()
platform = data.loc[data['Name']!='Wii Sports',['Platform','EU_Sales']]

platform['total_sales'] = platform.groupby('Platform')['EU_Sales'].transform('sum')

platform.drop('EU_Sales', axis=1, inplace=True)

platform = platform.drop_duplicates()

platform = platform.sort_values('total_sales', ascending=False)

platform = platform.head()



fig = px.pie(platform, names='Platform', values='total_sales', template='seaborn')

fig.update_traces(rotation=90, pull=0.06, textinfo="percent+label")

fig.show()
platform = data.loc[data['Name']!='Wii Sports',['Platform','JP_Sales']]

platform['total_sales'] = platform.groupby('Platform')['JP_Sales'].transform('sum')

platform.drop('JP_Sales', axis=1, inplace=True)

platform = platform.drop_duplicates()

platform = platform.sort_values('total_sales', ascending=False)

platform = platform.head()



fig = px.pie(platform, names='Platform', values='total_sales', template='seaborn')

fig.update_traces(rotation=90, pull=0.06, textinfo="percent+label")

fig.show()
platform = data.loc[data['Name']!='Wii Sports',['Platform','Global_Sales']]

platform['total_sales'] = platform.groupby('Platform')['Global_Sales'].transform('sum')

platform.drop('Global_Sales', axis=1, inplace=True)

platform = platform.drop_duplicates()

platform = platform.sort_values('total_sales', ascending=False)

platform = platform.head()



fig = px.pie(platform, names='Platform', values='total_sales', template='seaborn')

fig.update_traces(rotation=90, pull=0.06, textinfo="percent+label")

fig.show()
publisher = data.loc[data['Name']!='Wii Sports',['Publisher','NA_Sales']]

publisher['total_sales'] = publisher.groupby('Publisher')['NA_Sales'].transform('sum')

publisher.drop('NA_Sales', axis=1, inplace=True)

publisher = publisher.drop_duplicates()

publisher = publisher.sort_values('total_sales', ascending=False)

publisher = publisher.head()



fig = px.pie(publisher, names='Publisher', values='total_sales', template='seaborn')

fig.update_traces(rotation=90, pull=0.06, textinfo="percent+label")

fig.show()
publisher = data.loc[data['Name']!='Wii Sports',['Publisher','EU_Sales']]

publisher['total_sales'] = publisher.groupby('Publisher')['EU_Sales'].transform('sum')

publisher.drop('EU_Sales', axis=1, inplace=True)

publisher = publisher.drop_duplicates()

publisher = publisher.sort_values('total_sales', ascending=False)

publisher = publisher.head()



fig = px.pie(publisher, names='Publisher', values='total_sales', template='seaborn')

fig.update_traces(rotation=90, pull=0.06, textinfo="percent+label")

fig.show()
publisher = data.loc[data['Name']!='Wii Sports',['Publisher','JP_Sales']]

publisher['total_sales'] = publisher.groupby('Publisher')['JP_Sales'].transform('sum')

publisher.drop('JP_Sales', axis=1, inplace=True)

publisher = publisher.drop_duplicates()

publisher = publisher.sort_values('total_sales', ascending=False)

publisher = publisher.head()



fig = px.pie(publisher, names='Publisher', values='total_sales', template='seaborn')

fig.update_traces(rotation=90, pull=0.06, textinfo="percent+label")

fig.show()
publisher = data.loc[data['Name']!='Wii Sports',['Publisher','Global_Sales']]

publisher['total_sales'] = publisher.groupby('Publisher')['Global_Sales'].transform('sum')

publisher.drop('Global_Sales', axis=1, inplace=True)

publisher = publisher.drop_duplicates()

publisher = publisher.sort_values('total_sales', ascending=False)

publisher = publisher.head()



fig = px.pie(publisher, names='Publisher', values='total_sales', template='seaborn')

fig.update_traces(rotation=90, pull=0.06, textinfo="percent+label")

fig.show()
EU = df.pivot_table('EU_Sales', columns='Name', index='Year', aggfunc='sum').sum(axis=1)

NA = df.pivot_table('NA_Sales', columns='Name', index='Year', aggfunc='sum').sum(axis=1)

JP = df.pivot_table('JP_Sales', columns='Name', index='Year', aggfunc='sum').sum(axis=1)

Other = df.pivot_table('Other_Sales', columns='Name', index='Year', aggfunc='sum').sum(axis=1)

years = Other.index.astype(int)

regions = ['European Union','Japan','North America','Other']



plt.figure(figsize=(12,8))

ax = sns.pointplot(x=years, y=EU, color='mediumslateblue', scale=0.7)

ax = sns.pointplot(x=years, y=NA, color='cornflowerblue', scale=0.7)

ax = sns.pointplot(x=years, y=JP, color='orchid', scale=0.7)

ax = sns.pointplot(x=years, y=Other, color='thistle', scale=0.7)

ax.set_xticklabels(labels=years, fontsize=12, rotation=50)

ax.set_xlabel(xlabel='Year', fontsize=16)

ax.set_ylabel(ylabel='Revenue in $ Millions', fontsize=16)

ax.set_title(label='Distribution of Total Revenue Per Region by Year in $ Millions', fontsize=20)

ax.legend(handles=ax.lines[::len(years)+1], labels=regions, fontsize=18)

plt.show();
top_publisher = data.groupby(by=['Publisher'])['Year'].count().sort_values(ascending=False).head(20)

top_publisher = pd.DataFrame(top_publisher).reset_index()
plt.figure(figsize=(15, 10))

ax=sns.countplot(x="Publisher", data=data, order = data.groupby(by=['Publisher'])['Year'].count().sort_values(ascending=False).iloc[:20].index)

plt.xticks(rotation=90)

for p in ax.patches:

    ax.annotate(int(p.get_height()), (p.get_x()+0.25, p.get_height()+1), va='bottom',

                    color= 'black')
data = df.groupby(['Publisher']).sum()['Global_Sales']

data = pd.DataFrame(data.sort_values(ascending=False))[0:20]

publishers = data.index

data.columns = ['Global Sales']



colors = sns.color_palette("cool", len(data))

plt.figure(figsize=(12,8))

ax = sns.barplot(y = publishers , x = 'Global Sales', data=data, orient='h', palette=colors)

ax.set_xlabel(xlabel='Revenue in $ Millions', fontsize=16)

ax.set_ylabel(ylabel='Publisher', fontsize=16)

ax.set_title(label='Top 10 Total Publisher Game Revenue', fontsize=20)

ax.set_yticklabels(labels = publishers, fontsize=14)

plt.show();
x = df.groupby(['Year']).count()

x = x['Global_Sales']

y = x.index.astype(int)



plt.figure(figsize=(12,8))

colors = sns.color_palette("muted")

ax = sns.barplot(y = y, x = x, orient='h', palette=colors)

ax.set_xlabel(xlabel='Number of releases', fontsize=16)

ax.set_ylabel(ylabel='Year', fontsize=16)

ax.set_title(label='Game Releases Per Year', fontsize=20)

plt.show();
y = df.groupby(['Year']).sum()

y = y['Global_Sales']

x = y.index.astype(int)



plt.figure(figsize=(12,8))

ax = sns.barplot(y = y, x = x)

ax.set_xlabel(xlabel='$ Millions', fontsize=16)

ax.set_xticklabels(labels = x, fontsize=12, rotation=50)

ax.set_ylabel(ylabel='Year', fontsize=16)

ax.set_title(label='Game Sales in $ Millions Per Year', fontsize=20)

plt.show();
EU = df.pivot_table('EU_Sales', columns='Publisher', aggfunc='sum').T

EU = EU.sort_values(by='EU_Sales', ascending=False).iloc[0:3]

EU_publishers = EU.index



JP = df.pivot_table('JP_Sales', columns='Publisher', aggfunc='sum').T

JP = JP.sort_values(by='JP_Sales', ascending=False).iloc[0:3]

JP_publishers = JP.index



NA = df.pivot_table('NA_Sales', columns='Publisher', aggfunc='sum').T

NA = NA.sort_values(by='NA_Sales', ascending=False).iloc[0:3]

NA_publishers = NA.index



Other = df.pivot_table('Other_Sales', columns='Publisher', aggfunc='sum').T

Other = Other.sort_values(by='Other_Sales', ascending=False).iloc[0:3]

Other_publishers = Other.index



colors =  {'Nintendo':sns.xkcd_rgb["teal blue"], 'Electronic Arts':sns.xkcd_rgb["denim blue"], 'Activision':sns.xkcd_rgb["dark lime green"], 'Namco Bandai Games':sns.xkcd_rgb["pumpkin"], 'Konami Digital Entertainment':sns.xkcd_rgb["burnt umber"], 'Sony Computer Entertainment':sns.xkcd_rgb["yellow orange"]}

fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(1,4,1)

ax1.set_xticklabels(labels = EU_publishers, rotation=90, size=14)

sns.barplot(x=EU_publishers, y=EU['EU_Sales'], palette=colors)

plt.title('European Union', size=18)

plt.ylabel('Revenue in $ Millions', size=16)



ax2 = fig.add_subplot(1,4,2)

ax2.set_xticklabels(labels = JP_publishers, rotation=90, size=14)

sns.barplot(x=JP_publishers, y=JP['JP_Sales'], palette=colors)

plt.title('Japan', size=18)



ax3 = fig.add_subplot(1,4,3)

ax3.set_xticklabels(labels = NA_publishers, rotation=90, size=14)

sns.barplot(x=NA_publishers, y=NA['NA_Sales'], palette=colors)

plt.title('North America', size=18)



ax4 = fig.add_subplot(1,4,4)

ax4.set_xticklabels(labels = Other_publishers, rotation=90, size=14)

sns.barplot(x=Other_publishers, y=Other['Other_Sales'], palette=colors)

plt.title('Other', size=18)

plt.suptitle('Top 3 Publishers by Revenue Per Region', size=22)

plt.show();
plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='Black',

                          width=1920,

                          height=1080

                         ).generate(" ".join(df.Publisher))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('cast.png')

plt.show()