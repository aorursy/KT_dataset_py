import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline
auction = pd.read_excel('../input/ipl-auction-and-ipl-dataset/Auction.xlsx')
auction_2020 = pd.read_csv('../input/ipl-auction-dataset/IPL Data.csv')
batsmen = pd.read_excel('../input/ipl-auction-and-ipl-dataset/Top_100_batsman.xlsx')
bowlers = pd.read_excel('../input/ipl-auction-and-ipl-dataset/Top_100_bowlers.xlsx')
auction.head()
auction_2020.head()
batsmen.head()
bowlers.head()
auction
auction.isnull().sum()
auction_2020
auction_2020.isnull().sum()
auction_2020['IPL Matches'].fillna(0, inplace = True)
auction_2020['IPL 2019 Team'].fillna('Not Played', inplace = True)
auction_2020['IPL Team(s)'].fillna('No Previous clubs', inplace = True)
auction_2020.drop(['Country'],axis=1, inplace= True)
auction_2020
auction_2020.isnull().sum()
batsmen.isnull().sum()
bowlers.isnull().sum()
top_10 = auction_2020.sort_values('Auctioned Price(in ₹ Lacs)',ascending = False).head(10)
plt.xticks(rotation=90)

plt.xlabel('Player Name')

plt.ylabel('Price in Rs. Lacs')

plt.bar( data = top_10, x= 'Name', height = 'Auctioned Price(in ₹ Lacs)',color = (0.3,0.5,0.4,0.6),width = 0.8,align = 'edge',label = 'Auctioned price' )

plt.bar(data = top_10,x ='Name',height = 'Reserve Price(in ₹ Lacs)',color = (0.3,0.1,0.4,0.6), width =0.4, align = 'center', label = 'Reserved Price')

plt.legend()
auction_2020['IPL 2020 Team'].value_counts()
plt.xticks(rotation=90)

sns.countplot('IPL 2020 Team',data=auction_2020)

plt.xlabel('Team Name')

plt.ylabel('No of players bought')
top_10_history = auction.sort_values('PRICE(In crore Indian Rupees) ', ascending = False).head(10)


plt.xticks(rotation=90)

plt.xlabel('Player Name')

plt.ylabel('Price in Rs. Lacs')

plt.bar(x='PLAYER', height = 'PRICE(In crore Indian Rupees) ',data = top_10_history)
auction_2020['Playing Role'].value_counts().plot(kind='pie',autopct= '%0.1f%%')

auction_2020['Capped / Uncapped /Associate'].value_counts().plot(kind='pie',autopct='%.1f%%',radius=2)

circle = plt.Circle(xy=(0,0), radius=0.5, color='white')

plt.gca().add_artist(circle)
auction_2020['Reserve Price(in ₹ Lacs)'].value_counts().plot(kind='bar')

plt.xlabel('price in Rs.Lakhs')

plt.ylabel('No. of players')
sns.countplot(data=auction_2020,x='Reserve Price(in ₹ Lacs)')

plt.xlabel('Base price in lakhs')

plt.ylabel('No of players with that base price')
most_runs =batsmen.nlargest(10, ['Runs'])
plt.xticks(rotation=45)

sns.barplot(data = most_runs,x='PLAYER',y='Runs')
strike_rate = batsmen.nlargest(10,['SR'])

plt.xticks(rotation=45)

sns.barplot(data = strike_rate,x='PLAYER',y='SR')

plt.xlabel('Player Name')

plt.ylabel('Strike Rate')
avg = batsmen.nlargest(10,['Avg'])
plt.barh(data=avg,y='PLAYER',width='Avg',color='crb')

plt.ylabel("Player Name")

plt.xlabel('Average')
sns.distplot(batsmen.Avg)
sns.boxplot(batsmen.SR)

plt.xlabel('Strike Rate')
sixes = batsmen.nlargest(10,'6s')


sns.catplot(x='PLAYER',y='6s',data=sixes)

plt.xticks(rotation=90)

plt.xlabel('Player name')

plt.ylabel('sixes count')
fours = batsmen.sort_values(by = '4s', ascending = False).head(10)
sns.stripplot(data= fours, x= 'PLAYER',y='4s')

plt.xticks(rotation=90)

plt.xlabel('Player Name')

plt.ylabel('No. of Fours')
highest_wickets = bowlers.nlargest(10,'Wkts')
sns.barplot(x='PLAYER',y='Wkts',data=highest_wickets)

plt.xticks(rotation=90)

plt.xlabel('Player Name')

plt.ylabel('No. of wickets')
economy = bowlers.sort_values(by = 'Econ', ascending = True).head(10)
sns.swarmplot(x='PLAYER',y='Econ',data = economy)

plt.xticks(rotation = 90)

plt.xlabel('Player Name')

plt.ylabel('Economy Rate')
bowler_sr = bowlers.sort_values(by = 'SR').head(10)
sns.scatterplot(x=bowler_sr.PLAYER, y = bowler_sr.SR)

plt.xticks(rotation = 90)

plt.xlabel('Player Name')

plt.ylabel('Strike Rate')
bowler_avg = bowlers.sort_values(by = 'Avg').head(10)
sns.stripplot(bowler_avg.PLAYER,bowler_avg.Avg)

plt.xticks(rotation=90)

plt.xlabel('Player Name')

plt.ylabel('Average')
wickets_4 = bowlers.nlargest(10,'4w')
sns.pointplot(x='PLAYER', y='4w',data = wickets_4)

plt.xticks(rotation=90)

plt.xlabel('Player Name')

plt.ylabel('No. of 4 wicket haul')
wickets_5 = bowlers.nlargest(10,'5w')
sns.lineplot(x='PLAYER', y='5w',data = wickets_5)

plt.xticks(rotation=90)

plt.xlabel('Player Name')

plt.ylabel('No. of 5 wikcet haul')
all_players = pd.concat((batsmen,bowlers))
all_players
all_players.fillna('Not applicable', inplace = True)
all_players
all_players.drop_duplicates(subset ="PLAYER", 

                     keep = 'first',inplace= True) 
all_players
all_players.nlargest(10,'Mat').plot(kind='bar',x='PLAYER',y='Mat')

plt.xlabel('Player Name')

plt.ylabel('Match count')