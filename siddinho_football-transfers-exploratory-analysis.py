# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

plt.rcParams["figure.figsize"] = (20,10)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
transfers_dataframe = pd.read_csv('../input/top250-00-19.csv')

transfers_dataframe.tail()
print('The total number of players transferred till date  : ' + str(transfers_dataframe['Name'].nunique()))

print('The total number of unique player positions transferred till date  : ' + str(transfers_dataframe['Position'].nunique()))

print('The total number of unique teams that have transferred its players till date  : ' + str(transfers_dataframe['Team_from'].nunique()))

print('The total number of unique teams that have brought in players till date  : ' + str(transfers_dataframe['Team_to'].nunique()))

print('There is a good difference in count of selling clubs and buying clubs this means that some clubs are more interested in buying from other rather than growing academy players.')

print('Top Transfer based on Transfer Value is : ')

print(''+str(transfers_dataframe.iloc[transfers_dataframe['Transfer_fee'].idxmax()]))

top_10_transferred_players = pd.DataFrame(transfers_dataframe['Name'].value_counts())

# Based on the Number of Transfers

top_10_selling_clubs = pd.DataFrame(transfers_dataframe['Team_from'].value_counts())

top_10_selling_leagues = pd.DataFrame(transfers_dataframe['League_from'].value_counts())

top_10_buying_clubs = pd.DataFrame(transfers_dataframe['Team_to'].value_counts())

top_10_buying_leagues = pd.DataFrame(transfers_dataframe['League_to'].value_counts())

# Top Leagues and Teams as per Transfer value

top_10_selling_clubs_money = transfers_dataframe.groupby('Team_from',as_index = False)['Transfer_fee'].sum().sort_values(by = 'Transfer_fee',ascending = False)

top_10_selling_leagues_money =  transfers_dataframe.groupby('League_from',as_index = False)['Transfer_fee'].sum().sort_values(by = 'Transfer_fee',ascending = False)

top_10_buying_clubs_money =  transfers_dataframe.groupby('Team_to',as_index = False)['Transfer_fee'].sum().sort_values(by = 'Transfer_fee',ascending = False)

top_10_buying_leagues_money = transfers_dataframe.groupby('League_to',as_index = False)['Transfer_fee'].sum().sort_values(by = 'Transfer_fee',ascending = False)
# Lets inspect the above dataframes

top_10_buying_clubs_money.head(10)
plt.subplot(2,2,1)

plt.bar(x = top_10_selling_clubs.head(10).index , height =top_10_selling_clubs['Team_from'].head(10))

plt.xticks(rotation = 45)

plt.ylabel('Number of Players')

plt.title('Top Selling Clubs')



plt.subplot(2,2,2)

plt.bar(x = top_10_buying_clubs.head(10).index , height =top_10_buying_clubs['Team_to'].head(10))

plt.xticks(rotation = 45)

plt.ylabel('Number of Players')

plt.title('Top Buying Clubs')



plt.subplot(2,2,3)

plt.bar(x = top_10_selling_leagues.head(10).index , height =top_10_selling_leagues['League_from'].head(10))

plt.xticks(rotation = 45)

plt.ylabel('Number of Players')

plt.title('Top Selling Leagues')



plt.subplot(2,2,4)

plt.bar(x = top_10_buying_leagues.head(10).index , height =top_10_buying_leagues['League_to'].head(10))

plt.xticks(rotation = 45)

plt.ylabel('Number of Players')

plt.title('Top Buying Leagues')

plt.tight_layout()



plt.show()
plt.subplot(2,2,1)

plt.bar(x = top_10_selling_clubs_money['Team_from'].head(10) , height =top_10_selling_clubs_money['Transfer_fee'].head(10),color = 'green')

plt.xticks(rotation = 45)

plt.ylabel('Amount')

plt.title('Top Selling Clubs Based on Money')



plt.subplot(2,2,2)

plt.bar(x = top_10_buying_clubs_money['Team_to'].head(10) , height =top_10_selling_clubs_money['Transfer_fee'].head(10),color = 'green')

plt.xticks(rotation = 45)

plt.ylabel('Amount')

plt.title('Top Buying Clubs Based on Money')



plt.subplot(2,2,3)

plt.bar(x = top_10_selling_leagues_money['League_from'].head(10) , height =top_10_selling_leagues_money['Transfer_fee'].head(10),color = 'orange')

plt.xticks(rotation = 45)

plt.ylabel('Amount')

plt.title('Top Selling Leagues Clubs Based on Money')







plt.subplot(2,2,4)

plt.bar(x = top_10_buying_leagues_money['League_to'].head(10) , height =top_10_buying_leagues_money['Transfer_fee'].head(10),color = 'orange')

plt.xticks(rotation = 45)

plt.ylabel('Amount')

plt.title('Top Buying Leagues Clubs Based on Money')



plt.tight_layout()



plt.show()
#scatter_league = pd.merge(top_10_selling_clubs,top_10_buying_clubs,left_on =top_10_selling_clubs.index,right_on = top_10_buying_clubs 

scatter_clubs = top_10_selling_clubs.join(top_10_buying_clubs)

scatter_league = top_10_selling_leagues.join(top_10_buying_leagues)

plt.subplot(2,1,1)

plt.scatter(scatter_clubs['Team_from'],scatter_clubs['Team_to'])

plt.xlabel('Transfers Out')

plt.ylabel('Transfers In')

plt.title('Team Perspective')



plt.subplot(2,1,2)

plt.scatter(scatter_league['League_from'],scatter_league['League_to'], color  = 'red')

plt.xlabel('Transfers Out')

plt.ylabel('Transfers In')

plt.title('League Perspective')

plt.tight_layout()

plt.show()
print('Team Transfer Correlation')

print(str(scatter_clubs.corr()))
print('League Transfer Correlation ')

print(str(scatter_league.corr()))
transfers_dataframe['Age'].hist(bins = 35,color = 'green')
