# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import networkx as nx

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv('../input/Shakespeare_data.csv')
data.info()
data['Player'].replace(np.nan, 'Other',inplace = True)

data.head(5)
print("Number of plays are: " + str(data['Play'].nunique()))
pd.DataFrame(data['Play'].unique().tolist(), columns=['Play Name'])
numberPlayers = data.groupby(['Play'])['Player'].nunique().sort_values(ascending= False).to_frame()

numberPlayers['Play'] = numberPlayers.index.tolist()

numberPlayers.columns = ['Num Players','Play']

numberPlayers.index= np.arange(0,len(numberPlayers))

numberPlayers



plt.figure(figsize=(10,10))

ax = sns.barplot(x='Num Players',y='Play',data=numberPlayers)

ax.set(xlabel='Number of Players', ylabel='Play Name')

plt.show()

data.groupby('Play').count().sort_values(by='Player-Line',ascending=False)['Player-Line']
#converting the results above to a dataframe.

play_data = data.groupby('Play').count().sort_values(by='Player-Line',ascending=False)['Player-Line']

play_data = play_data.to_frame()

play_data['Play'] = play_data.index.tolist()

play_data.index = np.arange(0,len(play_data)) #changing the index from plays to numbers

play_data.columns =['Lines','Play']

play_data
plt.figure(figsize=(10,10))

ax= sns.barplot(x='Lines',y='Play',data=play_data, order = play_data['Play'])

ax.set(xlabel='Number of Lines', ylabel='Play Name')

plt.show()
data.groupby(['Play','Player']).count()['Player-Line']
lines_per_player= data.groupby(['Play','Player']).count()['Player-Line']

lines_per_player= lines_per_player.to_frame()

lines_per_player
play_name = data['Play'].unique().tolist()

for play in play_name:

    p_line = data[data['Play']==play].groupby('Player').count().sort_values(by='Player-Line',ascending=False)['Player-Line']

    p_line = p_line.to_frame()

    p_line['Player'] = p_line.index.tolist()

    p_line.index = np.arange(0,len(p_line))

    p_line.columns=['Lines','Player']

    plt.figure(figsize=(10,10))

    ax= sns.barplot(x='Lines',y='Player',data=p_line)

    ax.set(xlabel='Number of Lines', ylabel='Player')

    plt.title(play,fontsize=30)

    plt.show()

g= nx.Graph()
g = nx.from_pandas_dataframe(data,source='Play',target='Player')
print (nx.info(g))
plt.figure(figsize=(40,40)) 

nx.draw_networkx(g,with_labels=True,node_size=100)

plt.show()
centralMeasures = pd.DataFrame(nx.degree_centrality(g),index=[0]).T

centralMeasures.columns=['Degree Centrality']

centralMeasures['Page Rank']= pd.DataFrame(nx.pagerank(g),index=[0]).T

centralMeasures['Name']= centralMeasures.index.tolist()

centralMeasures.index = np.arange(0,len(centralMeasures))

centralMeasures
#Centrality measures only for players (or actors)

centralMeasures[centralMeasures['Name'].isin(data['Player'].unique().tolist())].sort_values(by='Degree Centrality',ascending=False)
#Centrality measures only for players (or actors)

centralMeasures[centralMeasures['Name'].isin(data['Player'].unique().tolist())].sort_values(by='Page Rank',ascending=False)
#number of nodes that the "Messenger" is connected to across all plays

len(g.neighbors('Messenger'))
#getting the number of lines a messanger spoke across all the plays

data[data['Player']=='Messenger']['Player-Line'].count()
centralMeasures[centralMeasures['Name'].isin(data['Play'].unique().tolist())].sort_values(by='Degree Centrality',ascending=False)
centralMeasures[centralMeasures['Name'].isin(data['Play'].unique().tolist())].sort_values(by='Page Rank',ascending=False)
#number of nodes that "Richard III" is connected to

len(g.neighbors('Richard III'))