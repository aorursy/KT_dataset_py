# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import networkx as nx



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



ts_data = pd.read_csv('/kaggle/input/nba-enhanced-stats/2012-18_teamBoxScore.csv')

#for str in ts_data.keys():

#    print(str)













# Any results you write to the current directory are saved as output.
def convert(s): 

  

    # initialization of string to "" 

    new = "" 

  

    # traverse in the string  

    for x in s: 

        new += x  

  

    # return string  

    return new
def cutYear(lst):

    tmp= convert([lst[0:4]])

    tmp1=int(tmp)

    return tmp1
def cutMonth(lst):

    tmp= convert([lst[5:7]])

    tmp1=int(tmp)

    return tmp1
def intTOstr(t):

    return str(t)
def getSeason(date):

    year=cutYear(date)

    month=cutMonth(date)

    ans=year

    if month<10:

        ans-=1

    return ans
def getwinner(games):

    games['win'] = games['teamPTS']>games['opptPTS']

    games['winner'] = np.where(games['win'],games['teamAbbr'],games['opptAbbr'])

    games['looser'] = np.where(games['win'],games['opptAbbr'],games['teamAbbr'])

    games['diff'] = abs(games['teamPTS']-games['opptPTS'])

    return games[['winner','looser','diff']]
def plot_degree_dist(G):

    degrees = [G.degree(n) for n in G.nodes()]

    plt.hist(degrees)

    plt.show()
games = ts_data[['gmDate','seasTyp','teamConf','teamDiv','teamAbbr','teamPTS','teamLoc','opptConf','opptDiv','opptAbbr','opptPTS']]

games = games[games.teamLoc=='Home']

games = games[games.seasTyp=='Regular']

games = games[['gmDate','seasTyp','teamConf','teamDiv','teamAbbr','teamPTS','opptConf','opptDiv','opptAbbr','opptPTS']]

games['Season'] = games.gmDate.apply(getSeason)

games = games[['Season','teamAbbr','teamConf','teamDiv','teamPTS','opptAbbr','opptConf','opptDiv','opptPTS']]





season12_13 = games[games.Season==2012]

season13_14 = games[games.Season==2013]

season14_15 = games[games.Season==2014]

season15_16 = games[games.Season==2015]

season16_17 = games[games.Season==2016]

season17_18 = games[games.Season==2017]



season12_13 = getwinner(season12_13[['teamAbbr','teamConf','teamDiv','teamPTS','opptAbbr','opptConf','opptDiv','opptPTS']])

season13_14 = getwinner(season13_14[['teamAbbr','teamConf','teamDiv','teamPTS','opptAbbr','opptConf','opptDiv','opptPTS']])

season14_15 = getwinner(season14_15[['teamAbbr','teamConf','teamDiv','teamPTS','opptAbbr','opptConf','opptDiv','opptPTS']])

season15_16 = getwinner(season15_16[['teamAbbr','teamConf','teamDiv','teamPTS','opptAbbr','opptConf','opptDiv','opptPTS']])

season16_17 = getwinner(season16_17[['teamAbbr','teamConf','teamDiv','teamPTS','opptAbbr','opptConf','opptDiv','opptPTS']])

season17_18 = getwinner(season17_18[['teamAbbr','teamConf','teamDiv','teamPTS','opptAbbr','opptConf','opptDiv','opptPTS']])



study_set= getwinner(games[games.Season<2017])

test_set = getwinner(games[games.Season==2017])

#centrality_ratings_table = study_set.groupby(['winner']).sum()

#sorted_centrality_ratings_table = centrality_ratings_table.sort_values(by=['diff'],ascending=False)

#team_west_or_east_table = (games[['teamAbbr', 'teamConf']]).drop_duplicates()

#team_west_or_east_table = team_west_or_east_table.rename(columns={"teamAbbr" : "winner"})

#sorted_centrality_ratings_table = sorted_centrality_ratings_table.merge(team_west_or_east_table,on ='winner') 

#west_playoff_teams = sorted_centrality_ratings_table[sorted_centrality_ratings_table['teamConf']=="West"][0:8]

#east_playoff_teams = sorted_centrality_ratings_table[sorted_centrality_ratings_table['teamConf']=="East"][0:8]

#play0ff_teams = pd.concat([west_playoff_teams,east_playoff_teams])

#sorted_centrality_ratings_table

#play0ff_teams
centrality_in_ratings_table = study_set.groupby(['looser']).count()

centrality_in_ratings_table=centrality_in_ratings_table.sort_values(by=['diff'],ascending=False)

#print(centrality_in_ratings_table)

def getout(abbr):

    redundent=study_set#[study_set['diff']>10]

    GA = nx.from_pandas_edgelist(redundent,'winner','looser',['diff'],create_using=nx.MultiDiGraph())

    b = nx.out_degree_centrality(GA)

    #b = sorted(b.items(), key=lambda x:x[1])

    return b[abbr]



redundent=study_set#[study_set['diff']>10]

GA = nx.from_pandas_edgelist(redundent,'winner','looser',['diff'],create_using=nx.MultiDiGraph())

b = nx.out_degree_centrality(GA)

b = sorted(b.items(), key=lambda x:x[1])

b

centrality_ratings_table = study_set.groupby(['winner']).sum()

sorted_centrality_ratings_table = centrality_ratings_table.sort_values(by=['diff'],ascending=False)

team_west_or_east_table = (games[['teamAbbr', 'teamConf']]).drop_duplicates()

team_west_or_east_table = team_west_or_east_table.rename(columns={"teamAbbr" : "winner"})

team_west_or_east_table['out_centerality']=team_west_or_east_table.winner.apply(getout)

sorted_centrality_ratings_table = sorted_centrality_ratings_table.merge(team_west_or_east_table,on ='winner') 

west_playoff_teams = sorted_centrality_ratings_table[sorted_centrality_ratings_table['teamConf']=="West"][0:8]

east_playoff_teams = sorted_centrality_ratings_table[sorted_centrality_ratings_table['teamConf']=="East"][0:8]

play0ff_teams = pd.concat([west_playoff_teams,east_playoff_teams])

sorted_centrality_ratings_table

play0ff_teams
centrality_in_ratings_table = test_set.groupby(['looser']).count()

centrality_in_ratings_table=centrality_in_ratings_table.sort_values(by=['diff'],ascending=False)

#print(centrality_in_ratings_table)

def getout(abbr):

    redundent=test_set#[study_set['diff']>10]

    GA = nx.from_pandas_edgelist(redundent,'winner','looser',['diff'],create_using=nx.MultiDiGraph())

    b = nx.out_degree_centrality(GA)

    #b = sorted(b.items(), key=lambda x:x[1])

    return b[abbr]



redundent=test_set#[study_set['diff']>10]

GA = nx.from_pandas_edgelist(redundent,'winner','looser',['diff'],create_using=nx.MultiDiGraph())

b = nx.out_degree_centrality(GA)

b = sorted(b.items(), key=lambda x:x[1])

centrality_ratings_table = test_set.groupby(['winner']).sum()

sorted_centrality_ratings_table = centrality_ratings_table.sort_values(by=['diff'],ascending=False)

team_west_or_east_table = (games[['teamAbbr', 'teamConf']]).drop_duplicates()

team_west_or_east_table = team_west_or_east_table.rename(columns={"teamAbbr" : "winner"})

team_west_or_east_table['out_centerality']=team_west_or_east_table.winner.apply(getout)

sorted_centrality_ratings_table = sorted_centrality_ratings_table.merge(team_west_or_east_table,on ='winner') 

west_playoff_teams = sorted_centrality_ratings_table[sorted_centrality_ratings_table['teamConf']=="West"][0:8]

east_playoff_teams = sorted_centrality_ratings_table[sorted_centrality_ratings_table['teamConf']=="East"][0:8]

play0ff_teams = pd.concat([west_playoff_teams,east_playoff_teams])

sorted_centrality_ratings_table

play0ff_teams