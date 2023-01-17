import sqlite3 as lite

import pandas as pd

from sklearn.neighbors import NearestNeighbors

import numpy as np

import matplotlib.pyplot as plt

import xml.etree.ElementTree as etree

%matplotlib inline



database = '../input/database.sqlite'

conn = lite.connect(database)

leagues = pd.read_sql_query("SELECT * from League", conn)



#select teams consistenty in Premier League

teams = pd.read_sql_query("SELECT * from Team", conn)

                  #ARS   CHE   EVE   LIV   MANC  MANU   STOKE  TOT

selected_teams = [9825, 8455, 8668, 8650, 8456, 10260, 10194, 8586]

teams = teams[teams.team_api_id.isin(selected_teams)]

team_Attributes = pd.read_sql_query("SELECT * from team_Attributes", conn)

    
matches = pd.read_sql_query("SELECT home_team_api_id, away_team_api_id, season, shoton from Match WHERE league_id=1729", conn)

matchesHome = matches[matches.home_team_api_id.isin(selected_teams)]

selectMatches = matchesHome[matchesHome.away_team_api_id.isin(selected_teams)]



matches_train = selectMatches[selectMatches.season != "2015/2016"]

matches_test = selectMatches[selectMatches.season == "2015/2016"]



print(selectMatches)

print(selectMatches[:1].shoton.to_string)

print(selectMatches[:2].shoton.to_string)







#for i in range(1,2):

    #myString = selectMatches[:i].shoton.to_string

    #print(myString)

    #root = etree.fromstring(myString)







fig, axs = plt.subplots(1, 3, sharey=True)

teams.plot(kind='scatter', x='id', y='', ax=axs[0], figsize=(16, 8))

teams.plot(kind='scatter', x='id', y='', ax=axs[1])

teams.plot(kind='scatter', x='id', y='', ax=axs[2])



from scipy.stats import entropy







#plot graph

ax = entropy_means.plot(figsize=(12,8),marker='o')



#set title

plt.title('Leagues Predictability', fontsize=16)



#set ticks roatation

plt.xticks(rotation=50)



#keep colors for next graph

colors = [x.get_color() for x in ax.get_lines()]

colors_mapping = dict(zip(leagues.id,colors))



#remove x label

ax.set_xlabel('')



#locate legend 

plt.legend(loc='lower left')



#add arrows

ax.annotate('', xytext=(7.2, 1),xy=(7.2, 1.039),

            arrowprops=dict(facecolor='black',arrowstyle="->, head_length=.7, head_width=.3",linewidth=1), annotation_clip=False)



ax.annotate('', xytext=(7.2, 0.96),xy=(7.2, 0.921),

            arrowprops=dict(facecolor='black',arrowstyle="->, head_length=.7, head_width=.3",linewidth=1), annotation_clip=False)



ax.annotate('less predictable', xy=(7.3, 1.028), annotation_clip=False,fontsize=14,rotation='vertical')

ax.annotate('more predictable', xy=(7.3, 0.952), annotation_clip=False,fontsize=14,rotation='vertical')
from scipy.stats import entropy



def match_entropy(column):



    odds = VCH.loc[column['VCH'],column['VCD'],column['VCA']]



    #change odds to probability



    probs = [1/o for o in odds]



    #normalize to sum to 1



    norm = sum(probs)



    probs = [p/norm for p in probs]



    return entropy(probs)



matchesHome['entropy'] = matchesHome.apply(match_entropy,axis=1)
x = np.linspace(0, 5, 11)

y = x ** 2

plt.plot(x, y, 'r') # 'r' is the color red

plt.xlabel('X Axis Title Here')

plt.ylabel('Y Axis Title Here')

plt.title('String Title Here')

plt.show()
colors = np.random.rand(N)

area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radiuses



plt.scatter(x, y, s=area, c=colors, alpha=0.5)

plt.show()