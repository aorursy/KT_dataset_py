import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import sqlite3

import numpy as np

import xml.etree.ElementTree as ET



with sqlite3.connect('../input/database.sqlite') as con:    

    matches = pd.read_sql_query("SELECT * from Match WHERE goal IS NOT NULL", con)

    leagues = pd.read_sql_query("SELECT * from League", con)      

    

matches = matches[['id', 'goal', 'home_team_api_id', 'away_team_api_id', 'league_id']]

print(type(matches))
factors = {}

for index, row in leagues.iterrows():

    factors[row['id']] = []

    

for index, row in matches.iterrows():

    goals = ET.fromstring(row['goal']).findall('value')    

    dict = { row['home_team_api_id'] : 0, row['away_team_api_id'] : 0 } 

    

    ind = True

    timeInd = 0

    timeDec = 0

    eLastGoal = 0

    

    for g in goals:

        if (g.find("team") is  not None):

            t = int(g.find("team").text)

            e = int(g.find('elapsed').text)

        

            if ( abs(dict[row['away_team_api_id']] - dict[row['home_team_api_id']]) < 2):

                timeInd = timeInd + (e - eLastGoal)

            else:

                timeDec = timeDec + (e - eLastGoal)



            dict[t] = dict[t]+1               

            eLastGoal = e

    

    if ( abs(dict[row['away_team_api_id']] - dict[row['home_team_api_id']]) < 2):

        timeInd = timeInd + (90 - eLastGoal)

    else:

        timeDec = timeDec + (90 - eLastGoal)

    

    indFact = timeInd * 100 / 90    

    factors[row['league_id']].append(indFact)   
res = {}

print('In %age of time spent with a score gap less than 2 goals')

for index, row in leagues.iterrows():

    s = sum(factors[row['id']])

    l = float(len(factors[row['id']]))    

    if (l != 0 ):

        avg = round(s / l, 2)

        res[row['name']] = avg

        print(row['name']+ ': ' +  str(avg) )
N = len(res)

b = len(res.values())



menMeans = res.values()



ind = np.arange(N)  # the x locations for the groups

width = 0.35       # the width of the bars



fig, ax = plt.subplots()

rects1 = ax.bar(ind, menMeans, width)



ax.set_title('Indecisiveness')



ax.set_xticklabels(res.keys())



ax.legend((rects1), ('Men'))





def autolabel(rects):

    # attach some text labels

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,

                '%d' % int(height),

                ha='center', va='bottom')



autolabel(rects1)



plt.setp(plt.xticks()[1], rotation=30)



plt.show()