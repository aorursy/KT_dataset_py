import numpy as np 

import pandas as pd 

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
points2008 = pd.read_csv('../input/ipl-points-table-2008-to-2019/IPL Points Table/IPL 2008 PT.csv')

points2009 = pd.read_csv('../input/ipl-points-table-2008-to-2019/IPL Points Table/IPL 2009 PT.csv')

points2010 = pd.read_csv('../input/ipl-points-table-2008-to-2019/IPL Points Table/IPL 2010 PT.csv')

points2011 = pd.read_csv('../input/ipl-points-table-2008-to-2019/IPL Points Table/IPL 2011 PT.csv')

points2012 = pd.read_csv('../input/ipl-points-table-2008-to-2019/IPL Points Table/IPL 2012 PT.csv')

points2013 = pd.read_csv('../input/ipl-points-table-2008-to-2019/IPL Points Table/IPL 2013 PT.csv')

points2014 = pd.read_csv('../input/ipl-points-table-2008-to-2019/IPL Points Table/IPL 2014 PT.csv')

points2015 = pd.read_csv('../input/ipl-points-table-2008-to-2019/IPL Points Table/IPL 2015 PT.csv')

points2016 = pd.read_csv('../input/ipl-points-table-2008-to-2019/IPL Points Table/IPL 2016 PT.csv')

points2017 = pd.read_csv('../input/ipl-points-table-2008-to-2019/IPL Points Table/IPL 2017 PT.csv')

points2018 = pd.read_csv('../input/ipl-points-table-2008-to-2019/IPL Points Table/IPL 2018 PT.csv')

points2019 = pd.read_csv('../input/ipl-points-table-2008-to-2019/IPL Points Table/IPL 2019 PT.csv')
points2010
point=points2008.append([points2009,points2010,points2011,points2012,points2013,points2014,points2015,points2016,points2018,points2017,points2019])
points = point.groupby('Team').mean()
matches = pd.read_csv('../input/ipl-cricket-dataset/matches1234.csv')
match = matches.drop(['city','venue','umpire1','win_by_runs','win_by_wickets','season'],axis=1)
def string_remover(df,list1=[],list2=[],drop=[],exclude=[],include=[]):

    a = df.select_dtypes(include='object')

    for i in a.columns:

        list1=[]

        d=0

        if not i in exclude and i in include:

            list1.append(i)

            for x in a.index:

                try:

                    c = list1.index(a[i].iloc[x:x+1][x])

                    a[i].iloc[x:x+1][x] = c

                except:

                    list1.append(a[i].iloc[x:x+1][x])

                    a[i].iloc[x:x+1][x] = len(list1)-1

            list2.append(list1)

            d+=1

    a.fillna(len(list1))

    d = df.select_dtypes(exclude='object').fillna(0)

    try:

        return pd.concat([d,a],axis=1).fillna(0).drop([drop],axis=1)

    except:

        return pd.concat([d,a],axis=1).fillna(0)
match['win']=None

match['toss_win']=None

for i in match.index:

    if match.winner[i] == match.team1[i]:

        match['win'][i] = 'team 1'

    else:

        match['win'][i] = 'team 2'    

    if match.toss_winner[i] == match.team1[i]:

        match['toss_win'][i] = 'team 1'

    else:

        match['toss_win'][i] = 'team 2'    
a=[]
b=['toss_decision', 'field', 'bat', 'toss_win', 'team 1', 'team 2']
match
match = string_remover(match,include=['toss_decision','toss_win'],list1=b,list2 = a)
b
match
points.drop('Tied',inplace=True,axis=1)
points.index = ['Chennai Super Kings', 'Deccan Chargers', 'Delhi Capitals',

       'Gujarat Lions', 'Kings XI Punjab', 'Kochi Tuskers Kerala',

       'Kolkata Knight Riders', 'Mumbai Indians',

       'Pune Warriors India', 'Rajasthan Royals',

       'Rising Pune Supergiants', 'Royal Challengers Bangalore',

       'Sunrisers Hyderabad']
points.index.name = 'Team'
for i in [match[match['team2']=='Delhi Daredevils'].index,match[match['team2']=='Pune Warriors'].index]:

    for i in i:

        match.drop(i,inplace=True)
for i in [match[match['team1']=='Delhi Daredevils'].index,match[match['team1']=='Pune Warriors'].index]:

    for i in i:

        match.drop(i,inplace=True)
points.to_csv('IPL_Points_Table.csv')
match['Team1_pts'] = None

match['Team1_RR'] = None

match['Team2_pts'] = None

match['Team2_RR'] = None

for i in match.index:

    match['Team1_pts'][i] = points.Pts[match.team1[i]]

    match['Team1_RR'][i] = points['Net RR'][match.team1[i]]

    match['Team2_pts'][i] = points.Pts[match.team2[i]]

    match['Team2_RR'][i] = points['Net RR'][match.team2[i]]

for i in match.index:

    match['Team2_pts'][i] = points.Pts[match.team2[i]]

    match['Team2_RR'][i] = points['Net RR'][match.team2[i]]
points
RC = RandomForestClassifier(random_state=815)
match.win = match['win']
match[['Team1_pts','Team1_RR','Team2_pts','Team2_RR',]]
RC.fit(X=match[['Team1_pts','Team1_RR','Team2_pts','Team2_RR','toss_decision','toss_win']],y=match.win)
def Predicter(team1,team2,toss_winner,toss_decision):

    Team1_pts = points.Pts[team1]

    Team1_RR = points['Net RR'][team1]

    Team2_pts = points.Pts[team2]

    Team2_RR = points['Net RR'][team2]

    pred = RC.predict([[Team1_pts,Team1_RR,Team2_pts,Team2_RR,toss_decision,toss_winner]])

    if pred == 'team 1':

        return team1

    else:

        return team2
Predicter('Chennai Super Kings','Mumbai Indians',toss_decision=1,toss_winner=2)