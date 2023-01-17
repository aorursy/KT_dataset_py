# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
from statsmodels.stats.proportion import proportions_ztest
from sqlalchemy import create_engine
from sqlalchemy import inspect

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def OnlyNum(s,oth=''):   
    fomart = '0123456789'   
    for c in s:   
        if (c in fomart) == False:   
             s = s.replace(c,'');   
    return s
#change the format of formation form str, this is for parse formation, only numbers show
engine = create_engine('sqlite:///../input/database.sqlite')
con = engine.connect()
inspector = inspect(engine)
print(inspector.get_table_names())
con = engine.connect()
#connect sqlite
rs = con.execute('SELECT match_api_id,home_team_goal,away_team_goal,home_player_Y1,home_player_Y2,home_player_Y3,\
                home_player_Y4,home_player_Y5,home_player_Y6,home_player_Y7,home_player_Y8,home_player_Y9,home_player_Y10,home_player_Y11,\
                away_player_Y1,away_player_Y2,away_player_Y3,away_player_Y4,away_player_Y5,away_player_Y6,away_player_Y7,away_player_Y8,\
                away_player_Y9,away_player_Y10,away_player_Y11 FROM Match')

df = pd.DataFrame(rs.fetchall())
df.columns = rs.keys()
#Select data from database
df.dropna(inplace=True)
#delete non row
con.close()
home_formation = {}
away_formation = {}
#create two dict to store home team formation and away team formation, key is match_api_id, value is formation like '442'
#every row is a match, take out formation according to Y coordinates
for index,row in df.iterrows():
    home_player_y = list()
    away_player_y = list()
    #create two list to store Y coordinate for each team,and they should be empty when we start parse a new match
    
    for i in range(2,12):
        home_player_y.append(row['home_player_Y%d' % i])
        away_player_y.append(row['away_player_Y%d' % i])
    #put two teams's Y coordinates into list    
    
    c_home = Counter(home_player_y)
    c_away = Counter(away_player_y)
    #transform list to Counter objects
    formation_home = Counter(sorted(c_home.elements())).values()
    #sorted,for exsample: Y coordinate may be record with wrong, like "5,5,3,3,3,3,5,7,7,10", the correct formation should be 4321,
    #but if we don't serted it, we will get 3421
    formation_home = OnlyNum(str(formation_home))
    #get only number, like '442', but it is still a string.
    formation_away = Counter(sorted(c_away.elements())).values()
    formation_away = OnlyNum(str(formation_away))
    
    home_formation.update({row['match_api_id'] : formation_home})
    away_formation.update({row['match_api_id'] : formation_away})
    #update match id and formation to dict
df['home_formation'] = df['match_api_id'].map(home_formation)
df['away_formation'] = df['match_api_id'].map(away_formation)
#map formation to DataFrame following match_api_id
df.info()
print(df.home_formation.unique())
print(df.away_formation.unique())
set(df.home_formation.unique())-set(df.away_formation.unique())
#compare home team and away team formation , the number of home formation is no more than away formation
set(df.away_formation.unique())-set(df.home_formation.unique())
#but away formation has one special item formation '31312', lets check it
df[df['away_formation']=='31312']
#that is only one match with equalize 0:0, so I drop it.
index31312 = df[df['away_formation']=='31312'].index.get_values()
df.drop(index31312,inplace=True)
df.drop(columns=['home_player_Y1','home_player_Y2','home_player_Y3','home_player_Y4','home_player_Y5','home_player_Y6','home_player_Y7','home_player_Y8','home_player_Y9','home_player_Y10','home_player_Y11'],inplace = True)
df.drop(columns=['away_player_Y1','away_player_Y2','away_player_Y3','away_player_Y4','away_player_Y5','away_player_Y6','away_player_Y7','away_player_Y8','away_player_Y9','away_player_Y10','away_player_Y11'],inplace = True)
# delete Y coordinate, because we'dont need them anymore.
df.head()
df.info()
df['resulte']=df['home_team_goal']-df['away_team_goal']
# define resulte as win or lose(for home team)
idx = (df['resulte']<0)
# select home team lose match
df.loc[idx,['home_formation','away_formation']] = df.loc[idx,['away_formation','home_formation']].values
df.loc[idx,['home_team_goal','away_team_goal']] = df.loc[idx,['away_team_goal','home_team_goal']].values
#switch home team and away team columns, than columns 'home_team_goal' is 'win team goal' same as formation
#so we will change columns names, there are no home and away, there are only 'win','lose'
df.rename(columns = {'home_team_goal':'win_team_goal',\
                     'away_team_goal':'lose_team_goal',\
                     'home_formation':'win_formation',\
                     'away_formation':'lose_formation'},inplace=True)
df['resulte']=df['win_team_goal']-df['lose_team_goal']
dd = df['match_api_id'].groupby([df['win_formation'],df['lose_formation']]).count()
#groupby win formatin and lose formation count number of matchs
formation_match_df = dd.unstack()
formation_match_df = formation_match_df.fillna(0)
formation_match_df.astype('int64')
# we got a dataframe include diffrent formation match between each other
#this dataframe is only for observed the data,and check the groupby method, it is not for use furter.
#next step, I will separate equalize an no-equalize match to check the data.
df_equalize = df.loc[df['resulte']==0]
df_noequalize = df.loc[df['resulte']!=0]
#now I saperate equalize match and no equalize to two dataframe
dd_equalize = df_equalize['match_api_id'].groupby([df_equalize['win_formation'],df_equalize['lose_formation']]).count()
formation_match_equalize = dd_equalize.unstack()
# I found there is one formation is missing on columns which name is '5311', I insert a null column, and set a #21 column
# that is for data organized 
formation_match_equalize.insert(21,'5311',np.nan)

formation_match_equalize = formation_match_equalize.fillna(0)
formation_match_equalize.astype('int64')
#fill NaN to zero, and set all number to int
dd_noequalize = df_noequalize['match_api_id'].groupby([df_noequalize['win_formation'],df_noequalize['lose_formation']]).count()
formation_match_noequalize = dd_noequalize.unstack()

formation_match_noequalize = formation_match_noequalize.fillna(0)
formation_match_noequalize.astype('int64')
#do same thing with 
df_stats=pd.DataFrame(columns =\
                      ['A formation',\
                       'B formation',\
                       'Total Number match',\
                       'Num A win',\
                       'Num Equalize',\
                       'Num B win',\
                       'Z score A win',\
                       'P value A win',\
                       'Z score B win',\
                       'P value B win',\
                       ])
# 'Total Number match' -bettween two formation, total number of match during 2008-2016
# 'Num A win',  --------how many matchs that A formation win
# 'Num Equalize',-------how many matchs that A and B get equalize
# 'Num B win'  ---------how many matchs that B formation win
# 'Z score A win',------hypothesis test: Ho: formations A and B are no diffrent, the probability of A win is 50%,P = Po=0.5
#  'P value A win',-----Ha: formation A is more aggressivity when A meet B, the probability of A win is more than 50%, P>Po
# 'Z score B win',------another hypothesis test, Ho is same with above
# 'P value B win',------Ha: formation B is more aggressivity when B meet A.
#df_stats will give us statistc resulte, each row represanted a combination of two formation.
# we have 24 formations, I dont count combination of same formation, than total combinations will be 23+22+21+....+3+2+1 = 276,
#following are the code:
#x = 0
#for i in range(1,24):
#    x += i
#print(x)
df_stats
same_formation = 0
#same_formation is store the number of match bettween same formation, for double check if I do some missing.

#this for loop, set value to df_stats from previous dataframes
for i in formation_match_noequalize.index:
    for j in formation_match_noequalize.columns:
        if i == j:
            same_formation += formation_match_noequalize.loc[i,j]
            same_formation += formation_match_equalize.loc[i,j]
            break
            #break here, we just get number form under diagonal,and diagonal is match bettween same formation
        else:
            df_stats = df_stats.append({'A formation':i,\
                                        'B formation':j,\
                                        'Num A win':formation_match_noequalize.loc[i,j],\
                                        #A win B number of match
                                        'Num Equalize':formation_match_equalize.loc[i,j]+formation_match_equalize.loc[j,i],\
                                        #A to B and B to A are symmetric diagonally, I count both.
                                        'Num B win':formation_match_noequalize.loc[j,i]\
                                        #B win A number of match,symmetric diagonally
                                        },ignore_index=True)

df_stats['Total Number match'] = df_stats['Num A win'] + df_stats['Num B win'] + df_stats['Num Equalize']
print(df_stats['Total Number match'].sum())
print(same_formation)
# I need to check if all match number are missing, total number should be 24139
z_score_awin = {}
p_value_awin = {}
z_score_bwin = {}
p_value_bwin = {}
#
for index,row in df_stats.iterrows():
    if row['Total Number match']>=20 : 
        # if we want to make statastic, the number of sample n should satisfy the conditions 
        #which allow you to use this test, the conditions is: n*Po >=10 and n*(1-Po)>=10. we know Po = 0.5, which is mean 
        #each formation in a match has same probability to win, so Po=0.5. According to Po, we can calculate n >=20 .
        #so I consider formation match which are more than 20 times only.
        count_awin = int(row['Num A win'])+int(row['Num Equalize']/2)  #A win plus half equalize number
        #count_awin = int(row['Num A win'])
        count_bwin = int(row['Num B win'])+int(row['Num Equalize']/2)  #B win plus half equalize number
        #count_bwin = int(row['Num B win'])
        nobs = int(row['Total Number match'])                          #sample number is total match number bettween two formation
        value = 0.5                                                    # Po=0.5
        z_awin,p_awin = proportions_ztest(count_awin,nobs,0.5,alternative='larger')
        z_bwin,p_bwin = proportions_ztest(count_bwin,nobs,0.5,alternative='larger')
        z_score_awin.update({index:z_awin})                            #z score z = (p-Po)/sqrt(Po*(1-Po)/n)
        p_value_awin.update({index:p_awin})
        z_score_bwin.update({index:z_bwin})
        p_value_bwin.update({index:p_bwin})
        
df_stats['Z score A win'] = df_stats.index.map(z_score_awin)
df_stats['P value A win'] = df_stats.index.map(p_value_awin)
df_stats['Z score B win'] = df_stats.index.map(z_score_bwin)
df_stats['P value B win'] = df_stats.index.map(p_value_bwin)
df_stats
df_a_win = df_stats.loc[df_stats['P value A win']<=0.05]  # if p value is smaller than 0.05, we have enough evidence 
df_b_win = df_stats.loc[df_stats['P value B win']<=0.05]  #to reject Null Hypothesis, there is a significance of the results
df_a_win  # A formation has significance probability to win B formation
df_b_win  # B formation has significance probability to win A formation
