import pandas as pd#data manipulation
pd.set_option('display.max_columns', None)
import numpy as np # mathematical operations
import scipy as sci # math ops
import seaborn as sns # visualizations
import matplotlib.pyplot as plt # for plottings
%matplotlib inline 
#Suppressing filters
import warnings
warnings.filterwarnings("ignore")
teams = pd.read_csv("../input/Teams.csv")
teams.head()
seasons = pd.read_csv("../input/Seasons.csv")
seasons.head()
seed = pd.read_csv("../input/NCAATourneySeeds.csv")
seed.tail()
#Separating seed into no and region
seed['region'] = seed['Seed'].apply(lambda x: x[0])
seed['num'] = seed['Seed'].apply(lambda x: x[1:])
seed.head()
rscr = pd.read_csv('../input/RegularSeasonCompactResults.csv')
rscr.head()
rscr.shape
print("percentage of games where Overtime was played = ",round(len(rscr[rscr['NumOT']>0])/len(rscr)*100,4))
(rscr['WLoc'].value_counts()/len(rscr))
ntcr = pd.read_csv('../input/NCAATourneyCompactResults.csv')
ntcr.head()
years = sorted(list(set(ntcr['Season'])))
year_team_dict = {}
for i in years:
    year_team_dict[str(i)] = list(set(list(set(ntcr[ntcr['Season'] ==i]['WTeamID'])) + list(set(ntcr[ntcr['Season'] ==i]['LTeamID']))))
rsdr = pd.read_csv('../input/RegularSeasonDetailedResults.csv')
rsdr.head()
set(rsdr['Season'])
len(set(rsdr[(rsdr['Season'] ==2018)]['LTeamID']))
#2 point throws for winning team
rsdr['WFGM2'] = rsdr['WFGM'] - rsdr['WFGM3']
rsdr['WFGA2'] = rsdr['WFGA'] - rsdr['WFGA3']
#2 point throws for losing teamm
rsdr['LFGM2'] = rsdr['LFGM'] - rsdr['LFGM3']
rsdr['LFGA2'] = rsdr['LFGA'] - rsdr['LFGA3']
rsdr['rr'] = (rsdr['WOR'] + rsdr['WDR']) / (rsdr['LOR'] + rsdr['LDR'])
rsdr.shape
rsdr.head()
clms=['Score','EScore','FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','Ast','TO','Stl','Blk','PF','FGM2','FGA2']
#Creating empty dataframes
df_2003 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2003]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2003]['WTeamID'])))
df_2004 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2004]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2004]['WTeamID'])))
df_2005 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2005]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2005]['WTeamID'])))
df_2006 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2006]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2006]['WTeamID'])))
df_2007 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2007]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2007]['WTeamID'])))
df_2008 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2008]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2008]['WTeamID'])))
df_2009 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2009]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2009]['WTeamID'])))
df_2010 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2010]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2010]['WTeamID'])))
df_2011 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2011]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2011]['WTeamID'])))
df_2012 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2012]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2012]['WTeamID'])))
df_2013 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2013]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2013]['WTeamID'])))
df_2014 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2014]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2014]['WTeamID'])))
df_2015 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2015]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2015]['WTeamID'])))
df_2016 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2016]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2016]['WTeamID'])))
df_2017 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2017]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2017]['WTeamID'])))
df_2018 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2018]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2017]['WTeamID'])))
#Merging these data frames in a list
df_list = [df_2003,df_2004,df_2005,df_2006,df_2007,df_2008,df_2009,df_2010,df_2011,df_2012,df_2013,df_2014,df_2015,df_2016,df_2017,df_2018]
%%time
#Taking statistics for each team for each year
year = 2003
for m in df_list:
    for i in list(set(rsdr[rsdr['Season'] ==year]['LTeamID'])):
        klm = pd.DataFrame()
        klm = rsdr[(rsdr['Season']==year)&((rsdr['WTeamID'] ==i)|(rsdr['LTeamID'] ==i))]
        for j in clms:
            if j=='EScore':
                m.loc[i,j] = (klm[klm['WTeamID'] == i]['LScore'].values.sum() + klm[klm['LTeamID'] == i]['WScore'].values.sum() ) / len(klm)
            else:
                m.loc[i,j] = (klm[klm['WTeamID'] == i]['W'+j].values.sum() + klm[klm['LTeamID'] == i]['L'+j].values.sum() ) / len(klm)
    year = year + 1
df_2018.head()
#Assigning zeros to seed values
df_2003['Seed'],df_2004['Seed'],df_2005['Seed'],df_2006['Seed'],df_2007['Seed'],df_2008['Seed'],df_2009['Seed'],df_2010['Seed'],df_2011['Seed'],df_2012['Seed'],df_2013['Seed'],df_2014['Seed'],df_2015['Seed'],df_2016['Seed'],df_2017['Seed'],df_2018['Seed']=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
#Taking the statistics only for tournament games
year = 2003
r = 0
for i in df_list:
    m = i.loc[pd.Series(seed[seed['Season'] ==year]['TeamID']).sort_values(ascending=True),:]
    for j in list(m.index):
        m.loc[j,'Seed'] = list(seed[(seed['Season'] ==year)&(seed['TeamID'] ==j)]['Seed'])[0]
    df_list[r] = m    
    year = year + 1
    r = r+1
df_list[0].head()
df_list[14].shape
#Splitting seed column into 2: no and region
k = 0
for i in df_list:
    i['Seed_no'] = i['Seed'].map(lambda x: int(x[1:3]))
    i['Seed_region'] = i['Seed'].map(lambda x: x[0])
    df_list[k] = i
    k = k+1
#Deleting seed column
k = 0
for i in df_list:
    del i['Seed']
    df_list[k] = i
    k = k + 1
#We are using the below for loops because some teams only won and didn't lose in regular season. It caused some errors.
#Therefore I assigned for 2 special rows in 2014 for team 1455 and in 2015 for team 1246
for i in clms:
    if i =='EScore':
        df_list[11].loc[1455,i] = rsdr[(rsdr['Season'] ==2014)&(rsdr['WTeamID'] ==1455)]['LScore'].mean()
    else:
        df_list[11].loc[1455,i] = rsdr[(rsdr['Season'] ==2014)&(rsdr['WTeamID'] ==1455)]['W'+i].mean()
for i in clms:
    if i =='EScore':
        df_list[12].loc[1246,i] = rsdr[(rsdr['Season'] ==2015)&(rsdr['WTeamID'] ==1246)]['LScore'].mean()
    else:
        df_list[12].loc[1246,i] = rsdr[(rsdr['Season'] ==2015)&(rsdr['WTeamID'] ==1246)]['W'+i].mean()
#Putting the diff between average scored points and average eaten points.
for m in df_list:
    m['diff'] = m['Score'] - m['EScore']
df_list[-1].head()



ntdr = pd.read_csv('../input/NCAATourneyDetailedResults.csv')
ntdr.head()
ntdr.shape
#Create output column indicating which team win
def winning(data):
    if data['WTeamID'] < data['LTeamID']:
        return 1
    else:
        return 0
ntdr['winning'] = ntdr.apply(winning,axis=1)
#2 point throws
ntdr['WFGM2'] = ntdr['WFGM'] - ntdr['WFGM3']
ntdr['WFGA2'] = ntdr['WFGA'] - ntdr['WFGA3']
#2 point throws
ntdr['LFGM2'] = ntdr['LFGM'] - ntdr['LFGM3']
ntdr['LFGA2'] = ntdr['LFGA'] - ntdr['LFGA3']
#Creating lists for column names
f_columns = ['f_'+str(x) for x in list(df_list[14].columns.values)]
s_columns = ['s_'+str(x) for x in list(df_list[14].columns.values)]
t_columns = f_columns + s_columns
#Creating empty data frame for past ncaa tourney matches
nc_2003 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2003].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2003][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2004 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2004].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2004][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2005 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2005].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2005][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2006 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2006].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2006][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2007 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2007].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2007][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2008 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2008].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2008][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2009 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2009].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2009][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2010 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2010].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2010][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2011 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2011].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2011][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2012 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2012].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2012][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2013 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2013].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2013][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2014 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2014].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2014][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2015 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2015].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2015][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2016 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2016].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2016][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2017 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2017].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2017][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2017.shape
nc_2017.head()
#Merging dataframes in a list
nc_list = [nc_2003,nc_2004,nc_2005,nc_2006,nc_2007,nc_2008,nc_2009,nc_2010,nc_2011,nc_2012,nc_2013,nc_2014,nc_2015,nc_2016,nc_2017]
len(nc_list)
%%time
r = 0
for i in nc_list:
    for j in range(len(i)):
        for m in range(len(f_columns)):
            i.iloc[j,m] = df_list[r].loc[min(i.loc[j,'LTeamID'],i.loc[j,'WTeamID']),f_columns[m][2:]]
        for m in range(len(f_columns),len(f_columns)*2):
            i.iloc[j,m] = df_list[r].loc[max(i.loc[j,'LTeamID'],i.loc[j,'WTeamID']),s_columns[m-len(f_columns)][2:]]
    nc_list[r] = i
    r = r+1
#Reading data
coaches= pd.read_csv('../input/TeamCoaches.csv')
coaches.tail()
#Creating lists for teamsID for each year
l_2003 = list(set(coaches[coaches['Season'] ==2003]['TeamID']))
l_2004 = list(set(coaches[coaches['Season'] ==2004]['TeamID']))
l_2005 = list(set(coaches[coaches['Season'] ==2005]['TeamID']))
l_2006 = list(set(coaches[coaches['Season'] ==2006]['TeamID']))
l_2007 = list(set(coaches[coaches['Season'] ==2007]['TeamID']))
l_2008 = list(set(coaches[coaches['Season'] ==2008]['TeamID']))
l_2009 = list(set(coaches[coaches['Season'] ==2009]['TeamID']))
l_2010 = list(set(coaches[coaches['Season'] ==2010]['TeamID']))
l_2011 = list(set(coaches[coaches['Season'] ==2011]['TeamID']))
l_2012 = list(set(coaches[coaches['Season'] ==2012]['TeamID']))
l_2013 = list(set(coaches[coaches['Season'] ==2013]['TeamID']))
l_2014 = list(set(coaches[coaches['Season'] ==2014]['TeamID']))
l_2015 = list(set(coaches[coaches['Season'] ==2015]['TeamID']))
l_2016 = list(set(coaches[coaches['Season'] ==2016]['TeamID']))
l_2017 = list(set(coaches[coaches['Season'] ==2017]['TeamID']))
#l_2018 = list(set(coaches[coaches['Season'] ==2018]['TeamID']))
l_list = [l_2003,l_2004,l_2005,l_2006,l_2007,l_2008,l_2009,l_2010,l_2011,l_2012,l_2013,l_2014,l_2015,l_2016,l_2017]
#Creating empty dicts and merging them
d_2003,d_2004,d_2005,d_2006,d_2007,d_2008,d_2009,d_2010,d_2011,d_2012,d_2013,d_2014,d_2015,d_2016,d_2017={},{},{},{},{},{},{},{},{},{},{},{},{},{},{}
d_list = [d_2003,d_2004,d_2005,d_2006,d_2007,d_2008,d_2009,d_2010,d_2011,d_2012,d_2013,d_2014,d_2015,d_2016,d_2017]
%%time
#The below code calculates which team has its coach for how many years
year = 2003
r = 0
for i in l_list:
    for j in i:
        temp = coaches[(coaches['Season']==year)&(coaches['TeamID'] ==j)]
        coach = temp.tail(1)['CoachName']
        coach = list(coach)[0]
        yr = len(coaches[(coaches['Season'] <= year) & (coaches['CoachName'] ==coach)&(coaches['TeamID'] ==j)])
        d_list[r][str(j)] =yr
    r = r+1
    year = year+1
%%time
k=0
for i in d_list:
    tempo = pd.Series(list(i.values()),index= list( i.keys() ) )
    nc_list[k]['f_coach'],nc_list[k]['s_coach'] = 0,0
    for j in range(len(nc_list[k])):
        nc_list[k]['f_coach'][j] = tempo[str(min(nc_list[k].iloc[j]['WTeamID'],nc_list[k].iloc[j]['LTeamID']))]
        nc_list[k]['s_coach'][j] = tempo[str(max(nc_list[k].iloc[j]['WTeamID'],nc_list[k].iloc[j]['LTeamID']))]
    k = k+1 
massey = pd.read_csv("../input/MasseyOrdinals.csv")
massey.shape
massey.tail()
#Filtering data according to years
m_2003 = massey[massey['Season'] ==2003]
m_2004 = massey[massey['Season'] ==2004]
m_2005 = massey[massey['Season'] ==2005]
m_2006 = massey[massey['Season'] ==2006]
m_2007 = massey[massey['Season'] ==2007]
m_2008 = massey[massey['Season'] ==2008]
m_2009 = massey[massey['Season'] ==2009]
m_2010 = massey[massey['Season'] ==2010]
m_2011 = massey[massey['Season'] ==2011]
m_2012 = massey[massey['Season'] ==2012]
m_2013 = massey[massey['Season'] ==2013]
m_2014 = massey[massey['Season'] ==2014]
m_2015 = massey[massey['Season'] ==2015]
m_2016 = massey[massey['Season'] ==2006]
m_2017 = massey[massey['Season'] ==2017]
m_2018 = massey[massey['Season'] ==2018]

m_list = [m_2003,m_2004,m_2005,m_2006,m_2007,m_2008,m_2009,m_2010,m_2011,m_2012,m_2013,m_2014,m_2015,m_2016,m_2017,m_2018]
#Which systemnames are in which years
s_2003 = list(set(m_2003['SystemName']))
s_2004 = list(set(m_2004['SystemName']))
s_2005 = list(set(m_2005['SystemName']))
s_2006 = list(set(m_2006['SystemName']))
s_2007 = list(set(m_2007['SystemName']))
s_2008 = list(set(m_2008['SystemName']))
s_2009 = list(set(m_2009['SystemName']))
s_2010 = list(set(m_2010['SystemName']))
s_2011 = list(set(m_2011['SystemName']))
s_2012 = list(set(m_2012['SystemName']))
s_2013 = list(set(m_2013['SystemName']))
s_2014 = list(set(m_2014['SystemName']))
s_2015 = list(set(m_2015['SystemName']))
s_2016 = list(set(m_2016['SystemName']))
s_2017 = list(set(m_2017['SystemName']))
s_list = [s_2003,s_2004,s_2005,s_2006,s_2007,s_2008,s_2009,s_2010,s_2011,s_2012,s_2013,s_2014,s_2015,s_2016,s_2017]
big_list = []
%%time
#We are taking evaluations only after 117th day for each team.
r = 0
for m in m_list:
    middle_dict = {}
    for l in l_list[r]:
        small_list = []
        temp = m[(m['TeamID'] ==l)&(m['RankingDayNum'] >117)]
        for s in s_list[r]:
            mn = temp[temp['SystemName'] ==s]['OrdinalRank'].mean()
            small_list.append(mn)
        middle_dict[str(l)] = small_list
    big_list.append(middle_dict)
    r = r+1
#Averaging evaluations
r = 0
for b in big_list:
    b = pd.DataFrame(b).T
    big_list[r] =b
    r = r+1
#Converting np.array to pd.Series
r = 0
for b in big_list:
    i = b.index
    b = np.nanmean(b,axis=1)
    big_list[r] = pd.Series(b,index=i)
    r = r + 1
#Filling NaN values with mean. Some teams aren't evaluated after 117th day.
r = 0
for i in big_list:
    k = i.mean()
    big_list[r] = i.fillna(k)
    r = r+1    
%%time
#The code below is rounding the results. If the rank is equal to 34.2, it was rounded to 34.
r = 0
for i in nc_list:
    i['f_rank'],i['s_rank'] = 0,0
    for j in range(len(i)):
        a = min(i.iloc[j]['WTeamID'],i.iloc[j]['LTeamID'])
        b = max(i.iloc[j]['WTeamID'],i.iloc[j]['LTeamID'])
        a = big_list[r].loc[str(a)]
        b = big_list[r].loc[str(b)]
        i['f_rank'][j] = a 
        i['s_rank'][j] = b
    nc_list[r] = i
    r = r + 1
#Assigning season to rows
j=2003
for i in nc_list:
    i['Season'] = j
    j = j + 1
nc_list[-1].tail()
m_list[0].head()
ntdr.head()
nc_list[0].head()
tourney_wins = ntdr.groupby(('Season', 'WTeamID'))['WTeamID'].count()
tourney_wins = pd.DataFrame(tourney_wins)
tourney_wins.index.rename(('Season', 'TeamID'), inplace=True)
tourney_wins.rename({'WTeamID':'tourney_wins'}, inplace=True, axis=1)
tourney_wins.head()
df_list[-2].head()
import copy
orig_df_list = copy.deepcopy(df_list)
df_list = copy.deepcopy(orig_df_list)
test_df_list = copy.deepcopy(orig_df_list)
orig_df_list[0].head()
df_list = copy.deepcopy(orig_df_list)
new_df_list = []
for idx, df in enumerate(df_list):
    df['Season'] = idx + 2003
    df.reset_index(inplace=True)
    df.rename({'index':'TeamID'}, inplace=True, axis=1)
    df.set_index(['Season', 'TeamID'], inplace=True)
    joined = df.join(tourney_wins, how='left', rsuffix='right_')
    joined.fillna(0, inplace=True)
    joined.reset_index(inplace=True)
    new_df_list.append(joined)
df_list = new_df_list
df_list[0].head()
tourney_wins = ntdr.groupby(('Season', 'WTeamID'))['WTeamID'].count()
tourney_wins = pd.DataFrame(tourney_wins)
tourney_wins.index.rename(('Season', 'TeamID'), inplace=True)
tourney_wins.rename({'WTeamID':'tourney_wins'}, inplace=True, axis=1)
tourney_wins.reset_index(inplace=True)
tourney_wins.head()

#Merging all years data
X = pd.concat(nc_list)
X_2018 = df_list[-1]
X_2018.head()
#Merging all years data
X = pd.concat(df_list[:-1])
X.head()
X.drop(('Seed_region'), axis=1,inplace=True)
X_2018.drop(('Seed_region'), axis=1, inplace=True)
Y = X['tourney_wins']
X.drop(('tourney_wins'), axis=1, inplace=True)
X_2018.drop(('tourney_wins'), axis=1, inplace=True)
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators = 1000)
model.fit(X, Y, eval_metric = 'logloss')
predictions_2018 = model.predict(X_2018)
predictions_2018
final = copy.deepcopy(X_2018)
final['predicted_wins'] = predictions_2018
normalized = (predictions_2018 - predictions_2018.min()) / (predictions_2018.max() - predictions_2018.min())
translated = predictions_2018 + 1.0
normalized = (predictions_2018 - predictions_2018.min()) / (predictions_2018.max() - predictions_2018.min())
normalized
final['normalized_predicted_wins'] = normalized
final.head()
processed = set()
team_matchups = []
prob_lower_team_number_wins = []
for i in range(len(final)):
    for j in range(len(final)):
        unique_key = str(min(final['TeamID'][i], final['TeamID'][j])) + str(max((final['TeamID'][i], final['TeamID'][j])))
        if i != j and unique_key not in processed:
            team_matchups.append('2018_' + str(final['TeamID'][i]) + '_' + str(final['TeamID'][j]))
            prob_lower_team_number_wins.append(final['normalized_predicted_wins'][i] / (final['normalized_predicted_wins'][i] + final['normalized_predicted_wins'][j]))
            processed.add(unique_key)
prob_lower_team_number_wins
final_submission = pd.DataFrame(list(zip(team_matchups, prob_lower_team_number_wins)), columns=['ID', 'Pred'])
final_submission.head()
final_submission.to_csv('final_submission.csv')
final_submission.shape
x_columns = list(X.columns.values)
x_columns = [x for x in x_columns if x not in ['f_Seed_region','s_Seed_region','winning','Season']]
y = X['winning']
X = X.drop('winning',axis=1)
X.head()
X.tail()
X.tail()
#Creating train and test data
X_train = X[:713]
y_train = y[:713]
X_test = X[713:]
y_test = y[713:]
#Dividing test data into 2014,2015,2016,2017
t_2014 = X_test[:67]
t_2015 = X_test[67:134]
t_2016 = X_test[134:201]
t_2017 = X_test[201:268]
teams_list = [teams_2014,teams_2015,teams_2016,teams_2017]
teams_list = id_and_predictions['TeamID'].values
sub_df.head()
#Creating year column.
sub_df['year'] = 0
sub_df['year'][:2278]=2014
sub_df['year'][2278:int(2278*2)]=2015
sub_df['year'][int(2278*2):int(2278*3)]=2016
sub_df['year'][int(2278*3):]=2017
testing = pd.DataFrame(np.zeros((len(sub_df),len(x_columns))) ,columns=x_columns)
testing = testing.rename(columns={'WTeamID':'FirstID','LTeamID':'SecondID'})
testing.head()
%%time
for i in range(len(sub_df)):
    season = int(sub_df.iloc[i]['year'])
    f_t = int(sub_df.iloc[i]['FirstID'])
    s_t = int(sub_df.iloc[i]['SecondID'])
    testing.iloc[i]['FirstID'] = f_t
    testing.iloc[i]['SecondID'] = s_t
    testing.iloc[i]['f_coach'] = int(d_list[int(season-2003)][str(f_t)])
    testing.iloc[i]['s_coach'] = int(d_list[int(season-2003)][str(s_t)])
    #testing.iloc[i]['f_rank'] = big_list[season-2003][str(f_t)]
    testing.iloc[i]['s_rank'] = big_list[season-2003][str(s_t)]
    '''for j in list(testing.columns.values)[:19]:
        k = j[2:]
        testing.iloc[i][j] = df_list[season-2003].loc[f_t,k]
    for j in list(testing.columns.values)[19:38]:
        k = j[2:]
        testing.iloc[i][j] = df_list[season-2003].loc[s_t,k]    
    '''
testing.tail()
#Deleting some columns which we don't put in the model.
del X['f_Seed_region'],X['s_Seed_region'],X['WTeamID'],X['LTeamID'],X['Season']
del testing['FirstID'],testing['SecondID']
X.shape
y.shape


