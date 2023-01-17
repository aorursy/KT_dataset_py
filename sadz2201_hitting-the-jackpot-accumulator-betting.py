import pandas as pd

import sqlite3

from datetime import timedelta

import warnings

warnings.filterwarnings("ignore")



#load data (make sure you have downloaded database.sqlite)

with sqlite3.connect('../input/database.sqlite') as con:

    countries = pd.read_sql_query("SELECT * from Country", con)

    matches = pd.read_sql_query("SELECT * from Match", con)

    leagues = pd.read_sql_query("SELECT * from League", con)

    teams = pd.read_sql_query("SELECT * from Team", con)
selected_countries = ['Scotland','France','Germany','Italy','Spain','Portugal','England']

countries = countries[countries.name.isin(selected_countries)]

leagues = countries.merge(leagues,on='id',suffixes=('', '_y'))



#There's a special character in the long name "Atlético Madrid".

#This can be a pain in the ass, so I'm gonna change it for simplicity.

teams.loc[teams.team_api_id==9906,"team_long_name"] = "Atletico Madrid"



#retain only data from 2011-12 season

matches=matches[matches.date>='2011-08-01']

matches = matches[matches.league_id.isin(leagues.id)]

matches = matches[['id', 'country_id' ,'league_id', 'season', 'stage', 'date','match_api_id', 'home_team_api_id', 'away_team_api_id','home_team_goal','away_team_goal','B365H', 'B365D' ,'B365A']]

matches.dropna(inplace=True)

matches.head()
matches=matches.merge(teams,left_on='home_team_api_id',right_on='team_api_id',suffixes=('','_h'))

matches=matches.merge(teams,left_on='away_team_api_id',right_on='team_api_id',suffixes=('','_a'))

matches=matches[['id', 'season', 'date','home_team_goal','away_team_goal','B365H', 'B365D' ,'B365A',

                 'team_long_name','team_long_name_a']]

matches.head()
accumulator_teams=['FC Barcelona','Real Madrid CF','Celtic','FC Porto','SL Benfica','Juventus','FC Bayern Munich','Paris Saint-Germain','Manchester City','Atletico Madrid','Borussia Dortmund','Arsenal']



#matches where any of our 12 teams is playing at home

matches_h=matches[matches.team_long_name.isin(accumulator_teams)]

#matches where any of our 12 teams is playing away

matches_a=matches[matches.team_long_name_a.isin(accumulator_teams)]

#concat & drop duplicates

matches=pd.concat([matches_h,matches_a],axis=0)

matches.drop_duplicates(inplace=True)



matches=matches.sort_values(by='date')

#remove matches where our teams are facing off against each other

matches=matches[~((matches.team_long_name.isin(accumulator_teams)) & (matches.team_long_name_a.isin(accumulator_teams)))]



matches.head()
matches.date=pd.to_datetime(matches.date)

#monday matches. subtract 2 to make it saturday

m0=matches[matches.date.dt.weekday==0]

m0.date=m0.date-timedelta(days=2)



#tuesday matches

m1=matches[matches.date.dt.weekday==1]

#wednesday matches. subtract 1 to make it tuesday

m2=matches[matches.date.dt.weekday==2]

m2.date=m2.date-timedelta(days=1)

#thursday matches. subtract 2 to make it tuesday

m3=matches[matches.date.dt.weekday==3]

m3.date=m3.date-timedelta(days=2)



#friday matches. add 1 to make it saturday

m4=matches[matches.date.dt.weekday==4]

m4.date=m4.date+timedelta(days=1)

#saturday matches

m5=matches[matches.date.dt.weekday==5]

#sunday matches. subtract 1 to make it saturday

m6=matches[matches.date.dt.weekday==6]

m6.date=m6.date-timedelta(days=1)



#merge all, sort by date

matches=pd.concat([m0,m1,m2,m3,m4,m5,m6],axis=0)

matches=matches.sort_values(by='date')

del m0,m1,m2,m3,m4,m5,m6



#checking if we have only saturday & tuesday now

matches.date.dt.weekday.value_counts()
matches['our_team']='abc'

matches['our_venue']='H'

matches['our_odds']=matches.B365H



is_home=matches.team_long_name.isin(accumulator_teams)

#our team is playing at home

matches.our_team[is_home==True]=matches.team_long_name[is_home==True]



#our team is playing away.

matches.our_team[is_home==False]=matches.team_long_name_a[is_home==False]

matches.our_venue[is_home==False]='A'

matches.our_odds[is_home==False]=matches.B365A[is_home==False]
matches['result']='H'

matches.loc[matches.home_team_goal==matches.away_team_goal,"result"]='D'

matches.loc[matches.home_team_goal<matches.away_team_goal,"result"]='A'



matches['payout']=matches.our_odds

#our team either lost or drew. reset payout to 0

matches.loc[~(matches.result==matches.our_venue),"payout"]=0

matches.head()
print(sum(matches.payout)/matches.shape[0])
team_n=matches.our_team.value_counts()

print ("win percentage by team:")

print(matches[matches.payout!=0].our_team.value_counts()/team_n)

print("_"*50)

print ("net payout by team:")

indiv_payout=matches.groupby('our_team')['payout'].sum()

indiv_payout=indiv_payout/team_n

print(indiv_payout)
#our teams list in sorted order of individual profits

accumulator_teams=['Juventus','SL Benfica','Atletico Madrid','FC Bayern Munich','Paris Saint-Germain','Manchester City','Real Madrid CF','Arsenal','Celtic','FC Porto','FC Barcelona','Borussia Dortmund']

#list of bet365 bonus payouts

#bonus[k]= bet365 bonus for k-fold accumulator

bonus=[1,1,1,1.05,1.1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8]



#blank dict

accum_payouts={}

for k in range(2,len(accumulator_teams)+1):

    #choose first k teams from the team list

    accum_subset=accumulator_teams[:k]

    

    #choose only matches involving these teams

    matches_kfold=matches[matches.our_team.isin(accum_subset)]

    #count of matches per date.

    date_counts=matches_kfold.date.value_counts().reset_index()

    date_counts.columns=['date','counts']

    

    #select only the dates where all k teams are in action

    dates_kfold=date_counts[date_counts.counts==k].date

    #retain only the matches happening on these dates

    matches_kfold=matches_kfold[matches_kfold.date.isin(dates_kfold)]

    #k-fold accumulator payout (product of payouts of all k teams on that date)

    payout_kfold=matches_kfold.groupby('date')['payout'].prod()

    #multiply bonus

    bonus_payout_kfold=payout_kfold* bonus[k]

    print(str(k) + " fold:")

    print(accum_subset)

    print("#bets: " + str(len(payout_kfold)))

    print("#correct predictions: " + str(len(payout_kfold[payout_kfold!=0])))

    print("Net outcome (without bonus): " + str(sum(payout_kfold)/len(payout_kfold)))

    print("Net outcome (after bonus): " + str(sum(bonus_payout_kfold)/len(payout_kfold)))

    print("_"*50)

    accum_payouts[k]=sum(bonus_payout_kfold)/len(payout_kfold)

    

#print the best choice of k, the corresponding teams & net payout.

best_k=max(accum_payouts,key=accum_payouts.get)

print("best k= " +str(best_k) )

print(accumulator_teams[:best_k])

print("best payout= " +str(accum_payouts[best_k]))