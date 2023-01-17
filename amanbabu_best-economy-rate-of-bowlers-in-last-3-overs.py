import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib

%matplotlib tk

deliveries=pd.read_csv('../input/deliveries.csv')

bowler_grp=deliveries.groupby(['match_id', 'inning', 'bowling_team', 'over', 'bowler'])

bowlers = bowler_grp['total_runs','wide_runs', 'bye_runs', 'legbye_runs', 'noball_runs'].sum().reset_index()

total_matches = bowlers.groupby('match_id').count()

total_matches = total_matches.shape[0]



ipl_matches=pd.DataFrame(columns=['match_id','inning','bowling_team', 'over', 'bowler','total_runs','wide_runs', 'bye_runs', 'legbye_runs', 'noball_runs'], index=[x for x in range(1, 578)])



last_3oversData1= []

for match in range(1, total_matches+1):

    for inning in range(1,3):

        ipl_matches=bowlers.loc[(bowlers['match_id']==match)& (bowlers['inning']==inning)]

        if(len(ipl_matches) > 0):

            last_3overs=ipl_matches['over'].max()-3

            ipl_matches=ipl_matches.iloc[last_3overs:]

            last_3oversData1.append(ipl_matches)

            last_3overs_total_matches1=pd.concat(last_3oversData1)

            



last_3overs_total_matches_a=last_3overs_total_matches1.reset_index(drop=True)
matches=pd.read_csv('../input/matches.csv')

bowlers_a=bowler_grp['total_runs', 'wide_runs', 'bye_runs', 'legbye_runs', 'noball_runs'].sum().reset_index()

bowlers_a=matches[['id', 'season']].merge(bowlers_a, left_on='id', right_on='match_id', how='left').drop('id', axis=1)



bowlers_b=bowlers_a.groupby(['season','bowling_team','match_id','bowler']).count()



bowlers_b=bowlers_b.reset_index()



bowlers_c=bowlers_b.groupby(['season','bowler'])['match_id'].count()



bowlers_c=bowlers_c.reset_index()

bowlers_c.head(100)

bowlers_c=bowlers_c.rename(columns={'match_id':'total_matches_played'})
bowlers_c.head()
bowlers_played=bowlers_c[bowlers_c['total_matches_played']>=3]
last_3overs_total_matches_a['runs'] = last_3overs_total_matches_a['total_runs']-(last_3overs_total_matches_a['legbye_runs'] + last_3overs_total_matches_a['bye_runs'])
last_3overs_total_matches_a['extras'] = last_3overs_total_matches_a['wide_runs'] + last_3overs_total_matches_a['noball_runs']
del(last_3overs_total_matches_a['total_runs'])

del(last_3overs_total_matches_a['bye_runs'])

del(last_3overs_total_matches_a['legbye_runs'])
dismissal_kinds_for_bowler=['bowled', 'caught', 'lbw' , 'stumped', 'caught and bowled', 'hit wicket']
dismissals=deliveries[deliveries['dismissal_kind'].isin(dismissal_kinds_for_bowler)]
dismissals=dismissals.groupby(['match_id', 'inning', 'bowling_team', 'bowler', 'over'])['dismissal_kind'].count().reset_index()
dismissals.rename(columns={'dismissal_kind':'wickets'}, inplace=True)
last_3overs_total_matches_a=last_3overs_total_matches_a.merge(dismissals, left_on=['match_id', 'inning', 'bowling_team', 'bowler','over'], right_on=['match_id', 'inning', 'bowling_team', 'bowler', 'over'],how='left')
last_3overs_total_matches_a['wickets']=last_3overs_total_matches_a['wickets'].fillna(0)
bowlers_over=last_3overs_total_matches_a.groupby(['match_id', 'inning', 'bowling_team', 'bowler'])['over'].count().reset_index()
last_3overs_total_matches_a=last_3overs_total_matches_a.groupby(['match_id', 'inning', 'bowling_team', 'bowler']).sum().reset_index().drop('over',1)
last_3overs_total_matches_a=bowlers_over.merge(last_3overs_total_matches_a, on=['match_id', 'inning', 'bowling_team', 'bowler'], how='left')
last_3overs_total_matches_a['Econ'] =np.round(last_3overs_total_matches_a['runs'] / last_3overs_total_matches_a['over'],2)
last_3overs_total_matches_a=matches[['id', 'season']].merge(last_3overs_total_matches_a, left_on='id', right_on='match_id', how='left').drop('id', axis=1)
bowlers_played_b=bowlers_played.merge(last_3overs_total_matches_a, on=['season','bowler'], how='inner')
bowlers_played_c=bowlers_played_b.groupby(['season','bowler','total_matches_played','bowling_team'])['Econ','over'].sum().reset_index()
bowlers_played_c['overall_economy']=(bowlers_played_c['Econ']/bowlers_played_c['over'])
bowlers_played_c=bowlers_played_c[bowlers_played_c['over']>= (bowlers_played_c['total_matches_played']/3)]
bowlers_played_c=bowlers_played_c.sort_values('overall_economy',ascending=True)
bowlers_played_c1=bowlers_played_c.groupby(['season', 'bowler'])['overall_economy'].sum().unstack().T
bowlers_played_c1['Average']=bowlers_played_c1.mean(axis=1)
bowlers_played_c1.sort_values('Average', ascending=False, inplace=True)
bowlers_played_c2=bowlers_played_c1[bowlers_played_c1.isnull().sum(axis=1) <5]
bowlers_played_c2.sort_values('Average',ascending=False, inplace=True)

bowlers_played_c2=bowlers_played_c2.reset_index()
bowlers_played_c2[:5].plot(x='bowler', y='Average',kind='bar')