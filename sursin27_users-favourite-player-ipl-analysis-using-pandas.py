import pandas as pd
deliveries=pd.read_csv('../input/ipl/deliveries.csv',sep=',')
matches=pd.read_csv('../input/ipl/matches.csv',sep=',')
deliveries
matches
deliveries.shape
matches.shape
matches.shape
matches.head()
matches['match_id']=matches['id']

matches.head()
ipl=deliveries.merge(matches,on='match_id',how='inner')
ipl
ipl.shape
ipl.isnull().any()
match=matches
match.isnull().any()
match.shape
match.isnull().any()
match.fillna(0)
ipl.describe()
ipl.shape
ipl.head()
ipl.columns
no_of_years=ipl['season'].unique()

no_of_years
team_names1=ipl['team1'].unique()
team_names1
team_names2=ipl['bowling_team'].unique()
team_names2
team_names1.shape
team_names2.shape
csk_details=ipl['team1'].str.contains('Chennai Super Kings') | ipl['team2'].str.contains('Chennai Super Kings')

csk_details
cskmatches=ipl[csk_details]
cskmatches
cskmatches.shape
everycskmatch=cskmatches['match_id'].unique()

everycskmatch
csk_match=ipl.loc[ipl.match_id.isin(everycskmatch)]

csk_match
csk_match.shape
csk_years=csk_match.season.unique()

csk_years
cskmatchlist=ipl.match_id.loc[everycskmatch]
cskmatchlist.shape
onlycskmatch=matches['team1'].str.contains('Chennai Super Kings') | matches['team2'].str.contains('Chennai Super Kings')

onlycskmatch
onlycskmatches=matches[onlycskmatch]
onlycskmatches
uniqueplayerofmatches=onlycskmatches['player_of_match'].unique()
uniqueplayerofmatches
cskwins=onlycskmatches.winner=='Chennai Super Kings'

cskwinmatches=onlycskmatches[cskwins]
cskwinmatches
cskwinmatches.shape
pomat=cskwinmatches.player_of_match.value_counts()

pomat
highwinners=matches.winner.value_counts()

highwinners
manofmatches=matches.player_of_match.value_counts()

manofmatches
umargul=matches.player_of_match=='Umar Gul'

umargulmatch=matches[umargul]

umargulmatch

#csd=ipl.match_id=='112'

#csd

performumargul=ipl[ipl.match_id==112]

performumargul
performumargul.shape
onlyhis=performumargul.batsman.str.contains('Umar Gul')|performumargul.bowler.str.contains('Umar Gul')
onlyhisper=performumargul[onlyhis]
onlyhisper
onlyhisper.shape
onlyhisper.columns
umargulbat=onlyhisper.batsman.str.contains('Umar Gul')
umargulbatting=onlyhisper[umargulbat]

umargulbatting
umargulruns=umargulbatting.batsman_runs

umargulruns
s=0

for di in umargulruns:

    s=s+di

print(s)
umarbowl=onlyhisper.bowler.str.contains('Umar Gul')

umarbowling=onlyhisper[umarbowl]

umarbowling
wickets=umarbowling['player_dismissed']

valwic=wickets.dropna()
valwic
wicketstook=valwic.count()
print(s,wicketstook)
showme=umarbowling[[ 'wide_runs',

       'bye_runs', 'legbye_runs', 'noball_runs', 'penalty_runs',

       'batsman_runs', 'extra_runs', 'total_runs', 'player_dismissed',

       'dismissal_kind']]

showme
umarbowlruns=umarbowling[[ 'wide_runs',

       'noball_runs',

       'batsman_runs']]

umarbowlruns
wides=(umarbowlruns.wide_runs>0 )| (umarbowlruns.noball_runs>0) |(umarbowlruns.batsman_runs>0) 

wide=umarbowlruns[wides]

runs=wide.batsman_runs

#wideru=wide.wide_runs.value_counts()

#wideru
runs
runss=wide.noball_runs

runsss=wide.wide_runs

runss

runsss

d=0

for c in runs:

    d=d+c

e=0

for f in runss:

    e=e+f

h=0

for g in runsss:

   h=h+g

xs=d+e+h

print(xs,s)
pla=input('enter player name')
print(pla)
ipl
ipl.shape
matches
matches.shape
pomuser=(matches.player_of_match==pla)

pla_pom=matches[pomuser]
pla_pom
pla_pom.player_of_match.count()
ipl.head()
ipl.columns
onlyhisbat=ipl[ipl['batsman']==pla]

onlyhisbat
hisruns=onlyhisbat['batsman_runs'].sum()
hisruns
checkruns=onlyhisbat['batsman_runs']

checkruns
su=0;

for dp in checkruns:

    su=su+dp

print(su)
onlyhisbowl=ipl[ipl.bowler==pla]

onlyhisbowl
onlyhisbowl.dismissal_kind.unique()
takewicket=[ 'caught', 'bowled', 'lbw', 'caught and bowled']
wickets=onlyhisbowl[onlyhisbowl.dismissal_kind.isin(takewicket)]

wickets
wickets.dismissal_kind.count()
pws=ipl[['season','bowler','dismissal_kind']].groupby(['season','bowler']).count().reset_index()

pws
pws=pws.sort_values('dismissal_kind',ascending=False)
pws=pws.drop_duplicates('season',keep='first')
pws
pws.sort_values('season')
rpes=ipl[ipl.batsman==pla]

rpes
runperseason=rpes[['match_id','season','batsman_runs']].groupby(['match_id','season']).count().reset_index()
runperseason=runperseason.groupby(['season']).sum()

runperseason.drop(['match_id'],axis=1,inplace=True) 
sur=runperseason.reset_index()
import matplotlib.pyplot as mlt
sur.columns

#seasons=runperseason.unique()

#seasons
seasons=sur.season.unique().tolist()

seasons
runs=sur.batsman_runs.tolist()

runs
mlt.figure(figsize=(10,5))

ax=mlt.bar(seasons,runs)

# Label the axes

mlt.xlabel('season')

mlt.ylabel('runs')





#label the figure

mlt.title('runs per year')

mlt.show()
pla='JH Kallis'

bwst=ipl[ipl.bowler==pla]

bwst
bowling=bwst[['season','dismissal_kind']].groupby('season').count()

bowling
bowlin=bowling.reset_index()
mlt.plot(bowlin['season'].values,bowlin['dismissal_kind'].values)



mlt.xlabel('season')

mlt.ylabel('wickets')





#label the figure

mlt.title('wickets per year')

mlt.show()