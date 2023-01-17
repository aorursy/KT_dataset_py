# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as mlt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
matches=pd.read_csv('../input/matches.csv')   

delivery=pd.read_csv('../input/deliveries.csv')

matches.head(2)
delivery.head(2)
matches.drop(['umpire3'],axis=1,inplace=True)  #since all the values are NaN

delivery.fillna(0,inplace=True)     #filling all the NaN values with 0
#Replacing the Team Names with their abbreviations



matches.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',

                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',

                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors']

                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW'],inplace=True)



delivery.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',

                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',

                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors']

                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW'],inplace=True)
print('Total Matches Played:',matches.shape[0])

print(' \n Venues Played At:',matches['city'].unique())     

print(' \n Teams :',matches['team1'].unique())
print('Total venues played at:',matches['city'].nunique())

print('\nTotal umpires ',matches['umpire1'].nunique())
print((matches['player_of_match'].value_counts()).idxmax(),' : has most man of the match awards')

print(((matches['winner']).value_counts()).idxmax(),': has the highest number of match wins')
df=matches.iloc[[matches['win_by_runs'].idxmax()]]

df[['season','team1','team2','winner','win_by_runs']]
df=matches.iloc[[matches['win_by_wickets'].idxmax()]]

df[['season','team1','team2','winner','win_by_wickets']]
print('Toss Decisions in %\n',((matches['toss_decision']).value_counts())/577*100)
mlt.subplots(figsize=(10,6))

sns.countplot(x='season',hue='toss_decision',data=matches)

mlt.show()
mlt.subplots(figsize=(10,6))

ax=matches['toss_winner'].value_counts().plot.bar(width=0.8)

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))

mlt.show()
matches_played_byteams=pd.concat([matches['team1'],matches['team2']])

matches_played_byteams=matches_played_byteams.value_counts().reset_index()

matches_played_byteams.columns=['Team','Total Matches']

matches_played_byteams['wins']=matches['winner'].value_counts().reset_index()['winner']

matches_played_byteams.set_index('Team',inplace=True)

matches_played_byteams.plot.bar(stacked=1,width=0.8)

fig=mlt.gcf()

fig.set_size_inches(10,5)
df=matches[matches['toss_winner']==matches['winner']]

slices=[len(df),(577-len(df))]

labels=['yes','no']

mlt.pie(slices,labels=labels,startangle=90,shadow=True,explode=(0,0),autopct='%1.1f%%')

fig = mlt.gcf()

fig.set_size_inches(6,6)

mlt.show()
mlt.subplots(figsize=(10,6))

sns.countplot(x='season',data=matches,palette="Set1")  #countplot automatically counts the frequency of an item

sns.plt.show()
mlt.subplots(figsize=(10,6))

batsmen = matches[['id','season']].merge(delivery, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)

#merging the matches and delivery dataframe by referencing the id and match_id columns respectively

season=batsmen.groupby(['season'])['total_runs'].sum().reset_index()

season['total_runs'].plot(marker='o')

mlt.show()
mlt.subplots(figsize=(10,6))

avgruns_each_season=matches.groupby(['season']).count().id.reset_index()

avgruns_each_season.rename(columns={'id':'matches'},inplace=1)

avgruns_each_season['total_runs']=season['total_runs']

avgruns_each_season['average_runs_per_match']=avgruns_each_season['total_runs']/avgruns_each_season['matches']

avgruns_each_season['average_runs_per_match'].plot(marker='o')

mlt.show()
Season_boundaries=batsmen.groupby("season")["batsman_runs"].agg(lambda x: (x==6).sum()).reset_index()

a=batsmen.groupby("season")["batsman_runs"].agg(lambda x: (x==4).sum()).reset_index()

Season_boundaries=Season_boundaries.merge(a,left_on='season',right_on='season',how='left')

Season_boundaries=Season_boundaries.rename(columns={'batsman_runs_x':'6"s','batsman_runs_y':'4"s'})

Season_boundaries[['6"s','4"s']].plot(marker='o')

fig=mlt.gcf()

fig.set_size_inches(10,6)

mlt.show()
runs_per_over = delivery.pivot_table(index=['over'],columns='batting_team',values='total_runs',aggfunc=sum)

runs_per_over[(matches_played_byteams[matches_played_byteams['Total Matches']>50].index)].plot(color=["b", "r", "#Ffb6b2", "g",'brown','y','#6666ff','black','#FFA500']) #plotting graphs for teams that have played more than 100 matches

x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

mlt.xticks(x)

mlt.ylabel('total runs scored')

fig=mlt.gcf()

fig.set_size_inches(10,8)

mlt.show()
mlt.subplots(figsize=(10,6))

ax = matches['venue'].value_counts().plot.bar(width=.8, color=["#999966", "#8585ad", "#c4ff4d", "#ffad33"])

ax.set_xlabel('Grounds')

ax.set_ylabel('count')

mlt.show()
mlt.subplots(figsize=(10,6))

#the code used is very basic but gets the job done easily

ax = matches['player_of_match'].value_counts().head(10).plot.bar(width=.8, color='R')  #counts the values corresponding 

# to each batsman and then filters out the top 10 batsman and then plots a bargraph 

ax.set_xlabel('player_of_match') 

ax.set_ylabel('count')

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+0.25))

mlt.show()
for i in range(2008,2017):

    df=((matches[matches['season']==i]).iloc[-1]) 

    print(df[[1,10]])

#getting the last match in every season since the last match will be the final match for the season
print('\n Total Matches with Super Overs:',delivery[delivery['is_super_over']==1].match_id.nunique())
teams=['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW']

play=delivery[delivery['is_super_over']==1].batting_team.unique()

play=list(play)

print('Teams who haven"t ever played a super over are:' ,list(set(teams)-set(play)))
mlt.subplots(figsize=(10,6))

sns.countplot(x='winner', data=matches)

mlt.xticks(rotation='vertical')

mlt.show()
mlt.subplots(figsize=(10,6))

ump=pd.concat([matches['umpire1'],matches['umpire2']]) 

ax=ump.value_counts().head(10).plot.bar(width=0.8,color='Y')

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+0.25))

mlt.show()
mlt.subplots(figsize=(10,6))

mt1=matches[((matches['team1']=='MI')|(matches['team2']=='MI'))&((matches['team1']=='KKR')|(matches['team2']=='KKR'))]

sns.countplot(x='season', hue='winner', data=mt1)

mlt.xticks(rotation='vertical')

mlt.show()
mlt.subplots(figsize=(10,6))

mt2=matches[((matches['team1']=='MI')|(matches['team2']=='MI'))&((matches['team1']=='CSK')|(matches['team2']=='CSK'))]

sns.countplot(x='season', hue='winner', data=mt2)

mlt.xticks(rotation='vertical')

mlt.show()
mlt.subplots(figsize=(10,6))

xyz=delivery.groupby(['match_id','inning','batting_team'])['total_runs'].sum().reset_index()

xyz.drop('match_id',axis=1,inplace=True)

xyz=xyz.sort_values(by=['batting_team','total_runs'],ascending=True)

score_1_inning=xyz[xyz['inning']==1]

score_2_inning=xyz[xyz['inning']==2]

sns.boxplot(x='batting_team',y='total_runs',data=score_1_inning).set_title('1st Innings')

mlt.show()

sns.boxplot(x='batting_team',y='total_runs',data=score_2_inning).set_title('2nd Innings')
mlt.subplots(figsize=(10,6))

sns.violinplot(x='batting_team',y='total_runs',data=score_1_inning).set_title('1st Innings')

mlt.show()

sns.violinplot(x='batting_team',y='total_runs',data=score_2_inning).set_title('2nd Innings')
high_scores=delivery.groupby(['match_id', 'inning','batting_team','bowling_team'])['total_runs'].sum().reset_index() 

#reset_index() converts the obtained series into a dataframe

high_scores=high_scores[high_scores['total_runs']>=200]

#nlargest is used to sort the given column

high_scores.nlargest(10,'total_runs')
fig, ax =mlt.subplots(1,2,figsize=(15,8))

sns.countplot(high_scores['batting_team'],ax=ax[0])

sns.countplot(high_scores['bowling_team'],ax=ax[1])

mlt.xticks(rotation=90)

fig=mlt.gcf()

fig.set_size_inches(16,6)

mlt.show()
print('Teams who have"nt ever scored 200 runs',list(set(teams)-set(high_scores['batting_team'])))

print('Teams who haven"t conceeded over 200 while bowling',list(set(teams)-set(high_scores['bowling_team'])))
high=delivery.groupby(['match_id', 'inning','batting_team','bowling_team'])['total_runs'].sum().reset_index()

high.set_index(['match_id'],inplace=True)

high['total_runs'].max()

high.columns

high=high.rename(columns={'total_runs':'count'})

high=high[high['count']>=200].groupby(['inning','batting_team','bowling_team']).count()

high
high_scores=delivery.groupby(['match_id', 'inning','batting_team','bowling_team'])['total_runs'].sum().reset_index()

high_scores1=high_scores[high_scores['inning']==1]

high_scores2=high_scores[high_scores['inning']==2]

high_scores1=high_scores1.merge(high_scores2[['match_id','inning', 'total_runs']], on='match_id')

high_scores1.rename(columns={'inning_x':'inning_1','inning_y':'inning_2','total_runs_x':'inning1_runs','total_runs_y':'inning2_runs'},inplace=True)

high_scores1=high_scores1[high_scores1['inning1_runs']>=200]

high_scores1['is_score_chased']=1

high_scores1['is_score_chased'] = np.where(high_scores1['inning1_runs']<=high_scores1['inning2_runs'], 

                                           'yes', 'no')

high_scores1.head()
slices=high_scores1['is_score_chased'].value_counts().reset_index().is_score_chased

list(slices)

labels=['target not chased','target chased']

mlt.pie(slices,labels=labels,colors=['#1f2ff3', '#0fff00'],startangle=90,shadow=True,explode=(0,0.1),autopct='%1.1f%%')

fig = mlt.gcf()

fig.set_size_inches(6,6)

mlt.show()
balls=delivery.groupby(['batsman'])['ball'].count().reset_index()

runs=delivery.groupby(['batsman'])['batsman_runs'].sum().reset_index()

balls=balls.merge(runs,left_on='batsman',right_on='batsman',how='outer')

balls.columns=[['batsman','ball_x','ball_y']]

sixes=delivery.groupby('batsman')['batsman_runs'].agg(lambda x: (x==4).sum()).reset_index()

fours=delivery.groupby(['batsman'])['batsman_runs'].agg(lambda x: (x==6).sum()).reset_index()

balls['strike_rate']=balls['ball_y']/balls['ball_x']*100

balls=balls.merge(sixes,left_on='batsman',right_on='batsman',how='outer')

balls=balls.merge(fours,left_on='batsman',right_on='batsman',how='outer')

compare=delivery.groupby(["match_id", "batsman","batting_team"])["batsman_runs"].sum().reset_index()

compare=compare.groupby(['batsman','batting_team'])['batsman_runs'].max().reset_index()

balls=balls.merge(compare,left_on='batsman',right_on='batsman',how='outer')

balls.columns=[['batsman','balls','runs','strike_rate',"6's","4's",'Team','Highest_score']]

balls.head()
def batsman_comparator(stat1,stat2,batsman1,batsman2):

    sns.FacetGrid(balls,hue='Team',size=8).map(mlt.scatter, stat1,stat2, alpha=0.5).add_legend()

    bats1=balls[balls['batsman'].str.contains(batsman1)].sort_values(by=stat1,ascending=False)

    bats2=balls[balls['batsman'].str.contains(batsman2)].sort_values(by=stat1,ascending=False)

    mlt.scatter(bats1[stat1],bats1[stat2]-1,s=75,c='#55ff33')

    mlt.text(x=bats1[stat1].values[0],y=bats1[stat2].values[0],s=batsman1,

            fontsize=10, weight='bold', color='#f46d43')

    mlt.scatter(bats2[stat1],bats2[stat2],s=75,c='#f73545')

    mlt.text(x=bats2[stat1].values[0],y=bats2[stat2].values[0]+1,s=batsman2, 

            fontsize=10, weight='bold', color='#ff58fd')

    fig=mlt.gcf()

    fig.set_size_inches(10,6)

    mlt.title('Batsman Comparator',size=25)

    mlt.show()



batsman_comparator("6's","4's",'Gayle','Villiers') #comparing gayle and de-villiers based on their respective boundaries
batsman_comparator("runs","strike_rate",'Dhoni','V Kohli')
mlt.subplots(figsize=(10,6))

max_runs=delivery.groupby(['batsman'])['batsman_runs'].sum()

ax=max_runs.sort_values(ascending=False)[:10].plot.bar(width=0.8,color='R')

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.1, p.get_height()+1),fontsize=11)

mlt.show()
toppers=delivery.groupby(['batsman','batsman_runs'])['total_runs'].count().reset_index()

toppers=toppers.pivot('batsman','batsman_runs','total_runs')

fig,ax=mlt.subplots(2,2,figsize=(14,12))

toppers[1].sort_values(ascending=False)[:5].plot(kind='barh',ax=ax[0,0],color='#45ff45')

ax[0,0].set_title("Most 1's")

ax[0,0].set_ylabel('')

toppers[2].sort_values(ascending=False)[:5].plot(kind='barh',ax=ax[0,1],color='#df6dfd')

ax[0,1].set_title("Most 2's")

ax[0,1].set_ylabel('')

toppers[4].sort_values(ascending=False)[:5].plot(kind='barh',ax=ax[1,0],color='#fbca5f')

ax[1,0].set_title("Most 4's")

ax[1,0].set_ylabel('')

toppers[6].sort_values(ascending=False)[:5].plot(kind='barh',ax=ax[1,1],color='#ffff00')

ax[1,1].set_title("Most 6's")

ax[1,1].set_ylabel('')

mlt.show()
top_scores = delivery.groupby(["match_id", "batsman","batting_team"])["batsman_runs"].sum().reset_index()

#top_scores=top_scores[top_scores['batsman_runs']>100]

top_scores.sort_values('batsman_runs', ascending=0).head(10)

top_scores.nlargest(10,'batsman_runs')
swarm=['CH Gayle','V Kohli','G Gambhir','SK Raina','YK Pathan','MS Dhoni','AB de Villiers','DA Warner']

scores = delivery.groupby(["match_id", "batsman","batting_team"])["batsman_runs"].sum().reset_index()

scores=scores[top_scores['batsman'].isin(swarm)]

sns.swarmplot(x='batsman',y='batsman_runs',data=scores,hue='batting_team',palette='Set1')

fig=mlt.gcf()

fig.set_size_inches(14,8)

mlt.ylim(-10,200)

mlt.show()
a=batsmen.groupby(['season','batsman'])['batsman_runs'].sum().reset_index()

a=a.groupby(['season','batsman'])['batsman_runs'].sum().unstack().T

a['Total']=a.sum(axis=1)

a=a.sort_values(by='Total',ascending=0)[:5]

a.drop('Total',axis=1,inplace=True)

a.T.plot(color=['red','blue','#772272','green','#f0ff00'],marker='o')

fig=mlt.gcf()

fig.set_size_inches(10,6)

mlt.show()
a=batsmen.groupby(['batsman','batsman_runs'])['total_runs'].count().reset_index()

b=max_runs.sort_values(ascending=False)[:10].reset_index()

c=b.merge(a,left_on='batsman',right_on='batsman',how='left')

c.drop('batsman_runs_x',axis=1,inplace=True)

c.set_index('batsman',inplace=True)

c.columns=['type','count']

c=c[(c['type']==1)|(c['type']==2)|(c['type']==4)|(c['type']==6)]

cols=['type','count']

c.reset_index(inplace=True)

c=c.pivot('batsman','type','count')

c.plot.barh(width=0.8)

fig=mlt.gcf()

fig.set_size_inches(10,10)

mlt.show()


gayle=delivery[delivery['batsman']=='CH Gayle']

gayle=gayle[gayle['dismissal_kind'].isin(['caught','lbw','bowled','stumped','caught and bowled',"hit wicket"])]

gayle=gayle.groupby('bowler').count().sort_values(by='dismissal_kind',ascending=0).dismissal_kind[:1].reset_index()

gayle['batsman']='CH Gayle'



kohli=delivery[delivery['batsman']=='V Kohli']

kohli=kohli[kohli['dismissal_kind'].isin(['caught','lbw','bowled','stumped','caught and bowled',"hit wicket"])]

kohli=kohli.groupby('bowler').count().sort_values(by='dismissal_kind',ascending=0).dismissal_kind[:1].reset_index()

kohli['batsman']='V Kohli'





raina=delivery[delivery['batsman']=='SK Raina']

raina=raina[raina['dismissal_kind'].isin(['caught','lbw','bowled','stumped','caught and bowled',"hit wicket"])]

raina=raina.groupby('bowler').count().sort_values(by='dismissal_kind',ascending=0).dismissal_kind[:1].reset_index()

raina['batsman']='SK Raina'



abd=delivery[delivery['batsman']=='AB de Villiers']

abd=abd[abd['dismissal_kind'].isin(['caught','lbw','bowled','stumped','caught and bowled',"hit wicket"])]

abd=abd.groupby('bowler').count().sort_values(by='dismissal_kind',ascending=0).dismissal_kind[:1].reset_index()

abd['batsman']='AB de Villiers'



msd=delivery[delivery['batsman']=='MS Dhoni']

msd=msd[msd['dismissal_kind'].isin(['caught','lbw','bowled','stumped','caught and bowled',"hit wicket"])]

msd=msd.groupby('bowler').count().sort_values(by='dismissal_kind',ascending=0).dismissal_kind[:1].reset_index()

msd['batsman']='MS Dhoni'





gg=delivery[delivery['batsman']=='G Gambhir']

gg=gg[gg['dismissal_kind'].isin(['caught','lbw','bowled','stumped','caught and bowled',"hit wicket"])]

gg=gg.groupby('bowler').count().sort_values(by='dismissal_kind',ascending=0).dismissal_kind[:1].reset_index()

gg['batsman']='G Gambhir'



rohit=delivery[delivery['batsman']=='RG Sharma']

rohit=rohit[rohit['dismissal_kind'].isin(['caught','lbw','bowled','stumped','caught and bowled',"hit wicket"])]

rohit=rohit.groupby('bowler').count().sort_values(by='dismissal_kind',ascending=0).dismissal_kind[:1].reset_index()

rohit['batsman']='RG Sharma'



uthapa=delivery[delivery['batsman']=='RV Uthappa']

uthapa=uthapa[uthapa['dismissal_kind'].isin(['caught','lbw','bowled','stumped','caught and bowled',"hit wicket"])]

uthapa=uthapa.groupby('bowler').count().sort_values(by='dismissal_kind',ascending=0).dismissal_kind[:1].reset_index()

uthapa['batsman']='RV Uthappa'



dhawan=delivery[delivery['batsman']=='S Dhawan']

dhawan=dhawan[dhawan['dismissal_kind'].isin(['caught','lbw','bowled','stumped','caught and bowled',"hit wicket"])]

dhawan=dhawan.groupby('bowler').count().sort_values(by='dismissal_kind',ascending=0).dismissal_kind[:1].reset_index()

dhawan['batsman']='S Dhawan'



warn=delivery[delivery['batsman']=='DA Warner']

warn=warn[warn['dismissal_kind'].isin(['caught','lbw','bowled','stumped','caught and bowled',"hit wicket"])]

warn=warn.groupby('bowler').count().sort_values(by='dismissal_kind',ascending=0).dismissal_kind[:1].reset_index()

warn['batsman']='DA Warner'



new = gayle.append([kohli,raina,abd,msd,gg,rohit,uthapa,dhawan,warn])

new = new[['batsman','bowler','dismissal_kind']]

new.columns=['batsman','bowler','No_of_Dismissals']

new
mlt.subplots(figsize=(10,6))

bins=range(0,180,10)

mlt.hist(top_scores["batsman_runs"],bins,histtype="bar",rwidth=1.2,color='#0ff0ff')

mlt.xlabel('Runs')

mlt.ylabel('Count')

mlt.axvline(top_scores["batsman_runs"].mean(), color='b', linestyle='dashed', linewidth=2)

mlt.plot()

mlt.show()
orange=matches[['id','season']]

orange=orange.merge(delivery,left_on='id',right_on='match_id',how='left')

orange=orange.groupby(['season','batsman'])['batsman_runs'].sum().reset_index()

orange=orange.sort_values('batsman_runs',ascending=0)

orange=orange.drop_duplicates(subset=["season"],keep="first")

orange.sort_values(by='season')
mlt.subplots(figsize=(10,6))

dismissal_kinds = ["bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"]  #since run-out is not creditted to the bowler

ct=delivery[delivery["dismissal_kind"].isin(dismissal_kinds)]

ax=ct['bowler'].value_counts()[:10].plot.bar(width=0.8,color='#1167a0')

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.10, p.get_height()+1))

mlt.show()
eco=delivery.groupby(['bowler']).sum()

eco['total balls']=delivery['bowler'].value_counts()

eco['overs']=(eco['total balls']//6)

eco[eco['overs']>200].sort_values(by='overs',ascending=0)['overs'].head(5).reset_index()

eco['economy']=(eco['total_runs']/(eco['overs']))

eco[(eco['overs']>300)].sort_values('economy')[:10].economy.reset_index()
mlt.subplots(figsize=(10,6))

eco.replace([np.inf, -np.inf], np.nan,inplace=True)

eco.fillna(0,inplace=True)

bins=range(0,26)

mlt.hist(eco['economy'],bins,histtype="bar",rwidth=1.2,color='#0ff0ff')

mlt.xlabel('Economy')

mlt.ylabel('Count')

mlt.axvline(eco["economy"].mean(), color='b', linestyle='dashed', linewidth=2)

mlt.plot()

mlt.show()
dismissal_kinds = ["bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"]  #since run-out is not creditted to the bowler

purple=delivery[delivery["dismissal_kind"].isin(dismissal_kinds)]

purple=purple.merge(matches,left_on='match_id',right_on='id',how='outer')

purple=purple.groupby(['season','bowler'])['dismissal_kind'].count().reset_index()

purple=purple.sort_values('dismissal_kind',ascending=False)

purple=purple.drop_duplicates('season',keep='first').sort_values(by='season')

purple.columns=[['season','bowler','count_wickets']]

purple
mlt.subplots(figsize=(10,6))

extras=delivery[['wide_runs','bye_runs','legbye_runs','noball_runs']].sum()

sizes=[5161,680,3056,612]

mlt.pie(sizes, labels=['wide_runs','bye_runs','legbye_runs','noball_runs'],

        colors=['Y', '#1f2ff3', '#0fff00', 'R'],explode=(0,0,0,0),autopct='%1.1f%%', shadow=True, startangle=90)

mlt.title("Percentage of Extras")

fig = mlt.gcf()

fig.set_size_inches(6,6)

mlt.plot()

mlt.show()
mlt.subplots(figsize=(10,6))

dismiss=["run out","bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"]

ct=delivery[delivery["dismissal_kind"].isin(dismiss)]

ax=ct.dismissal_kind.value_counts()[:10].plot.bar(width=0.8,color='#005566')

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.1, p.get_height()+1))

mlt.show()
mlt.subplots(figsize=(10,6))

ax=delivery[delivery['batsman_runs']==6].batting_team.value_counts().plot.bar(width=0.8,color='G')

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x(), p.get_height()+10))

mlt.show()
finals=matches.drop_duplicates(subset=['season'],keep='last')

finals=finals[['id','season','city','team1','team2','toss_winner','toss_decision','winner']]

most_finals=pd.concat([finals['team1'],finals['team2']]).value_counts().reset_index()

most_finals.columns=[['team','count']]

xyz=finals['winner'].value_counts().reset_index()

most_finals=most_finals.merge(xyz,left_on='team',right_on='index',how='outer')

most_finals=most_finals.replace(np.NaN,0)

most_finals.drop('index',axis=1,inplace=True)

most_finals.set_index('team',inplace=True)

most_finals.columns=['finals_played','won_count']

most_finals.plot.bar(width=0.8)

fig=mlt.gcf()

fig.set_size_inches(10,6)

mlt.show()
df=finals[finals['toss_winner']==finals['winner']]

slices=[len(finals),(9-len(df))]

labels=['yes','no']

mlt.pie(slices,labels=labels,startangle=90,shadow=True,colors=['G','R'],explode=(0,0.1),autopct='%1.1f%%')

fig = mlt.gcf()

fig.set_size_inches(5,5)

mlt.show()
finals['is_tosswin_matchwin']=finals['toss_winner']==finals['winner']

sns.countplot(x='toss_decision',hue='is_tosswin_matchwin',data=finals)

mlt.show()