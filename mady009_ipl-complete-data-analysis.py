import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sn

%matplotlib inline



# create a function for labeling #

def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,

                '%d' % int(height),

                ha='center', va='bottom')
data_path = "../input/"

match_df = pd.read_csv(data_path+"matches.csv")

score_df = pd.read_csv(data_path+"deliveries.csv")

score_df.head()
bowler_caught=score_df.groupby(['bowler'])['dismissal_kind'].agg(lambda a: (a=='caught').sum()).reset_index().sort_values(by='bowler',

                                                                                                                         ascending=True).reset_index(drop=True)



bowler_bowled=score_df.groupby(['bowler'])['dismissal_kind'].agg(lambda a: (a=='bowled').sum()).reset_index().sort_values(by='bowler',

                                                                                                                         ascending=True).reset_index(drop=True)



bowler_lbw=score_df.groupby(['bowler'])['dismissal_kind'].agg(lambda a: (a=='lbw').sum()).reset_index().sort_values(by='bowler',

                                                                                                                         ascending=True).reset_index(drop=True)



bowler_stumped=score_df.groupby(['bowler'])['dismissal_kind'].agg(lambda a: (a=='stumped').sum()).reset_index().sort_values(by='bowler',

                                                                                                                         ascending=True).reset_index(drop=True)



bowler_caught_and_bowled=score_df.groupby(['bowler'])['dismissal_kind'].agg(lambda a: (a=='caught and bowled').sum()).reset_index().sort_values(by='bowler',

                                                                                                                         ascending=True).reset_index(drop=True)
bowler_wickets=pd.DataFrame()
bowler_wickets['bowler']=bowler_caught['bowler']

bowler_wickets['caught']=bowler_caught['dismissal_kind']

bowler_wickets['bowled']=bowler_bowled['dismissal_kind']

bowler_wickets['lbw']=bowler_lbw['dismissal_kind']

bowler_wickets['c&b']=bowler_caught_and_bowled['dismissal_kind']

bowler_wickets['stumped']=bowler_stumped['dismissal_kind']

bowler_wickets['total_wickets']=bowler_wickets['caught']+bowler_wickets['bowled']+bowler_wickets['lbw']+bowler_wickets['c&b']+bowler_wickets['stumped']
bowler_wickets=bowler_wickets.sort_values(by='total_wickets',ascending=False)[:10]

bowler_wickets
#plotting

labels=np.array(bowler_wickets['bowler'])

ind=np.arange(len(labels))

values=np.array(bowler_wickets['total_wickets'])



fig,ax = plt.subplots()

rects = ax.bar(ind,values,color='b')

ax.set_xticklabels(labels,rotation='vertical')

ax.set_xticks(ind)

ax.set_title("Top 10 wicket takers")

ax.set_ylabel('wickets')

autolabel(rects)

plt.show()
catchers=score_df.groupby(['fielder'])['dismissal_kind'].agg(lambda a: (a=='caught').sum()).reset_index().sort_values(by='fielder',

                                                                                                                         ascending=True).reset_index(drop=True)

catchers=catchers.sort_values(by='dismissal_kind',ascending=False)[:10]
#plotting

labels=np.array(catchers['fielder'])

ind=np.arange(10)

values=np.array(catchers['dismissal_kind'])



fig,ax = plt.subplots()

rects = ax.bar(ind,values,color='b')

ax.set_xticklabels(labels,rotation='vertical')

ax.set_xticks(ind)

ax.set_title("Top 10 catchers")

ax.set_ylabel('catches')

autolabel(rects)

plt.show()
bowler_runs=score_df.groupby(['bowler'])['total_runs'].agg(lambda a: (a>0).sum()).reset_index().sort_values(by='bowler',

                                                                                ascending=True).reset_index(drop=True)



bowler_runs=bowler_runs.sort_values(by='total_runs',ascending=False)[:10]

bowler_runs
#plotting

labels=np.array(bowler_runs['bowler'])

ind=np.arange(len(labels))

values=np.array(bowler_runs['total_runs'])

fig,ax = plt.subplots()

rects = ax.bar(ind,values,color='r')

ax.set_xticklabels(labels,rotation='vertical')

ax.set_xticks(ind)

ax.set_title("Top 10 bowlers with more runs")

ax.set_ylabel('runs')

autolabel(rects)

plt.show()
temp=score_df.groupby(['match_id','inning','over','bowler'])['total_runs'].agg('sum').reset_index().sort_values(by='total_runs',

                                                                                            ascending=False).reset_index(drop=True)

temp=temp[:10]

temp
#plotting

labels=np.array(temp['bowler'])

ind=np.arange(len(labels))

values=np.array(temp['total_runs'])



fig,ax = plt.subplots()

rects = ax.bar(ind,values,color='r')

ax.set_xticklabels(labels,rotation='vertical')

ax.set_xticks(ind)

ax.set_title("Top 10 bowlers with Highest runs in an over")

ax.set_ylabel('runs')

autolabel(rects)

plt.show()
temp=score_df.groupby(['match_id','inning','batsman'])['batsman_runs'].agg('sum').reset_index().sort_values(by='batsman_runs',

                                                                                            ascending=False).reset_index(drop=True)

temp=temp.drop_duplicates(subset='batsman', keep='first', inplace=False)[:10]

temp
#plotting

labels=np.array(temp['batsman'])

ind=np.arange(len(labels))

values=np.array(temp['batsman_runs'])



fig,ax = plt.subplots()

rects = ax.bar(ind,values,color='g')

ax.set_xticklabels(labels,rotation='vertical')

ax.set_xticks(ind)

ax.set_title("Top 10 batsmen with Highest runs in an innings")

ax.set_ylabel('runs')

autolabel(rects)

plt.show()
temp=score_df.groupby(['match_id','inning','batsman'])['batsman_runs'].agg('sum').reset_index().sort_values(by='batsman_runs',

                                                                                            ascending=False).reset_index(drop=True)



temp['no_of_100s']=0

temp['no_of_100s'].loc[temp['batsman_runs']>=100]=1

temp=temp.groupby(['batsman'])['no_of_100s'].agg(lambda a:(a==1).sum()).reset_index().sort_values(by='no_of_100s',

                                                                                                 ascending=False).reset_index(drop=True)

temp=temp[:9]
#plotting

labels=np.array(temp['batsman'])

ind=np.arange(len(labels))

values=np.array(temp['no_of_100s'])



fig,ax = plt.subplots()

rects = ax.bar(ind,values,color='g')

ax.set_xticklabels(labels,rotation='vertical')

ax.set_xticks(ind)

ax.set_title("Top batsmen with 2 or more than 2 centuries")

ax.set_ylabel('Count')

autolabel(rects)

plt.show()


temp_df = score_df.groupby('batsman')['batsman_runs'].agg('sum').reset_index().sort_values(by='batsman_runs', 

                                                                                           ascending=False).reset_index(drop=True)
temp_df=temp_df[:10]

labels=np.array(temp_df['batsman'])

values=np.array(temp_df['batsman_runs'])

ind=np.arange(len(labels))



fig,ax=plt.subplots()

rects=ax.bar(ind,values,color='g')

ax.set_xticks(ind)

ax.set_xticklabels(labels,rotation='vertical')

ax.set_ylabel("Runs")

ax.set_title("Top run scorers in IPL")

autolabel(rects)

plt.show()
temp=score_df.groupby(['match_id','inning','batsman'])['batsman_runs'].agg('sum').reset_index().sort_values(by='batsman_runs',

                                                                                            ascending=False).reset_index(drop=True)



temp['no_of_50s']=0

temp['no_of_50s'].loc[temp['batsman_runs']>=50]=1

temp=temp.groupby(['batsman'])['no_of_50s'].agg(lambda a:(a==1).sum()).reset_index().sort_values(by='no_of_50s',

                                                                                                 ascending=False).reset_index(drop=True)

temp=temp[:10]

temp
#plotting

labels=np.array(temp['batsman'])

ind=np.arange(len(labels))

values=np.array(temp['no_of_50s'])



fig,ax = plt.subplots()

rects = ax.bar(ind,values,color='g')

ax.set_xticklabels(labels,rotation='vertical')

ax.set_xticks(ind)

ax.set_title("Top 10 batsmen with highest no of 50's")

ax.set_ylabel('Count')

autolabel(rects)

plt.show()
temp_df = score_df.groupby('bowler')['wide_runs'].agg(lambda x: (x>0).sum()).reset_index().sort_values(by='wide_runs', ascending=False).reset_index(drop=True)

temp_df = temp_df.iloc[:10,:]



labels = np.array(temp_df['bowler'])

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(temp_df['wide_runs']), width=width, color='r')

ax.set_xticks(ind)

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Count")

ax.set_title("Bowlers with more wides in IPL")

autolabel(rects)

plt.show()
#top players for 4's

temp_df = score_df.groupby('batsman')['batsman_runs'].agg(lambda a: (a==4).sum()).reset_index().sort_values(by='batsman_runs', 

                                                                                           ascending=False).reset_index(drop=True)



temp_df=temp_df[:10]

labels=np.array(temp_df['batsman'])

values=np.array(temp_df['batsman_runs'])

ind=np.arange(len(labels))



fig,ax=plt.subplots()

rects=ax.bar(ind,values,color='g')

ax.set_xticks(ind)

ax.set_xticklabels(labels,rotation='vertical')

ax.set_ylabel("4's count")

ax.set_title("Batsman with most number of fours")

autolabel(rects)

plt.show()
#top players for 6's

temp_df = score_df.groupby('batsman')['batsman_runs'].agg(lambda a: (a==6).sum()).reset_index().sort_values(by='batsman_runs', 

                                                                                           ascending=False).reset_index(drop=True)



temp_df=temp_df[:10]

labels=np.array(temp_df['batsman'])

values=np.array(temp_df['batsman_runs'])

ind=np.arange(len(labels))



fig,ax=plt.subplots()

rects=ax.bar(ind,values,color='b')

ax.set_xticks(ind)

ax.set_xticklabels(labels,rotation='vertical')

ax.set_ylabel("6's count")

ax.set_title("Batsman with most number of sixes")

autolabel(rects)

plt.show()
#how many balls faced by each batsman

balls_faced_df=score_df.groupby(['batsman'])['wide_runs'].agg(lambda a:(a==0).sum()).reset_index().sort_values(by='batsman', 

                                                                                        ascending=True).reset_index(drop=True)



noballs_faced_df=score_df.groupby(['batsman'])['noball_runs'].agg(lambda a:(a!=0).sum()).reset_index().sort_values(by='batsman', 

                                                                                        ascending=True).reset_index(drop=True)



batsman_df=score_df.groupby(['batsman'])['batsman_runs'].agg('sum').reset_index().sort_values(by='batsman',

                                                                                        ascending=True).reset_index(drop=True)



balls_faced_df['total_balls_faced']=0

balls_faced_df['total_balls_faced'].loc[balls_faced_df['batsman']==noballs_faced_df['batsman']]=balls_faced_df['wide_runs']-noballs_faced_df['noball_runs']



del balls_faced_df['wide_runs']



balls_faced_df['batsman_runs']=batsman_df['batsman_runs']

balls_faced_df=balls_faced_df.sort_values(by='total_balls_faced',ascending=False)[:10]

balls_faced_df
balls_faced_df['more_runs']=balls_faced_df['batsman_runs']-balls_faced_df['total_balls_faced']

balls_faced_df
labels=balls_faced_df['batsman']

ind=np.arange(len(labels))

values=balls_faced_df['total_balls_faced']

values2=balls_faced_df['batsman_runs']



fig,ax=plt.subplots()

ax2 = ax.twinx()

rects=ax.bar(ind,values,width=0.35,color='b')

ax.set_xticks(ind)

ax.set_xticklabels(labels,rotation='vertical')

ax.set_title("Most total balls and runs")

ax.set_ylabel('count')



rects1=ax.bar(ind+0.35,values2,width=0.35,color='g')



autolabel(rects)

autolabel(rects1)



ax2.plot(ind+0.35, np.array(balls_faced_df['more_runs']), color='r', marker='o')

ax2.set_ylabel("More runs", color='r')

#ax2.set_ylim([0,100])

#ax2.grid(b=False)

plt.show()
temp_df = score_df.groupby('bowler')['extra_runs'].agg(lambda x: (x>0).sum()).reset_index().sort_values(by='extra_runs', ascending=False).reset_index(drop=True)

temp_df = temp_df.iloc[:10,:]



labels = np.array(temp_df['bowler'])

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(temp_df['extra_runs']), width=width, color='r')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Count")

ax.set_title("Bowlers with more extras in IPL")

autolabel(rects)

plt.show()
plt.figure(figsize=(12,6))

sn.countplot(x='dismissal_kind', data=score_df)

plt.xticks(rotation='vertical')

plt.show()
match_df[:2]
# Let us get some basic stats #

print("Number of matches played so far : ", match_df.shape[0])

print("Number of seasons : ", len(match_df.season.unique()))

print('number of cities : ',len(match_df["city"].unique()))
matches_count=match_df.groupby(['season'])

matches_count_season=matches_count['id'].count()

matches_count_season.sort_values(ascending=False)

# or match_df['season'].value_counts()
#no of macthes in each season

sn.countplot(x='season',data=match_df)

plt.show()
plt.plot(matches_count_season)

plt.ylim(40,80)

plt.show()
match_count=match_df.groupby(['venue'])

match_count_venue=match_count['id'].count()

match_count_venue.sort_values(ascending=False)

#or match_df['venue'].value_counts()
plt.figure(figsize=(12,6))

sn.countplot(x='venue', data=match_df)

plt.xticks(rotation='vertical')

plt.show()
melt_df = pd.melt(match_df, id_vars=['id','season'], value_vars=['team1', 'team2'])

melt_df
t=melt_df.groupby(['value'])

t=t['id'].count()

t.sort_values(ascending=False)
plt.figure(figsize=(12,6))

sn.countplot(x='value', data=melt_df)

plt.xticks(rotation='vertical')

plt.show()
win_count=match_df.groupby(['winner'])

win_count_teams=win_count['id'].count()

win_count_teams.sort_values(ascending=False)

#or match_df['winner'].value_counts()
plt.figure(figsize=(12,6))

sn.countplot(x='winner',data=match_df)

plt.xticks(rotation='vertical')

plt.show()
melt_df = pd.melt(match_df, id_vars=['id','season','winner'], value_vars=['team1', 'team2'])

melt_df
t=melt_df.groupby(['winner'])

t=(t['id'].count()/2)

t.sort_values(ascending=False)
t=melt_df.groupby(['winner'])

t1=melt_df.groupby(['value'])

t=(t['id'].count()/2)/(t1['id'].count())*100
t.sort_values(ascending=False)
labels=(np.array(t.keys()))

win_pers=(np.array(t))

ind = np.arange(len(labels))



fig,ax=plt.subplots()

rects = ax.bar(ind,win_pers,width=0.9, color='g')

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Percentage")

ax.set_xticks(ind)

autolabel(rects)

plt.show()
toss_count=match_df.groupby(['toss_decision'])

toss_count=toss_count['id'].count()

toss_count
plt.figure(figsize=(5,4))

sn.countplot(x='toss_decision', data=match_df)

plt.xticks(rotation='vertical')

plt.show()
ts=match_df.toss_decision.value_counts()

ts
ts.index
labels=(np.array(ts.index))

labels
sizes=np.array(((ts/ts.sum())*100))

sizes
colors = ['gold', 'lightskyblue']

plt.pie(sizes,labels=labels,colors=colors,

       autopct='%1.1f%%', shadow=True, startangle=90)

plt.title("Toss decision percentage")

plt.show()
plt.figure(figsize=(12,6))

sn.countplot(x='season', hue='toss_decision', data=match_df)

plt.xticks(rotation='vertical')

plt.show()
#wins over batting second or filed first

num_of_wins = (match_df.win_by_wickets>0).sum()

num_of_loss = (match_df.win_by_wickets==0).sum()



labels = ["Wins", "Loss"]

total = float(num_of_wins + num_of_loss)

sizes = [(num_of_wins/total)*100, (num_of_loss/total)*100]



colors = ['gold', 'lightskyblue']

plt.pie(sizes, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=90)

plt.title("Win percentage batting second")

plt.show()
tp=match_df['player_of_match'].value_counts()[:10]

labels=np.array(tp.keys())

values=np.array(tp)

ind=np.arange(len(labels))



fig,ax=plt.subplots()

rects=ax.bar(ind,values,color='g')

ax.set_xticklabels(labels,rotation='vertical')

ax.set_xticks(ind)

ax.set_ylabel("Count")

ax.set_title("Top players of the matches")

autolabel(rects)

plt.show()
#top empires

temp_df = pd.melt(match_df, id_vars=['id'], value_vars=['umpire1', 'umpire2'])

temp_series = temp_df.value.value_counts()[:10]

labels = np.array(temp_series.index)

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(temp_series), width=width, color='r')

ax.set_xticks(ind)

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Count")

ax.set_title("Top Umpires")

autolabel(rects)

plt.show()
match_df['toss_winner_is_winner']='n'

match_df['toss_winner_is_winner'].loc[match_df['toss_winner']==match_df['winner']]='y'

temp=match_df['toss_winner_is_winner'].value_counts()

labels=np.array(temp.index)

values=np.array((temp/temp.sum())*100)

colors = ['gold', 'lightskyblue']



plt.pie(sizes, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=90)

plt.title("Toss winner is match winner")

plt.show()
plt.figure(figsize=(12,6))

sn.countplot(x='toss_winner', hue='toss_winner_is_winner', data=match_df)

plt.xticks(rotation='vertical')

plt.show()
match_df['win_by']="tie"

match_df['win_by'].loc[match_df['win_by_runs']>0]='runs'

match_df['win_by'].loc[match_df['win_by_wickets']>0]='wickets'
plt.figure(figsize=(12,6))

sn.countplot(x='winner', hue='win_by', data=match_df)

plt.xticks(rotation='vertical')

plt.show()