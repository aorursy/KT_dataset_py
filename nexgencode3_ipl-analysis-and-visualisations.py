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

df[[1,3,4,5,10,11]]
df=matches.iloc[[matches['win_by_wickets'].idxmax()]]

df[[1,3,4,5,10,12]]
print('Toss Decisions in %\n',((matches['toss_decision']).value_counts())/577*100)
sns.countplot(x='season',hue='toss_decision',data=matches)

mlt.show()
ax=matches['toss_winner'].value_counts().plot.bar(width=0.8)

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))

mlt.show()
w=matches['toss_winner'].value_counts()

m=pd.concat([matches['team1'],matches['team2']]).value_counts()

ax=(w/m*100).sort_values().plot.bar(width=0.8,color='#232323')

for p in ax.patches:                       #used to display the values on the top of the bar

    ax.annotate('{:.2f}%'.format(p.get_height()), (p.get_x(), p.get_height()+1),fontsize=9)

mlt.show()
df=matches[matches['toss_winner']==matches['winner']]

slices=[len(df),(577-len(df))]

labels=['yes','no']

mlt.pie(slices,labels=labels,startangle=90,shadow=True,explode=(0,0),autopct='%1.1f%%')

fig = mlt.gcf()

fig.set_size_inches(5,5)

mlt.show()
sns.countplot(x='season',data=matches,palette="Set1")  #countplot automatically counts the frequency of an item

sns.plt.show()
batsmen = matches[['id','season']].merge(delivery, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)

#merging the matches and delivery dataframe by referencing the id and match_id columns respectively

season=batsmen.groupby(['season'])['total_runs'].sum()

season.plot()

mlt.show()
ax = matches['venue'].value_counts().plot.bar(width=.8, color=["#999966", "#8585ad", "#c4ff4d", "#ffad33"])

ax.set_xlabel('Grounds')

ax.set_ylabel('count')

mlt.show()
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
sns.countplot(x='winner', data=matches)

mlt.xticks(rotation='vertical')

mlt.show()
ump=pd.concat([matches['umpire1'],matches['umpire2']]) 

ax=ump.value_counts().head(10).plot.bar(width=0.8,color='Y')

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+0.25))

mlt.show()
mt1=matches[((matches['team1']=='MI')|(matches['team2']=='MI'))&((matches['team1']=='KKR')|(matches['team2']=='KKR'))]

sns.countplot(x='season', hue='winner', data=mt1)

mlt.xticks(rotation='vertical')

mlt.show()
mt2=matches[((matches['team1']=='MI')|(matches['team2']=='MI'))&((matches['team1']=='CSK')|(matches['team2']=='CSK'))]

sns.countplot(x='season', hue='winner', data=mt2)

mlt.xticks(rotation='vertical')

mlt.show()
high_scores=delivery.groupby(['match_id', 'inning','batting_team','bowling_team'])['total_runs'].sum().reset_index() 

#reset_index() converts the obtained series into a dataframe

high_scores.nlargest(10,'total_runs')

#nlargest is used to sort the given column
high=delivery.groupby(['match_id', 'inning','batting_team'])['total_runs'].sum().reset_index()

high.set_index(['match_id'],inplace=True)

high['total_runs'].max()

high.columns

high=high.rename(columns={'total_runs':'count'})

high=high[high['count']>200].groupby(['inning','batting_team']).count()

high.T  #transpose
high_scores=delivery.groupby(['match_id', 'inning','batting_team','bowling_team'])['total_runs'].sum().reset_index()

high_scores1=high_scores[high_scores['inning']==1]

high_scores2=high_scores[high_scores['inning']==2]

high_scores1=high_scores1.merge(high_scores2[['match_id','inning', 'total_runs']], on='match_id')

high_scores1.rename(columns={'inning_x':'inning_1','inning_y':'inning_2','total_runs_x':'inning1_runs','total_runs_y':'inning2_runs'},inplace=True)

high_scores1=high_scores1[high_scores1['inning1_runs']>=200]

high_scores1['is_score_chased']=1

high_scores1['is_score_chased'] = np.where(high_scores1['inning1_runs']<=high_scores1['inning2_runs'], 

                                           'yes', 'no')

high_scores1.head(5)

slices=high_scores1['is_score_chased'].value_counts().reset_index().is_score_chased

list(slices)

labels=['target not chased','target chased']

mlt.pie(slices,labels=labels,colors=['#1f2ff3', '#0fff00'],startangle=90,shadow=True,explode=(0,0.1),autopct='%1.1f%%')

fig = mlt.gcf()

fig.set_size_inches(6,6)

mlt.show()
max_runs=delivery.groupby(['batsman'])['batsman_runs'].sum()

ax=max_runs.sort_values(ascending=False)[:10].plot.bar(width=0.8,color='R')

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.1, p.get_height()+1),fontsize=11)

mlt.show()
ax=delivery[delivery['batsman_runs']==6].batsman.value_counts()[:10].plot.bar(width=0.8,color='G')

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x(), p.get_height()+10))

mlt.show()

top_scores = delivery.groupby(["match_id", "batsman","batting_team"])["batsman_runs"].sum().reset_index()

top_scores.sort_values('batsman_runs', ascending=0).head(10)


a=batsmen.groupby(['season','batsman'])['batsman_runs'].sum().reset_index()

a=a.groupby(['season','batsman'])['batsman_runs'].sum().unstack().T

a['Total']=a.sum(axis=1)

a=a.sort_values(by='Total',ascending=0)[:5]

a.drop('Total',axis=1,inplace=True)

a.T.plot()

mlt.show()
a=batsmen.groupby(['batsman','batsman_runs'])['total_runs'].count().reset_index()

b=max_runs.sort_values(ascending=False)[:10].reset_index()

c=b.merge(a,left_on='batsman',right_on='batsman',how='left')

c.drop('batsman_runs_x',axis=1,inplace=1)

c.set_index('batsman',inplace=1)

c.columns=['type','count']

c=c[(c['type']==1)|(c['type']==2)|(c['type']==4)|(c['type']==6)]

cols=['type','count']

c
bins=range(0,180,10)

mlt.hist(top_scores["batsman_runs"],bins,histtype="bar",rwidth=1.2,color='#0ff0ff')

mlt.xlabel('Runs')

mlt.ylabel('Count')

mlt.axvline(top_scores["batsman_runs"].mean(), color='b', linestyle='dashed', linewidth=2)

mlt.plot()

mlt.show()
dismissal_kinds = ["bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"]  #since run-out is not creditted to the bowler

ct=delivery[delivery["dismissal_kind"].isin(dismissal_kinds)]

ax=ct['bowler'].value_counts()[:10].plot.bar(width=0.8,color='G')

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.10, p.get_height()+1))

mlt.show()
eco=delivery.groupby(['bowler']).sum()

eco['total balls']=delivery['bowler'].value_counts()

eco['overs']=(eco['total balls']//6)

eco[eco['overs']>200].sort_values(by='overs',ascending=0)[[14]].head(5).T
eco['economy']=(eco['total_runs']/(eco['overs']))

eco[(eco['overs']>300)].sort_values('economy')[:10].economy.reset_index().T
eco.replace([np.inf, -np.inf], np.nan,inplace=1)

eco.fillna(0,inplace=1)

bins=range(0,26)

mlt.hist(eco['economy'],bins,histtype="bar",rwidth=1.2,color='#0ff0ff')

mlt.xlabel('Economy')

mlt.ylabel('Count')

mlt.axvline(eco["economy"].mean(), color='b', linestyle='dashed', linewidth=2)

mlt.plot()

mlt.show()
extras=delivery[['wide_runs','bye_runs','legbye_runs','noball_runs']].sum()

sizes=[5161,680,3056,612]

mlt.pie(sizes, labels=['wide_runs','bye_runs','legbye_runs','noball_runs'],

        colors=['Y', '#1f2ff3', '#0fff00', 'R'],explode=(0,0,0,0),autopct='%1.1f%%', shadow=True, startangle=90)

mlt.title("Percentage of Extras")

fig = mlt.gcf()

fig.set_size_inches(6,6)

mlt.plot()

mlt.show()
dismiss=["run out","bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"]

ct=delivery[delivery["dismissal_kind"].isin(dismiss)]

ax=ct.dismissal_kind.value_counts()[:10].plot.bar(width=0.8,color='#005566')

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.1, p.get_height()+1))

mlt.show()
ax=delivery[delivery['batsman_runs']==6].batting_team.value_counts().plot.bar(width=0.8,color='G')

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x(), p.get_height()+10))

mlt.show()