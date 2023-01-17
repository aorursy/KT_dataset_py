



import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



df=pd.read_csv('/kaggle/input/ipldata/matches.csv')

df.loc[(df.winner=='Delhi Daredevils','winner')]='Delhi Capitals'

df.loc[(df.team1=='Delhi Daredevils','team1')]='Delhi Capitals'

df.loc[(df.team2=='Delhi Daredevils','team2')]='Delhi Capitals'

df.loc[(df.toss_winner=='Delhi Daredevils','toss_winner')]='Delhi Capitals'



df.loc[(df.winner=='Rising Pune Supergiant','winner')]='Rising Pune Supergiants'

df.loc[(df.team1=='Rising Pune Supergiant','team1')]='Rising Pune Supergiants'

df.loc[(df.team2=='Rising Pune Supergiant','team2')]='Rising Pune Supergiants'

df.loc[(df.toss_winner=='Rising Pune Supergiant','toss_winner')]='Rising Pune Supergiants'

teams=np.array(df.winner.value_counts().index.values)[0:8]

win=df.winner.value_counts()

home_wins=df[df.winner==df.team1].team1.value_counts()



names=['MI','CSK','KKR','RCB','KXIP','DC','RR','SRH']

color=pd.read_csv('/kaggle/input/colors/color.csv')

c=[color[i].values for i in names]



years=np.array(df.season.value_counts().index.values)

years.sort()





toss_winners=df[df.toss_winner==df.winner].winner.value_counts() # won toss won match

toss_choice=df.toss_decision.value_counts()

won_field=df[np.logical_or(np.logical_and((df.toss_winner==df.winner),(df.toss_decision=='field')),\

                           np.logical_and((df.winner!=df.toss_winner),(df.toss_decision!='field')))]



winner_by_season=pd.DataFrame(df[df.winner.isin(teams)].groupby('winner').season.value_counts().sort_index())



wickets=df[df.win_by_wickets!=0].win_by_wickets.values

runs=df[df.win_by_runs!=0].win_by_runs.values
sns.set_style('darkgrid')

plt.figure(figsize=(10,10))

plt.subplot('221')

plt.bar(names,win.values[0:8],color=c,alpha=.3,label='Total wins',edgecolor=c)

plt.bar(names,home_wins.values[0:8],color=c,label='Home wins',edgecolor=c)

plt.xticks(rotation=45)

plt.title("Matches Won")

plt.legend()



plt.subplot('222')



plt.bar(names,win.values[0:8],color=c,alpha=.3,label='Total wins',edgecolor=c)

plt.bar(names,toss_winners.values[0:8],color=c,label='toss wins',edgecolor=c)

plt.xticks(rotation=45)

plt.title("Matches Won")

plt.legend()



plt.subplot('223')



plt.bar(names,win.values[0:8],color=c,alpha=.3,label='Total wins',edgecolor=c)

plt.bar(names,won_field.winner.value_counts().values[0:8],color=c,label='Field first')

plt.xticks(rotation=45)

plt.title("Matches Won")

plt.legend()



plt.subplot('224')

s=list()

for i in teams:

    w=df[df.winner==i].winner.count()

    total=df[np.logical_or((df.team1==i),(df.team2==i))].winner.count()

    s.append(w*100/total)

plt.bar(names,s,color=c)

plt.title('Winning %')

plt.xticks(rotation=45)

#plt.yticks(range(0,101,10))

plt.gca().axis([-1,8,0,100])

sns.set_style('white')

winner_by_season=pd.DataFrame(df[df.winner.isin(teams)].groupby('winner').season.value_counts().sort_index())

d=df.groupby('season').winner.count().values

plt.figure(figsize=(7,7))

temp=pd.DataFrame(index=years)

temp['season']=0

a=np.zeros(12)

k=0

for i in teams:

    t=pd.DataFrame(winner_by_season.loc[i].season)

    mergedDf = temp.merge(t,how='left' ,left_index=True, right_index=True).fillna(0)



    

    plt.barh(years,d,label=names[k],color=c[k])

   # plt.plot(years,d)

    d=d-mergedDf.season_y.values

    

    k=k+1

plt.barh(years,d,color='black',label='others')

plt.legend()

plt.yticks(years)

plt.title('Matches Won by each team')

plt.margins(0)

plt.gca().spines["top"].set_visible(False)

plt.gca().spines["right"].set_visible(False)

plt.gca().spines["bottom"].set_visible(False)
sns.set_style('darkgrid')



plt.figure(figsize=(10,5))

plt.subplot('121')

sns.distplot(wickets, hist_kws={'color': 'Teal'}, kde_kws={'color': 'Navy'})

plt.xlabel('wickets')

plt.title("mean wickets - {}".format(int(wickets.mean())))

plt.plot([6,6],[0,.4],'--b')

plt.subplot('122')



sns.distplot(runs, hist_kws={'color': 'Teal'}, kde_kws={'color': 'Navy'})

plt.title("median runs - {}".format(int(np.median(runs))))

plt.plot([np.median(runs),np.median(runs)],[0,.030],'--b')

plt.xlabel('runs')

sns.set_style('darkgrid')

ranking =[[1,1,3,2,2,4,2,4,1,4,1,4],[3,2,4,4,3,3,2,3,0,0,4,3],[1,1,1,2,4,1,4,1,2,2,2,1],[1,3,1,3,1,1,1,2,3,1,1,1],[2,1,1,1,1,1,3,1,1,1,1,1],[2,2,1,1,2,1,1,1,1,1,1,2],[4,1,1,1,1,2,1,2,0,0,2,1],[0,0,0,0,0,2,1,1,4,2,3,2]]

ranking=np.array(ranking)

fig,ax=plt.subplots(4,2,figsize=(10,7),sharex=True,sharey=True)

t=0



plt.setp(ax, yticks=[0,1,2,3,4], yticklabels=['not played','group stage','semi-final','final','trophy'],xticks=years)

fig.suptitle("Team Standings at the end of each season",size=20)



for i in range(len(names)):

    s=i%4

    if i>3:

        t=1

    ax[s,t].scatter(years,ranking[i],label=names[i],c=c[i])

    

    ax[s,t].legend(loc = "upper right",framealpha = 0.70)

    ax[s,t].plot(years,ranking[i],'--',c=c[i],alpha=.5)

    ax[s,t].set_xticklabels(years, rotation=45)

    

    

        

plt.subplots_adjust(wspace=0.025, hspace=0.05)
head_to_head=pd.DataFrame(columns=list(teams)+['other'],index=list(teams)+['other']).fillna(0)

team1=df[df.winner.isin(teams)].team1.values

team2=df[df.winner.isin(teams)].team2.values

winner=df[df.winner.isin(teams)].winner.values



for a,b,c in zip(team1,team2,winner):

    if a not in teams:

        a='other'

    if b not in teams:

        b='other'

    if c==a:

        head_to_head.loc[c,b]+=1

    else:

        head_to_head.loc[c,a]+=1

head_to_head.drop(['other'],axis=1,inplace=True)
color=pd.read_csv('/kaggle/input/colors/color.csv')

c=[color[i].values for i in names]

sns.set_style("darkgrid")



fig ,ax=plt.subplots(1,8,figsize=(10,5),sharey=True)

plt.setp(ax, xticks=[0,5,10,15])

fig.suptitle("Head to Head Wins",size=20)

for i in range(8):

    #plt.setp(ax, yticks=[0,5,10,15])

    

    x=head_to_head.columns.values

   

    y=head_to_head.iloc[i].values

    y1=head_to_head.iloc[:,i].values[0:8]

    

    ax[i].barh(x,y1+y,color=c+[[0,0,0]],height=.3)

    ax[i].barh(x,y,color=c[i],height=.3)

    ax[i].set_yticklabels(names+['other'], rotation=45,size=15)

   # ax[i].set_xticklabels( rotation=45,size=15)

    #ax[i].set_ylabel(names[i])

    ax[i].set_title(names[i],size=15)

plt.setp(ax, xticks=[0,10,20])

plt.subplots_adjust(wspace=0.025, hspace=0.5)

sns.despine(left=True)

plt.margins(0)