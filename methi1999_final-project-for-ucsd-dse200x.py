#Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Read data
matches = pd.read_csv('../input/matches.csv')
matches = matches.drop(['date','win_by_runs','win_by_wickets','venue','umpire3','id'],axis=1)
# matches.head(10)
matches.replace('Rising Pune Supergiant','Rising Pune Supergiants',inplace=True)
matches = matches.fillna({'city':'Dubai'})
matches = matches.dropna(subset=['winner'])
matches=matches.reset_index()
# matches['winner'].unique()
# matches.isnull().any()
seasons_list = [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017]
seasonwise_slices = [matches[matches['season'] == s] for s in seasons_list]
teams_list = ['Sunrisers Hyderabad', 'Rising Pune Supergiants',
       'Kolkata Knight Riders', 'Kings XI Punjab',
       'Royal Challengers Bangalore', 'Mumbai Indians', 'Delhi Daredevils',
       'Gujarat Lions', 'Chennai Super Kings', 'Rajasthan Royals',
       'Deccan Chargers', 'Pune Warriors', 'Kochi Tuskers Kerala']
teams_abb = ['SRH','RPS','KKR','KXIP','RCB','MI','DD','GL','CSK','RR','DC','PW','KTK']
team_dict=dict(zip(teams_list,teams_abb))
pc_wins_list = []
for season in seasonwise_slices:
    noofmatches = [season[season['team1'] == team_name].shape[0] + season[season['team2'] == team_name].shape[0] for team_name in teams_list]
    noofwins = [season[season['winner'] == team_name].shape[0] for team_name in teams_list]
    #If team did not play that season, no need to plot
    noofmatches = [np.nan if x == 0 else x for x in noofmatches]
    winpc = [100*x/y for x,y in zip(noofwins,noofmatches)]
    pc_wins_list.append(winpc)
# pc_wins_list
#Plot wins percent data
for i in range(0,len(pc_wins_list)):
    plt.scatter(teams_abb,pc_wins_list[i],label=str(2008+i),marker='x',s=40)

plt.ylabel("% of games won")
plt.ylim([0,90])
plt.xlabel('Teams')
plt.title('Consistency of teams across seasons',fontdict={"fontsize":18})
# plt.xticks(=90)
plt.legend(loc=9, bbox_to_anchor=(1.1, 1.02), ncol=1)
plt.show()
dl = matches[matches['dl_applied'] == 1]
total_dl_matches = dl.shape[0]
fieldfirst = dl[dl['toss_decision'] == 'field']
batfirst = dl[dl['toss_decision'] == 'bat']
fieldfirst_wins = fieldfirst[fieldfirst['toss_winner'] == fieldfirst['winner']].shape[0]
batfirst_losses = batfirst[batfirst['toss_winner'] != batfirst['winner']].shape[0]
dl_fieldfirst_wins = fieldfirst_wins + batfirst_losses
def iden(x):
    return (str(x) + '%')
plt.pie([dl_fieldfirst_wins,total_dl_matches-dl_fieldfirst_wins],labels = ['Bowl First','Bat First'],colors=['orange','yellow'],autopct=iden,explode=[0,0.1])
plt.title('Do DL methods favour the field-first team?',fontdict={"fontsize":18})
plt.show()
mom = matches['player_of_match'].value_counts()
#Pick top 10 players
mom = mom.iloc[:10]
#mom.index
sns.set_style('darkgrid')
ax = sns.barplot(x=mom.index,y=mom)
ax.set(xlabel='Player', ylabel='Number of Man of the Match awards')
plt.title('Match winners',fontdict={"fontsize":18})
plt.yticks(range(0,20,2))
plt.xticks(rotation=45)
plt.show()
umpire1 = matches['umpire1'].value_counts()
umpire2 = matches['umpire2'].value_counts()
df = pd.concat([umpire1,umpire2],axis=1)
df = df.fillna(0)
df['Sum'] = df['umpire1'] + df['umpire2']
df = df.sort_values(['Sum'])
#Pick top 10 umpires
finalump = df.iloc[-10:]
plt.bar(height=finalump['Sum'],x=finalump.index,color='green')
plt.ylabel("Number of matches umpired")
plt.yticks(range(0,90,10))
plt.xlabel('Umpire')
plt.title('Umpires with most matches',fontdict={"fontsize":18})
plt.xticks(rotation=90)
plt.show()
tosswin_byseason=[]
for season in seasonwise_slices:
    noofmatches = season.shape[0]
    tosswinners = season[season['toss_winner'] == season['winner']].shape[0]
    tosswin_byseason.append(100*tosswinners/noofmatches)
plt.plot(seasons_list,tosswin_byseason,marker='o')
plt.xlabel('Season')
plt.xticks(range(2008,2018))
plt.ylabel('% matches won by team winning the toss')
plt.yticks(range(0,80,10))
plt.title('Does winning the toss affect the result?',fontdict={"fontsize":18})
plt.show()
batfirstwins=[]
for season in seasonwise_slices:
    noofmatches=season.shape[0]
    mask1=(season['toss_decision'] == 'bat')
    mask2=(season['toss_winner'] == season['winner'])
    tossbatwin = season[mask1 & mask2].shape[0]
    mask3=(season['toss_decision'] == 'field')*1
    mask4=(season['toss_winner'] != season['winner'])*1
    tossfieldlose = mask3.dot(mask4)
    totalwinsbatfirst = tossbatwin+tossfieldlose
    batfirstwins.append(100*totalwinsbatfirst/noofmatches)
# print(batfirstwins)
plt.plot(seasons_list,batfirstwins,marker='o',color='green')
plt.xlabel('Season')
plt.xticks(range(2008,2018))
plt.ylabel('% matches won by team batting first')
plt.yticks(range(20,80,10))
plt.title('Is batting first a good decision?',fontdict={"fontsize":18})
plt.show()
# matches['city'].value_counts()
#Drop overseas values
matches_nover = matches.replace(['Dubai','Abu Dhabi','Bloemfontein','East London','Kimberley','Sharjah','Cape Town','Port Elizabeth','Johannesburg','Centurion','Durban'],'Overseas')
newmatches = matches_nover[matches_nover['city'] != 'Overseas']
#Cuttack has no fixed home team
newmatches = newmatches[newmatches['city'] != 'Cuttack']
#Drop unnecessary fields:
newmatches=newmatches.drop(['toss_decision','result','dl_applied','player_of_match','umpire1','umpire2'],axis=1)
# newmatches.shape
# newmatches['city'].value_counts()
#By examining the home teams of each ground, we get the following groups (for established teams):
matchreplace = newmatches.replace(['Mumbai'],'Mumbai Indians')
matchreplace = matchreplace.replace(['Bangalore'],'Royal Challengers Bangalore')
matchreplace = matchreplace.replace(['Kolkata'],'Kolkata Knight Riders')
matchreplace = matchreplace.replace(['Delhi','Raipur'],'Delhi Daredevils')
matchreplace = matchreplace.replace(['Chennai','Ranchi'],'Chennai Super Kings')
matchreplace = matchreplace.replace(['Chandigarh','Dharamsala','Indore'],'Kings XI Punjab')
matchreplace = matchreplace.replace(['Jaipur','Ahmedabad'],'Rajasthan Royals')
matchreplace = matchreplace.replace(['Rajkot','Kanpur'],'Gujarat Lions')
matchreplace = matchreplace.replace(['Kochi'],'Kochi Tuskers Kerala')
#Temporarily drop SRH,DC and RPS,PW fields
temp = matchreplace.copy(deep=True)
matchreplace = matchreplace[(matchreplace['city'] != 'Hyderabad')&(matchreplace['city'] != 'Pune')&(matchreplace['city'] != 'Nagpur')&(matchreplace['city'] != 'Visakhapatnam')]
# matchreplace
slice1 = temp[(temp['season']>2012) & ((temp['city'] == 'Nagpur')|(temp['city'] == 'Visakhapatnam')|(temp['city'] == 'Hyderabad')) ].replace(['Hyderabad','Visakhapatnam','Nagpur'],'Sunrisers Hyderabad')
slice2 = temp[(temp['season']<2012) & ((temp['city'] == 'Nagpur')|(temp['city'] == 'Visakhapatnam')|(temp['city'] == 'Hyderabad')) ].replace(['Hyderabad','Visakhapatnam','Nagpur'],'Deccan Chargers')
merge1 = pd.concat([slice1,slice2])
slice3 = temp[(temp['season']>2014) & (temp['city'] == 'Pune') ].replace('Pune','Rising Pune Supergiants')
slice4 = temp[(temp['season']<2014) & (temp['city'] == 'Pune') ].replace('Pune','Pune Warriors')
merge2 = pd.concat([slice3,slice4])
bothmerged = pd.concat([merge1,merge2])
fdata = pd.concat([matchreplace,bothmerged])
#Gathering data:
winpclisthome=[]
winpclistaway=[]
for team in teams_list:
    noofhomematches = fdata[(fdata['city'] == team) & ((fdata['team1'] == team) | (fdata['team2'] == team))].shape[0]
    noofawaymatches = fdata[(fdata['city'] != team) & ((fdata['team1'] == team) | (fdata['team2'] == team))].shape[0]
    noofhomewins = fdata[(fdata['city'] == team) & ((fdata['team1'] == team) | (fdata['team2'] == team)) & (fdata['winner'] == team)].shape[0]
    noofawaywins = fdata[(fdata['city'] != team) & ((fdata['team1'] == team) | (fdata['team2'] == team)) & (fdata['winner'] == team)].shape[0]
    winpclisthome.append(100*noofhomewins/noofhomematches)
    winpclistaway.append(100*noofawaywins/noofawaymatches)
# winpclist
noofteams = len(teams_abb)
groups = 2*np.arange(noofteams)
barwidth=0.5
rects1 = plt.bar(groups, winpclisthome, barwidth,
                 alpha=0.5,
                 color='b',
                 label='Home')
rects2 = plt.bar(groups+barwidth, winpclistaway, barwidth,
                 alpha=0.5,
                 color='r',
                 label='Away')

plt.xlabel('Teams')
plt.ylabel('% of games won')
plt.ylim([0,80])
plt.title('Does home ground matter?',fontdict={"fontsize":18})
plt.xticks(groups + barwidth / 2, teams_abb,)
plt.legend()

plt.tight_layout()
plt.show()
plt.show()
dat = pd.read_csv('../input//deliveries.csv')
dat.shape
most_runs=dat[['batsman','batsman_runs']]
most_runs.insert(1,'balls',np.ones(most_runs.shape[0]))
most_runs_grouped=most_runs.groupby('batsman',as_index=False).sum()
most_runs_grouped['strike_rate']=100*most_runs_grouped['batsman_runs']/most_runs_grouped['balls']
final_highest_runs=most_runs_grouped.sort_values(by=['batsman_runs'],ascending=False)
final_higest_sr=most_runs_grouped[most_runs_grouped['batsman_runs']>1000].sort_values(by=['strike_rate'],ascending=False)
plot_sr=final_higest_sr.iloc[:10]
plot_runs=final_highest_runs.iloc[:10]
# plot_sr.head(10)
sns.set_style('darkgrid')
ax = sns.barplot(x=plot_sr['batsman'],y=plot_sr['strike_rate'])
ax.set(xlabel='Batsman', ylabel='Strike Rate(runs per 100 balls)')
plt.title('Hard hitters(Atleast 1000 runs)',fontdict={"fontsize":18})
plt.ylim([120,160])
plt.yticks(range(120,161,5))
plt.xticks(rotation=45)
plt.show()
sns.set_style('darkgrid')
ax = sns.barplot(x=plot_runs['batsman'],y=plot_runs['batsman_runs'])
ax.set(xlabel='Batsman', ylabel='Total runs in IPL')
plt.title('Most runs',fontdict={"fontsize":18})
plt.ylim([3000,4700])
plt.yticks(range(3000,4701,200))
plt.xticks(rotation=45)
plt.show()
bowling=dat[['bowler','dismissal_kind','total_runs']]
bowling.insert(1,'balls',np.ones(bowling.shape[0]))
bowling=bowling.fillna(0)
#Consider only wickets taken by bowlers
bowling=bowling.replace(['caught','bowled','stumped','caught and bowled','lbw','hit wicket'],1)
bowling=bowling.replace(['run out','retired hurt','obstructing the field'],0)
bowling_grouped=bowling.groupby('bowler',as_index=False).sum()
bowling_grouped['economy']=6*bowling_grouped['total_runs']/bowling_grouped['balls']

mosteconomical=(bowling_grouped[bowling_grouped['balls']>840].sort_values(by=['economy']))
mostwickets=(bowling_grouped.sort_values(by=['dismissal_kind'],ascending=False))
# mostwickets


# final_higest_sr=most_runs_grouped[most_runs_grouped['batsman_runs']>1000].sort_values(by=['strike_rate'],ascending=False)
sns.set_style('whitegrid')
ax = sns.barplot(x=mostwickets['bowler'].iloc[:10],y=mostwickets['dismissal_kind'].iloc[:10])
ax.set(xlabel='Bowler', ylabel='Wickets')
plt.title('Most wickets',fontdict={"fontsize":18})
plt.ylim([80,160])
plt.yticks(range(80,171,10))
plt.xticks(rotation=45)
plt.show()
sns.set_style('whitegrid')
ax = sns.barplot(x=mosteconomical['bowler'].iloc[:10],y=mosteconomical['economy'].iloc[:10])
ax.set(xlabel='Bowler', ylabel='Economy')
plt.title('Best Economy Rate(Atleast 140 overs)',fontdict={"fontsize":18})
plt.ylim([6,7.5])
plt.yticks([6,6.3,6.6,6.9,7.2,7.5])
plt.xticks(rotation=45)
plt.show()
dismissals=pd.DataFrame(dat['dismissal_kind'].value_counts())
dismissals=dismissals.reset_index()
dismissals.columns=['Kind','Frequency']
dismissals.iloc[0],dismissals.iloc[6]=dismissals.iloc[6],dismissals.iloc[0]
dismissals.iloc[3],dismissals.iloc[8]=dismissals.iloc[8],dismissals.iloc[3]
dismissals.iloc[3],dismissals.iloc[2]=dismissals.iloc[2],dismissals.iloc[3]

# dismissals
def iden(x):
    return (str(round(x,2)) + '%')
fig1, ax1 = plt.subplots()
# ax1.pie(dismissals['Frequency'], labels=dismissals['Kind'], autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.pie(x=dismissals['Frequency'],labels = dismissals['Kind'],autopct=iden,radius=1.5,pctdistance=0.8,center=(0.5,0.5))
# colors=['orange','yellow'],autopct=iden,explode=list(range(0,0.6,0.1))
plt.title('Distribution of dismissals',y=1.2,fontdict={"fontsize":18})
plt.show()
firstin=dat[dat['inning']==1][['match_id','total_runs']]
firstin=firstin.groupby(by=['match_id'],as_index=False).sum()
secondin=dat[dat['inning']==2][['match_id','total_runs']]
secondin=secondin.groupby(by=['match_id'],as_index=False).sum()

firstmax=firstin.sort_values(by='total_runs',ascending=False).iloc[:10]
secondmax=secondin.sort_values(by='total_runs',ascending=False).iloc[:10]
firstmax_bat=[dat[(dat['match_id']==x) & (dat['inning']==1) ].iloc[0]['batting_team'] for x in firstmax['match_id']]
firstmax_bowling=[dat[(dat['match_id']==x) & (dat['inning']==1) ].iloc[0]['bowling_team'] for x in firstmax['match_id']]
# secondmax_teams=[dat[(dat['match_id']==x) & (dat['inning']==2) ].iloc[0]['batting_team'] for x in secondmax['match_id']]
# firstmax_bowling
abbfirstmax_bat=[team_dict[x] for x in firstmax_bat]
abbfirstmax_bowl=[team_dict[x] for x in firstmax_bowling]
vsformat=[(x + ' vs ' + y ) for x,y in zip(abbfirstmax_bat,abbfirstmax_bowl)]
plt.bar(vsformat,firstmax['total_runs'])
plt.xlabel('Batting vs Bowling team')
plt.ylabel('Score')
plt.title('Highest 1st innings score',fontdict={"fontsize":18})
plt.ylim([200,270])
plt.yticks(range(200,280,10))
plt.xticks(rotation=45)
plt.show()
deathdata1=dat[(dat['over']>15) & (dat['inning']==1)][['match_id','total_runs']]
# deathdata2=dat[(dat['over']>15) & (dat['inning']==2)][['match_id','total_runs']]

firstindeath=deathdata1.groupby(by=['match_id'],as_index=False).sum()
# secondindeath=deathdata2.groupby(by=['match_id'],as_index=False).sum()
fidmax=firstindeath.sort_values(by='total_runs',ascending=False).iloc[:10]
# sidmax=secondindeath.sort_values(by='total_runs',ascending=False).iloc[:10]
team_dict['Rising Pune Supergiant']='RPS'
deathbatfirst=[dat[(dat['match_id']==x) & (dat['inning']==1) ].iloc[0]['batting_team'] for x in fidmax['match_id']]
deathbowlfirst=[dat[(dat['match_id']==x) & (dat['inning']==1) ].iloc[0]['bowling_team'] for x in fidmax['match_id']]
abbdeathbatfirst=[team_dict[x] for x in deathbatfirst]
abbdeathbowlfirst=[team_dict[x] for x in deathbowlfirst]
vsformat=[(x + ' vs ' + y ) for x,y in zip(abbdeathbatfirst,abbdeathbowlfirst)]

plt.bar(vsformat,fidmax['total_runs'],color='red')
plt.xlabel('Batting vs Bowling team')
plt.ylabel('Runs scored in last 5 overs')
plt.title('Death overs carnage',fontdict={"fontsize":18})
plt.ylim([60,120])
plt.yticks(range(60,120,10))
plt.xticks(rotation=45)
plt.show()
most_runs_death=dat[dat['over']>15][['batsman','batsman_runs']]
most_runs_death.insert(1,'balls',np.ones(most_runs_death.shape[0]))
most_runs_death_grouped=most_runs_death.groupby('batsman',as_index=False).sum()
most_runs_death_grouped['strike_rate']=100*most_runs_death_grouped['batsman_runs']/most_runs_death_grouped['balls']
final_highest_runs_death=most_runs_death_grouped.sort_values(by=['batsman_runs'],ascending=False)
final_higest_sr_death=most_runs_death_grouped[most_runs_death_grouped['batsman_runs']>400].sort_values(by=['strike_rate'],ascending=False)

plot_sr=final_higest_sr_death.iloc[:10]
plot_runs=final_highest_runs_death.iloc[:10]
# plot_sr.head(10)

sns.set_style('darkgrid')
ax = sns.barplot(x=plot_sr['batsman'],y=plot_sr['strike_rate'])
ax.set(xlabel='Batsman', ylabel='Strike Rate(runs per 100 balls)')
plt.title('Highest strike rate in the last 5 overs',fontdict={"fontsize":18})
plt.ylim([150,220])
plt.yticks(range(150,221,10))
plt.xticks(rotation=45)
plt.show()
sns.set_style('darkgrid')
ax = sns.barplot(x=plot_runs['batsman'],y=plot_runs['batsman_runs'])
ax.set(xlabel='Batsman', ylabel='Total runs')
plt.title('Most runs in the last 5 overs',fontdict={"fontsize":18})
plt.ylim([500,2500])
plt.yticks(range(500,2501,200))
plt.xticks(rotation=45)
plt.show()
bowling_death=dat[dat['over']>15][['bowler','dismissal_kind','total_runs']]
bowling_death.insert(1,'balls',np.ones(bowling_death.shape[0]))
bowling_death=bowling_death.fillna(0)
#Consider only wickets taken by bowlers
bowling_death=bowling_death.replace(['caught','bowled','stumped','caught and bowled','lbw','hit wicket'],1)
bowling_death=bowling_death.replace(['run out','retired hurt','obstructing the field'],0)
bowling_grouped_death=bowling_death.groupby('bowler',as_index=False).sum()
bowling_grouped_death['economy']=6*bowling_grouped_death['total_runs']/bowling_grouped_death['balls']

mosteconomical_death=(bowling_grouped_death[bowling_grouped_death['balls']>400].sort_values(by=['economy'],ascending=False))
mostwickets_death=(bowling_grouped_death.sort_values(by=['dismissal_kind'],ascending=False))

sns.set_style('darkgrid')
ax = sns.barplot(x=mostwickets_death['bowler'].iloc[:10],y=mostwickets_death['dismissal_kind'].iloc[:10])
ax.set(xlabel='Bowler', ylabel='Wickets')
plt.title('Most wickets in the last 5 overs',fontdict={"fontsize":18})
plt.ylim([20,110])
plt.yticks(range(20,111,10))
plt.xticks(rotation=45)
plt.show()
sns.set_style('darkgrid')
ax = sns.barplot(x=mosteconomical_death['bowler'].iloc[:10],y=mosteconomical_death['economy'].iloc[:10])
ax.set(xlabel='Bowler', ylabel='Economy(Runs per over)')
plt.title('Most expensive in the last 5 overs(Atleast 300 balls)',fontdict={"fontsize":15})
plt.ylim([6,11])
# plt.yticks([6,12,1])
plt.xticks(rotation=45)
plt.show()
