# HOW MUCH MONEY SPENTING CHANGED OVER THE YEARS #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.0f}'.format   # changing format so no 'e+' appears in numbers.


df=pd.read_csv('../input/top-250-football-transfers-from-2000-to-2018/top250-00-19.csv')
mps=df.groupby('Season').agg({'Transfer_fee':np.mean}).reset_index()  #money per season
maxperseason=df.groupby('Season').agg({'Transfer_fee':np.max}).reset_index()
s=[]
d=[]
for a in df['Transfer_fee']:
    if a>40000000:
        s.append(1)
    else:
        s.append(0)
    if a>60000000:
        d.append(1)
    else:
        d.append(0)
over=pd.Series(s,index=df.index)
over6=pd.Series(d,index=df.index)
df['ver40']=over
df['ver60']=over6
o50=df.groupby('Season').agg({'ver40':np.sum,'ver60':np.sum}).reset_index()



ax1=mps.plot('Season','Transfer_fee',kind='bar',alpha=0.5,color='y',label='Average money per season',figsize=(12,6))
plt.clf()
plt.cla()
plt.close()

ax2=maxperseason.plot('Season','Transfer_fee',kind='bar',alpha=0.5,color='b',label='Most money for a player',figsize=(12,6))

for spine in plt.gca().spines.values():
    spine.set_visible(False)
xvals = range(len(mps['Season']))
bars=plt.bar(xvals,mps['Transfer_fee'],color='red',width=0.5,align='edge',log=True,label='Average money per season')

ax1=plt.gca()
ax1.set_xlabel('Season')
ax1.set_title('Max and average money spent per season')
ax1.set_ylabel('Money Spent')
x=plt.gca().xaxis
for item in x.get_ticklabels():
        item.set_rotation(60)
plt.legend()

plt.grid(True,axis='y',linestyle='--',linewidth=0.3,color='brown')
plt.show()
#In the above graph i can observe the difference in money spent each season along with the most money spent for a single player
#The first thing i can witness is that there is a steadily increase in the money spent during the years 2007-2018
pd.options.display.float_format = '{:.0f}'.format  # changing format so no 'e+' appears in numbers.
leaguefrom=df.groupby('League_from').agg({'Transfer_fee':np.sum}).reset_index()
leagueto=df.groupby('League_to').agg({'Transfer_fee':np.sum}).reset_index()
leaguefrom=leaguefrom.rename(columns={'Transfer_fee':'Profits Per League','League_from':'League'})
leagueto=leagueto.rename(columns={'Transfer_fee':'Expenses Per League','League_to':'League'})
leagueto=leagueto.set_index('League')
leaguefrom=leaguefrom.set_index('League')
merge=pd.merge(leagueto,leaguefrom,how='outer',left_index=True,right_index=True)
s=[]

merge=merge.fillna(0)
for a in merge.index:
    dif=merge.loc[a]['Profits Per League']-merge.loc[a]['Expenses Per League']
    s.append(dif)
s=pd.Series(s,index=merge.index)
merge['Total Diff']=s


merge=merge.sort_values('Expenses Per League',ascending=False)
top20spenters=merge[:19]
merge=merge.sort_values('Profits Per League',ascending=False)
top20takers=merge[:19]
top20takers
merge=pd.merge(top20takers,top20spenters,how='inner',right_index=True,left_index=True).T.drop_duplicates().T.reset_index() #sbinei tis 2ples stiles
merge=merge.sort_values('Expenses Per League_x',ascending=False)
print(merge)

###Above i can notice that Premier league teams have spent the most money.So i decide to exam Premier league football for the 
###years 2007 and later
aa=df[df['League_to']=='Premier League']
ss=df[df['League_from']=='Premier League']
ss=ss.rename(columns={'Transfer_fee':'Money gained'})
aa=aa.rename(columns={'Transfer_fee':'Money spent'})

###DATAFRAME,PLAYERS TEAM BOUGHT
ss=ss.sort_values('Team_from')
ss=ss.set_index(['Team_from','Team_to'])
del (ss['League_from'],ss['ver40'],ss['ver60']) 

###DATAFRAME,PLAYERS TEAM SOLD
aa=aa.sort_values('Team_to')
aa=aa.set_index(['Team_to','Team_from'])
del (aa['League_to'],aa['ver40'],aa['ver60'])

###DATABASE RESULTS
df1=pd.read_csv('../input/epl-results-19932018/EPL_Set.csv'
)
df1['Season']=df1['Season'].map(lambda x: str(x)[:-3]) # Σβηνει τη δευτερη σεζον
df1['Season']=df1['Season'].astype(int)
df1=df1[df1['Season']>2007]  



w=[]
d=[]
l=[]
for a in df1['FTR']:
    if a=='H':
        w.append(1)
        d.append(0)
        l.append(0)
    elif a=='D':
        w.append(0)
        d.append(1)
        l.append(0)
    else:
        w.append(0)
        d.append(0)
        l.append(1)
        
wh=pd.Series(w,index=df1['HomeTeam'])
dh=pd.Series(d,index=df1['HomeTeam'])
lh=pd.Series(l,index=df1['HomeTeam'])
wa=pd.Series(l,index=df1['AwayTeam'])
da=pd.Series(d,index=df1['AwayTeam'])
la=pd.Series(w,index=df1['AwayTeam'])

df1=df1.set_index('HomeTeam')
df1['WinsHome']=wh
df1['LosesHome']=lh
df1['DrawsHome']=dh
df1=df1.reset_index()


df1=df1.set_index('AwayTeam')
df1['WinsAway']=wa
df1['DrawsAway']=da
df1['LosesAway']=la
df1=df1.reset_index()


a=df1.groupby(['HomeTeam','Season']).agg({'WinsHome':np.sum,'DrawsHome':np.sum,'LosesHome':np.sum}).reset_index()
b=df1.groupby(['AwayTeam','Season']).agg({'WinsAway':np.sum,'DrawsAway':np.sum,'LosesAway':np.sum}).reset_index()
a=a.rename(columns={'HomeTeam':'Team'})
b=b.rename(columns={'AwayTeam':'Team'})
a=a.set_index(['Team','Season'])
b=b.set_index(['Team','Season'])

#EVERY PREMIER LEAGUE TEAM AND HOW MUCH MONEY THEY HAVE SPENT DURING THIS DECATE,dataframe name:ab
ab=pd.merge(a,b,how='outer',left_index=True,right_index=True).reset_index()
ab['WinsTotal']=pd.Series(ab['WinsHome']+ab['WinsAway']).astype(float)
ab['LosesTotal']=pd.Series(ab['LosesHome']+ab['LosesAway']).astype(float)
ab['DrawsTotal']=pd.Series(ab['DrawsHome']+ab['DrawsAway']).astype(float)
ab['Points']=pd.Series(ab['WinsTotal']*3+ab['DrawsTotal']).astype(float)

##Money income and outcome during these years
aa['Season']=aa['Season'].map(lambda x: str(x)[:-5])
aa['Season']=aa['Season'].astype(int)
aa=aa[(aa['Season']>2007)&aa['Season']<2018]
ss['Season']=ss['Season'].map(lambda x:str(x)[:-5])
ss['Season']=ss['Season'].astype(int)
ss=ss[(ss['Season']>2007)&(ss['Season']<2018)]
aa=aa.reset_index()
ss=ss.reset_index()
aa=aa.rename(columns={'Name':'Player acquired','Team_to':'Team'})
ss=ss.rename(columns={'Name':'Player sold','Team_from':'Team'})
aa=aa.groupby(['Team','Season']).agg({'Money spent':np.sum}).reset_index()
ss=ss.groupby(['Team','Season']).agg({'Money gained':np.sum}).reset_index()
aa=aa.set_index(['Team','Season'])
ss=ss.set_index(['Team','Season'])
ba=pd.merge(aa,ss,left_index=True,right_index=True,how='inner').reset_index()

columns=['Team','Season','WinsTotal','LosesTotal','DrawsTotal','Points']
ab=ab[columns]






ba['SeasonEstimate']=pd.Series((ba['Money gained']-ba['Money spent']))

s=[]
for a in ab['Team']:
    if a=='Tottenham':
        a='Spurs'
    if a=='Hull':
        a='Hull City'
    if a=='Man United':
        a='Man Utd'
    if a=='Stoke':
        a='Stoke City'
    s.append(a)
ab['Team']=pd.Series(s)
ab['Team']

##Merging the two dataframes to compare money spent and gained with their win rate
ba=ba.set_index(['Team','Season'])
ab=ab.reset_index()
ab=ab.set_index(['Team','Season'])
aba=pd.merge(ba,ab,left_index=True,right_index=True).reset_index()
del aba['index']
aba['WinsTotal']=aba['WinsTotal'].astype(float)
aba['LosesTotal']=aba['LosesTotal'].astype(float)
aba['DrawsTotal']=aba['DrawsTotal'].astype(float)
d=[]
for a in aba['WinsTotal']:
    s=a/38
    s=s*100
    d.append(s)
w=[]
for a in aba['LosesTotal']:
    s=a/38
    s=s*100
    w.append(s)
aba['LoseRate%']=pd.Series(w)
aba['WinRate%']=pd.Series(d)
aba=aba.fillna(0)
a=aba[aba['WinRate%']>50]
a=a[a['SeasonEstimate']>0].sort_values('SeasonEstimate')
print(aba)
print(a)
#the first matrix has all the premier league clubs since 2007,their income and 
#outcome and their win rate.

#The second matrix has the most successful teams with winrate over 50% and which 
#gained more money than spent during the end of the transfer period
##Importing an new dataframe with all the results of the Premier League clubs 


points=pd.read_csv('../input/premierleague-league-tables-188889-201617/result.csv')
points['year']=points['year'].map(lambda x: str(x)[:-5]).astype(int)
relegation=points[points['year']>1999].groupby('Pos').agg({'Pts':np.mean})
Winrate=aba.groupby('Points').agg({'WinRate%':np.mean}).sort_values('WinRate%',ascending=False)
print(relegation)
print(Winrate)


###THE MEAN VALUES FOR CHAMPION,RELEGATION,AND QUALIFICATION FOR EUROPEAN FOOTBAL
###RELEGATION:BELOW 38 AND A WIN RATE OF 22 AND BELOW
###CHAMPION:OVER 81 POINTS AND A WIN RATE OVER 62%
###QUALIFICATION FOR CHAMPIONS LEAGUE OR EUROPE LEAGUE:OVER 63 POINTS AND OVER 45%
###MIDDLE VALUE OF THE TABLE:48 POINTS AND A WIN RATE OF 32%
#Comparing the Total Points with the Money Spent.
plt.figure(figsize=(20,10))
ax=plt.subplots_adjust(hspace=0,wspace=0)
plt.legend()

###ΓΙΑ ΝΑ ΒΑΛΩ ΑΞΟΝΕΣ ΣΕ ΚΑΘΕ ΠΛΟΤ ΕΧΩ ΕΛΕΓΞΧΕΙ ΤΑ SEASON ESTIMATE ΑΝΑΛΟΓΑ ΜΕ ΤΟΝ WINRATE ΓΙΑ ΝΑ ΠΡΟΣΑΡΜΟΣΩ ΤΟΥΣ ΑΞΟΝΕΣ

a2=plt.subplot(2,2,3)
axx2=aba[(aba['Points']<48)&(aba['Money spent']<40000000)]
ax2=plt.scatter(axx2['Points'],axx2['Money spent'],marker='x',c='r',alpha=0.4,s=150,label='Below middle ,didnt spent a lot')
plt.xticks(np.arange(0,48,5))
plt.yticks(np.arange(0,40000000,10000000))
plt.ylim(0,400000000)
plt.gca().spines['right'].set_visible(False)
plt.xlabel('Points below average')
plt.ylabel('Spent less Than 40.000.000')
plt.axvline(x=37,linestyle='-',label='Relegation',color='r')
plt.legend()

a4=plt.subplot(2,2,4,sharey=a2)
axx4=aba[(aba['Points']>48)&(aba['Money spent']<40000000)]
ax4=plt.scatter(axx4['Points'],axx4['Money spent'],marker='o',alpha=0.4,c='y',s=150,label='Over the Middle,didnt Spent a Lot=>SUCCESS')
plt.gca().axes.get_yaxis().set_visible(False)
plt.xticks(np.arange(48,100,5))
plt.yticks(np.arange(0,40000000,10000000))
plt.ylim(0,40000000)
plt.xlabel('Points above average')
plt.axvline(x=63,linestyle='-',label='Europe')
plt.axvline(x=80,linestyle='-',label='Champ',color='green')
plt.legend()

a1=plt.subplot(2,2,1,sharex=a2)
plt.xticks(np.arange(0,48,5))
plt.yticks(np.arange(40000000,320000000,20000000))
axx1=aba[(aba['Points']<48)&(aba['Money spent']>40000000)]
ax1=plt.scatter(axx1['Points'],axx1['Money spent'],marker='x',c='orange',alpha=0.4,s=150,label='Below the Middle,spent Big=>FAILURE')
plt.gca().axes.get_xaxis().set_visible(False)
plt.ylim(40000000,320000000)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.ylabel('Spent Over 40.000.000')
plt.axvline(x=37,linestyle='-',label='Relegation',color='r')
plt.legend()


plt.subplot(2,2,2,sharey=a1,sharex=a4)
axx3=aba[(aba['Points']>48)&(aba['Money spent']>40000000)]
ax3=plt.scatter(axx3['Points'],axx3['Money spent'],marker='o',alpha=0.4,c='green',s=150,label='Over the Middle,Spent a Lot')
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.xticks(np.arange(48,100,5))
plt.yticks(np.arange(40000000,320000000,20000000))
plt.ylim(40000000,320000000)
plt.gca().spines['bottom'].set_visible(False)
plt.ylabel('aaa')
plt.axvline(x=63,linestyle='-',label='Europe')
plt.axvline(x=80,linestyle='-',label='Champ',color='green')
plt.legend()


plt.show()
#By Observing the graph comparing Points to money spent i can notice first of all that to only two teams managed to win 
#by spending less than 40.000.000€ and they got less points than all the other champs who spent more than 40.000.000€,so maybe
#the years these teams were champions,the level of the other teams wasnt so high.

#The most teams below the middle of the table spent less than 40.000.000€
#And the most teams that relegated spent less than 20.000.000€
#Spending over 40.000.000 might avoid relegation(there were just 3 exceptions)

#All the teams which spent more than 100.000.000€ finished over the middle of the table and the most qualified for European 
#Football. However,spending over 100.000.000 doesnt guarantee success because 5 teams didnt qualify for European Football. 


#Comparing Season Estimates with the Win Rate of every team
plt.figure(figsize=(20,10))
ax=plt.subplots_adjust(hspace=0,wspace=0)
plt.legend()

###ΓΙΑ ΝΑ ΒΑΛΩ ΑΞΟΝΕΣ ΣΕ ΚΑΘΕ ΠΛΟΤ ΕΧΩ ΕΛΕΓΞΧΕΙ ΤΑ SEASON ESTIMATE ΑΝΑΛΟΓΑ ΜΕ ΤΟΝ WINRATE ΓΙΑ ΝΑ ΠΡΟΣΑΡΜΟΣΩ ΤΟΥΣ ΑΞΟΝΕΣ

a2=plt.subplot(2,2,3)
axx2=aba[(aba['WinRate%']<32)&(aba['SeasonEstimate']<0)]
ax2=plt.scatter(axx2['WinRate%'],axx2['SeasonEstimate'],marker='x',c='r',alpha=0.4,s=150,label='Below the middle WinRate,Lost Money => FAILURE')
plt.xticks(np.arange(0,32,5))
plt.yticks(np.arange(-236400000,0,15000000))
plt.ylim(-236400000,0)
plt.gca().spines['right'].set_visible(False)
plt.xlabel('WinRate below average')
plt.ylabel('Negative Season Estimate')
plt.axvline(x=22,linestyle='-',label='Relegation',color='r')
plt.legend()

a4=plt.subplot(2,2,4,sharey=a2)
axx4=aba[(aba['WinRate%']>32)&(aba['SeasonEstimate']<0)]
ax4=plt.scatter(axx4['WinRate%'],axx4['SeasonEstimate'],marker='o',alpha=0.4,c='y',s=150,label='Over the middle WinRate,Lost Money')
plt.gca().axes.get_yaxis().set_visible(False)
plt.xticks(np.arange(32,80,5))
plt.yticks(np.arange(-236400000,0,15000000))
plt.ylim(-236400000,0)
plt.xlabel('WinRate above average')
plt.axvline(x=45,linestyle='-',label='Europe')
plt.axvline(x=63,linestyle='-',label='Champ',color='green')
plt.legend()

a1=plt.subplot(2,2,1,sharex=a2)
plt.xticks(np.arange(0,32,5))
plt.yticks(np.arange(0,80000000,5000000))
axx1=aba[(aba['WinRate%']<32)&(aba['SeasonEstimate']>0)]
ax1=plt.scatter(axx1['WinRate%'],axx1['SeasonEstimate'],marker='x',c='orange',alpha=0.4,s=150,label='Below the middle WinRate,Won Money => OK')
plt.gca().axes.get_xaxis().set_visible(False)
plt.ylim(0,80000000)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.ylabel('Positive Season Estimate')
plt.axvline(x=22,linestyle='-',label='Relegation',color='r')
plt.legend()


plt.subplot(2,2,2,sharey=a1,sharex=a4)
axx3=aba[(aba['WinRate%']>32)&(aba['SeasonEstimate']>0)]
ax3=plt.scatter(axx3['WinRate%'],axx3['SeasonEstimate'],marker='o',alpha=0.4,c='green',s=150,label='Over the middle WinRate,Got Money => BIG SUCCESS')
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.xticks(np.arange(32,80,5))
plt.yticks(np.arange(0,80800000,5000000))
plt.ylim(0,80000000)
plt.gca().spines['bottom'].set_visible(False)
plt.ylabel('aaa')
plt.axvline(x=45,linestyle='-',label='Europe')
plt.axvline(x=63,linestyle='-',label='Champ',color='green')
plt.legend()


plt.show()
###Having a +/- of over 100.000.000 means over 42% WinRate and that thats qualification for European Football



###Most of the teams that spent more money than gained through transfers avoided relegation.
###Thats almost the same observation with the above graph,that means,to surive in the Premier League you have to SPENT.
###I WILL TRY TO FIGURE HOW MUCH MONEY YOU HAVE TO SPENT IN ORDER T GET A 63% AND WIN THE LEAGUE.
###BY VIEWING AT THE GRAPHS I HAVE NOTICED A TEAM SPENT OVER 300.000.000 TO WIN SO I WILL REMOVE IT
a=aba[aba['WinRate%']>63].drop(labels=66)
print(np.mean(a['Money spent']))

##So you will need to spent at least 84.000.000€
##Lets see how many teams that spent over 84.000.000 won the league.
c=aba[aba['Money spent']>82080000]
d=aba[(aba['Money spent']>82080000)&(aba['Points']>81)]
print(len(d)/len(c))
###20% of the teams did it.
###So money is an important factor in avoiding relegation but doesnt guarantee you will win the league.
###However,it will certainly improve your chances.