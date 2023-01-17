import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
pd.options.mode.chained_assignment = None
df=pd.read_csv('../input/Seasons_Stats.csv')
df.columns
del df['Unnamed: 0']

###REMOVING DUPICATE PLAYERS,PLAYERS WHO GOT TRADED IN A SEASON ARE BEING MENTIONED 3 TIMES,ILL KEEP THEIR TOTAL STATS.
def remove_duplicate_players(df):
    player_occurrences = {}
    for i in range(len(df)):
        player_name = data_frame.iloc[i]['Player']
        player_team = data_frame.iloc[i]['Tm']
        index_row = data_frame.index[i]
        if player_name not in player_occurrences:
            player_occurrences[player_name] = []
        player_occurrences[player_name].append((player_team, index_row))

    for key in player_occurrences:
        curr_list = player_occurrences[key]
        if len(curr_list) == 1:
            continue
        for team, index in curr_list:
            if team != "TOT":
                df = df.drop(index)
    return df

### NOW ILL PUT AN EXTRA COLUMN FOR GUARDS FORWARDS AND CENTERS
def position(x):
    if x=='SG':
        return ('Guard')
    if x=='PG':
        return('Guard')
    if (x=='SF')| (x=='PF'):
        return('Forward')
    else:
        return('Center')
df['Position']=df['Pos'].apply(position)
### ILL ADD PPG,APG,RPG,FTPG,TOPG



df['Year']=df['Year'].map(lambda x:str(x)[:-2])
df=df.set_index('Player').sort_values('PTS',ascending=False)
df.loc['Michael Jordan*'].sort_values('Year')
df=df[df['Year']!='n']
df['Year']=df['Year'].astype(int)
##SO WE SEE THAT MIKE WAS PLAYING DURING 1985-1998 SO FIRSTLY ILL COMPARE HIS ERA WITH THE 2015-2018 ERA.

###FIRST OF ALL ILL GROUPBY THE 2 ERAS.THE 1985-1998 AND THE 2015-2018
mjera=df[(df['Year']>1984)&(df['Year']<1999)]
newera=df[(df['Year']>2013)&(df['Year']<2018)]

###WITH HOLYNGERS PER GUIDE FOR PLAYERS WHO PLAYED OVER 15 MATCHES ILL GET THE BEST ONES WITH PER>20
mjera['PPG']=mjera['PTS']/mjera['G']
mjera['PPG']=round(mjera['PPG'],1)
mjera['APG']=mjera['AST']/mjera['G']
mjera['RPG']=mjera['TRB']/mjera['G']
newera['PPG']=newera['PTS']/newera['G']
newera['PPG']=round(newera['PPG'],1)
newera['APG']=newera['AST']/newera['G']
newera['RPG']=newera['TRB']/newera['G']
newera['MPG']=newera['MP']/newera['G']
mjera['MPG']=mjera['MP']/mjera['G']
mjera['BPG']=mjera['BLK']/mjera['G']
newera['BPG']=newera['BLK']/newera['G']
mjera['SPG']=mjera['STL']/mjera['G']
newera['SPG']=newera['STL']/newera['G']
mjera['PFPG']=mjera['PF']/mjera['G']
newera['PFPG']=newera['PF']/newera['G']
mjera['TPG']=mjera['TOV']/mjera['G']
newera['TPG']=newera['TOV']/newera['G']

### SORTING WITH PER over 15,so they are rotation players

mjbest=mjera[(mjera['G']>15)&(mjera['PER']>15)&(mjera['PPG']>12)].sort_values('PER',ascending=False)
newbest=newera[(newera['G']>15)&(newera['PER']>15)&(newera['PPG']>12)].sort_values('PER',ascending=False)
### ILL CHECK MINUTES PER GAME HERE FOR PLAYERS WITH PER OVER 20(ALLSTAR CALIBER PLAYERS)
mjper20=mjbest[mjbest['PER']>20]
newper20=newbest[newbest['PER']>20]
print(np.mean(mjper20['MPG']))
print(np.mean(newper20['MPG']))
### WE SEE THAT BEST PLAYERS IN MJ ERA PLAYED 3 MORE MINS. 
###MAYBE THIS IS BECAUSE NOWDAYS PLAYERS ARE MORE FOCUSED FOR PLAYOFFS
mjper20=mjper20[:138] ###I WILL USE THE SAME AMOUNT OF PLAYERS FOR BOTH ERAS
mjper20
mjper20=mjper20.reset_index()
newper20=newper20.reset_index()
columns=['Player','MP', 'PER', 'TS%',
       '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%',
       'USG%',  'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 'DBPM',
       'BPM', 'VORP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA',
       '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL',
       'BLK', 'TOV', 'PF', 'PTS', 'PPG', 'APG', 'RPG', 'MPG','SPG','BPG','PFPG','TPG' ,'Position']
##ILL TRY TO FIGURE SOME CHANGES ON EACH POSITION
###BASIC STATS DIFFERENCES(OFFENSE)


aa=mjper20.groupby('Position').mean().reset_index().sort_values('PPG')
col=['PTS', 'PPG','TPG', 'APG','FGA','Position']
aa=aa[col].set_index('Position')
newper20=newper20[columns]
bb=newper20.groupby('Position').mean().reset_index().sort_values('PPG')
bb=bb[col].set_index('Position')
ab=pd.merge(aa,bb,how='outer',left_index=True,right_index=True)##>>>>>>>>basic stats.
ab=ab.reset_index()

na=range(len(ab['Position']))
plt.figure(figsize=(18,11))

plt.subplot(2,3,1)
ax1=plt.bar(na,ab['PPG_x'],width=0.5,label='MJera')
ax2=plt.bar(na,ab['PPG_y'],label='NewEra',width=0.5,align='edge')
plt.xticks(na,ab['Position'])
plt.yticks(np.arange(1,30,2))
plt.xlabel('Position',color='red')
plt.ylabel('Points Per Game',color='red')
plt.grid(True,axis='y',linestyle='--',linewidth=0.3,color='brown')
plt.legend(loc=4)

plt.subplot(2,3,2)
ax3=plt.bar(na,ab['APG_x'],label='MJera',width=0.5)
ax4=plt.bar(na,ab['APG_y'],label='NewEra',width=0.5,align='edge')
plt.xticks(na,ab['Position'])
plt.yticks(np.arange(1,10,1))
plt.xlabel('Position',color='red')
plt.ylabel('Assists Per Game',color='red')
plt.grid(True,axis='y',linestyle='--',linewidth=0.3,color='brown')
plt.legend(loc=4)

plt.subplot(2,3,3)
ax4=plt.bar(na,ab['FGA_x'],label='MJera',width=0.5)
ax5=plt.bar(na,ab['FGA_y'],label='NewEra',width=0.5,align='edge')
plt.xticks(na,ab['Position'])
plt.xlabel('Position',color='red')
plt.ylabel('Field Goal Attemps Per Game',color='red')
plt.grid(True,axis='y',linestyle='--',linewidth=0.3,color='brown')
plt.legend(loc=4)

plt.subplot(2,3,4)
ax4=plt.bar(na,ab['TPG_x'],label='MJera',width=0.5)
ax5=plt.bar(na,ab['TPG_y'],label='NewEra',width=0.5,align='edge')
plt.xticks(na,ab['Position'])
plt.yticks(np.arange(0,3,0.3))
plt.xlabel('Position',color='red')
plt.ylabel('Turnovers Per Game',color='red')
plt.grid(True,axis='y',linestyle='--',linewidth=0.3,color='brown')
plt.legend(loc=4)

plt.show()
###SOME CONCLUSIONS
# 1) DURING MJ'S ERA CENTERS AND FORWARDS HAD THE MOST POINTS(AND FIELD GOALS) BUT THAT CHANGES IN THE NEW ERA
#    WHERE GUARDS SCORE THE MOST.
# 2) DURING MJ'S ERA EVERY POSITION HAD MOST ASSIST(EXCEPT FORWARDS),AND ESPECIALY GUARDS.THAT CHANGES TOO,MAYBE BECAUSE 
#    PLAYERS ARE CREATING MORE FOR THEIRSELFS
# 3) NOWDAYS WE HAVE LESS TURNOVERS. WORSE DEFENSE OR MORE CAREFUL OFFENSE??
###BASIC STATS(DEFENSE)
aa=mjper20.groupby('Position').mean().reset_index().sort_values('PPG')
col=['RPG', 'BPG', 'SPG','PFPG','Position']
aa=aa[col].set_index('Position')
newper20=newper20[columns]
bb=newper20.groupby('Position').mean().reset_index().sort_values('PPG')
bb=bb[col].set_index('Position')
ab=pd.merge(aa,bb,how='outer',left_index=True,right_index=True)##>>>>>>>>basic stats.
ab=ab.reset_index()

na=range(len(ab['Position']))
plt.figure(figsize=(18,11))

plt.subplot(2,3,1)
ax1=plt.bar(na,ab['RPG_x'],width=0.5,label='MJera')
ax2=plt.bar(na,ab['RPG_y'],label='NewEra',width=0.5,align='edge')
plt.xticks(na,ab['Position'])
plt.yticks(np.arange(0,15,1))
plt.xlabel('Position',color='red')
plt.ylabel('Rebound Per Game',color='red')
plt.grid(True,axis='y',linestyle='--',linewidth=0.3,color='brown')
plt.legend(loc=1)

plt.subplot(2,3,2)
ax3=plt.bar(na,ab['BPG_x'],label='MJera',width=0.5)
ax4=plt.bar(na,ab['BPG_y'],label='NewEra',width=0.5,align='edge')
plt.xticks(na,ab['Position'])
plt.yticks(np.arange(0,3,0.2))
plt.xlabel('Position',color='red')
plt.ylabel('Blocks Per Game',color='red')
plt.grid(True,axis='y',linestyle='--',linewidth=0.3,color='brown')
plt.legend(loc=1)

plt.subplot(2,3,3)
ax4=plt.bar(na,ab['SPG_x'],label='MJera',width=0.5)
ax5=plt.bar(na,ab['SPG_y'],label='NewEra',width=0.5,align='edge')
plt.xticks(na,ab['Position'])
plt.yticks(np.arange(0,2,0.2))
plt.xlabel('Position',color='red')
plt.ylabel('Steals Per Game',color='red')
plt.grid(True,axis='y',linestyle='--',linewidth=0.3,color='brown')
plt.legend(loc=2)

plt.subplot(2,3,4)
ax4=plt.bar(na,ab['PFPG_x'],label='MJera',width=0.5)
ax5=plt.bar(na,ab['PFPG_y'],label='NewEra',width=0.5,align='edge')
plt.xticks(na,ab['Position'])
plt.yticks(np.arange(0,5,0.4))
plt.xlabel('Position',color='red')
plt.ylabel('Personal Fouls Per Game',color='red')
plt.grid(True,axis='y',linestyle='--',linewidth=0.3,color='brown')
plt.legend(loc=1)

plt.show()
###CONLUSIONS:
##1) THE GAME SEEMS HARDER BACK IN MJ'S ERA
##2) PLAYERS IN EACH POSITION WERE COMMITING MORE FOULS
##3) MORE BLOCKS AND MORE STEALS FOR EVERY POSITION BACK IN MJ'S ERA
###ADVANCE STATS DIFFERENCES(OFFENSE)
aa=mjper20.groupby('Position').mean().reset_index().sort_values('PPG')
col=['USG%','3PAr', 'AST%','TOV%','PER', 'Position']
aa=aa[col].set_index('Position')
newper20=newper20[columns]
bb=newper20.groupby('Position').mean().reset_index().sort_values('PPG')
bb=bb[col].set_index('Position')
ba=pd.merge(aa,bb,how='outer',left_index=True,right_index=True)##>>>>>>>>ADVANCE
ba=ba.reset_index()

na=range(len(ba['Position']))
plt.figure(figsize=(18,11))

plt.subplot(2,3,1)
ax1=plt.bar(na,ba['USG%_x'],width=0.5,label='MJera')
ax2=plt.bar(na,ba['USG%_y'],label='NewEra',width=0.5,align='edge')
plt.xticks(na,ba['Position'])
plt.yticks(np.arange(0,30,2))
plt.xlabel('Position',color='red')
plt.ylabel('Usage Rate %',color='red')
plt.grid(True,axis='y',linestyle='--',linewidth=0.3,color='brown')
plt.legend(loc=4)

plt.subplot(2,3,2)
ax3=plt.bar(na,ba['3PAr_x'],label='MJera',width=0.5)
ax4=plt.bar(na,ba['3PAr_y'],label='NewEra',width=0.5,align='edge')
plt.xticks(na,ba['Position'])
plt.yticks(np.arange(0,0.5,0.03))
plt.xlabel('Position',color='red')
plt.ylabel('Percentage Of Field Goals That is a 3',color='red')
plt.grid(True,axis='y',linestyle='--',linewidth=0.3,color='brown')
plt.legend(loc=4)

plt.subplot(2,3,3)
ax4=plt.bar(na,ba['TOV%_x'],label='MJera',width=0.5)
ax5=plt.bar(na,ba['TOV%_y'],label='NewEra',width=0.5,align='edge')
plt.xticks(na,ba['Position'])
plt.yticks(np.arange(0,15,1))
plt.xlabel('Position',color='red')
plt.ylabel('Turnover Per 100 Possesions ',color='red')
plt.grid(True,axis='y',linestyle='--',linewidth=0.3,color='brown')
plt.legend(loc=4)

plt.subplot(2,3,4)
ax4=plt.bar(na,ba['AST%_x'],label='MJera',width=0.5)
ax5=plt.bar(na,ba['AST%_y'],label='NewEra',width=0.5,align='edge')
plt.xticks(na,ba['Position'])
plt.yticks(np.arange(0,35,2))
plt.xlabel('Position',color='red')
plt.ylabel('Fields Goals Assisted While Player on Game',color='red')
plt.grid(True,axis='y',linestyle='--',linewidth=0.3,color='brown')
plt.legend(loc=4)

plt.subplot(2,3,5)
ax4=plt.bar(na,ba['PER_x'],label='MJera',width=0.5)
ax5=plt.bar(na,ba['PER_y'],label='NewEra',width=0.5,align='edge')
plt.xticks(na,ba['Position'])
plt.yticks(np.arange(0,30,2))
plt.xlabel('Position',color='red')
plt.ylabel('Players Efficiency Rating',color='red')
plt.grid(True,axis='y',linestyle='--',linewidth=0.3,color='brown')
plt.legend(loc=4)

plt.show()
# 1) USAGE RATE OF THE BIGS HAS GREATLY DECREASED WHILE THE USAGE RATE OF THE GUARDS IS INCREASED
#    IT WAS BIGS ERA AND NOW ITS A GUARDS ERA.
#2) A LOT OF MORE THREES 
#3) LESS MISTAKES
#4) WE SEE THAT FORWARDS TEND TO ASSIST MORE WHILE GUARDS ASSIST LESS.THE GAME IS SEEMS MORE POSITIONLESS

aa=mjper20.groupby('Position').mean().reset_index().sort_values('PPG')
col=['ORB%','DRB%' ,'STL%', 'BLK%','Position','BPM']
aa=aa[col].set_index('Position')
newper20=newper20[columns]
bb=newper20.groupby('Position').mean().reset_index().sort_values('PPG')
bb=bb[col].set_index('Position')
ba=pd.merge(aa,bb,how='outer',left_index=True,right_index=True)##>>>>>>>>ADVANCE
ba=ba.reset_index()

na=range(len(ba['Position']))
plt.figure(figsize=(18,11))

plt.subplot(2,3,1)
ax1=plt.bar(na,ba['ORB%_x'],width=0.5,label='MJera')
ax2=plt.bar(na,ba['ORB%_y'],label='NewEra',width=0.5,align='edge')
plt.xticks(na,ba['Position'])
plt.yticks(np.arange(0,15,2))
plt.xlabel('Position',color='red')
plt.ylabel('Offensive Rebounds available Grabbed',color='red')
plt.grid(True,axis='y',linestyle='--',linewidth=0.3,color='brown')
plt.legend(loc=4)

plt.subplot(2,3,2)
ax3=plt.bar(na,ba['DRB%_x'],label='MJera',width=0.5)
ax4=plt.bar(na,ba['DRB%_y'],label='NewEra',width=0.5,align='edge')
plt.xticks(na,ba['Position'])
plt.yticks(np.arange(0,30,3))
plt.xlabel('Position',color='red')
plt.ylabel('Defensive Rebounds available Grabbed',color='red')
plt.grid(True,axis='y',linestyle='--',linewidth=0.3,color='brown')
plt.legend(loc=4)

plt.subplot(2,3,3)
ax4=plt.bar(na,ba['STL%_x'],label='MJera',width=0.5)
ax5=plt.bar(na,ba['STL%_y'],label='NewEra',width=0.5,align='edge')
plt.xticks(na,ba['Position'])
plt.yticks(np.arange(0,3,0.2))
plt.xlabel('Position',color='red')
plt.ylabel('Opponent Possesions Ended With Steal % ',color='red')
plt.grid(True,axis='y',linestyle='--',linewidth=0.3,color='brown')
plt.legend(loc=4)

plt.subplot(2,3,4)
ax4=plt.bar(na,ba['BLK%_x'],label='MJera',width=0.5)
ax5=plt.bar(na,ba['BLK%_y'],label='NewEra',width=0.5,align='edge')
plt.xticks(na,ba['Position'])
plt.yticks(np.arange(0,5,0.5))
plt.xlabel('Position',color='red')
plt.ylabel('Opponent Possesions Ended With Block % ',color='red')
plt.grid(True,axis='y',linestyle='--',linewidth=0.3,color='brown')
plt.legend(loc=4)



plt.show()
#1) WHILE CENTERS IN THE NEW ERA HAVE LESS USG% THEY GRAB MORE EBOUNDS,SO THEY CONTRIBUTE LESS IN SCORING BUT MORE IN REBOUNDS
#2) IN MJ'S ERA GUARDS AND FORWARDS WERE GRABING MORE REBOUNDS THAN NOWDAYS,MAYBE BECAUSE CENTERS HAD THE BALL MORE??
#3) THATS NOT HAPPENING IN DEFENSIVE REBOUNDS WERE GUARDS TEND TO CONTRIBUTE IN REBOUNDING MORE.
#4) FAR MORE STEALS IN MJ'S ERA.TOUGHER DEFENSES??FOULING SYSTEM HAS CHANGED??
#5) IN MJ ERAS CENTERS WERE BLOCKING A LOT MORE THAN TODAY.HOWEVER,FORWARDS BLOCK FAR MORE IN THIS ERA

###Lets See the How much was mj shooting and his usage rate during his best years.1987-1993(according to how much he won)
mjera=mjera.reset_index()
mj=mjera[(mjera['Player']=='Michael Jordan*')&(mjera['Year']>1986)&(mjera['Year']<1999)]
mj['3PG']=mj['3PA']/mj['G']
mj['2PG']=mj['2PA']/mj['G']
mj['FGAPG']=mj['FGA']/mj['G']
mj=mj.groupby('Player').agg({'USG%':np.mean,'3PG':np.mean,'2PG':np.mean,'FGAPG':np.mean})


#Thats obvious cuz we had a lot of players and not all allstars.Show i have to narrow down the modern era and get more USG%
newbg=newera[(newera['Position']=='Guard')|(newera['Position']=='Forward')]
newbg=newbg[(newbg['USG%']>30)&(newbg['G']>50)]
newbg['3PG']=newbg['3PA']/newbg['G']
newbg['2PG']=newbg['2PA']/newbg['G']
newbg['FGAPG']=newbg['FGA']/newbg['G']
newbg=newbg.reset_index()
newbg=newbg.groupby('Year').agg({'USG%':np.mean,'FGA':np.mean,'3PG':np.mean,'2PG':np.mean,'FGAPG':np.mean})

newbg['Year']='Modern Era'
newbg=newbg.groupby('Year').agg({'USG%':np.mean,'3PG':np.mean,'2PG':np.mean,'FGAPG':np.mean})
mj=mj.reset_index()
mj=mj.rename(columns={'Player':'Year'})
newbg=newbg.reset_index()
a=pd.merge(mj,newbg,how='outer')


###SO MJ WAS SHOOTING A LOT CUZ OBVIOUSLY HE WAS THE BEST PLAYER IN THE WORLD
### BUT LETS THE REST OF THE ALLSTARS IN MJ ERA,TO COMPARE THE DIFFERENCE.
mjbg=mjera[(mjera['Position']=='Guard')|(mjera['Position']=='Forward')]
mjbg=mjbg[(mjbg['USG%']>30)&(mjbg['G']>50)]
mjbg['3PG']=mjbg['3PA']/mjbg['G']
mjbg['2PG']=mjbg['2PA']/mjbg['G']
mjbg['FGAPG']=mjbg['FGA']/mjbg['G']

mjbg=mjbg.groupby('Year').agg({'USG%':np.mean,'FGA':np.mean,'3PG':np.mean,'2PG':np.mean,'FGAPG':np.mean})
mjbg=mjbg.reset_index()
mjbg['Year']='Mj era'
mjbg=mjbg.groupby('Year').agg({'USG%':np.mean,'3PG':np.mean,'2PG':np.mean,'FGAPG':np.mean})
mjbg=mjbg.reset_index()
b=pd.merge(a,mjbg,how='outer')
b['% that is a three']=round(b['3PG']/b['FGAPG'],2)
b['% that is a two']=round(b['2PG']/b['FGAPG'],2)
print(b)
#SO MJ HAD HIGHER USG% THAN THE OTHER ALLSTARS,MAYBE BECAUSE HE WAS THE BEST IN THE WORLD.
#WHAT IS INTRIGUING IS THE HUGE DIFFERENCE IN THE % OF THREES AND TWOS.
a
## OWS,DWS FOR GUARDS,BECAUSE MJ IS A GUARD AND NOWADAYS IT IS A FACT THAT WE HAVE A GUARDS ERA.
import matplotlib.gridspec as gridspec

mjguards=mjper20[mjper20['Position']=='Guard']
mjforwards=mjper20[mjper20['Position']=='Forward']
mjcenters=mjper20[mjper20['Position']=='Center']
newguards=newper20[newper20['Position']=='Guard']
newforwards=newper20[newper20['Position']=='Forward']
newcenters=newper20[newper20['Position']=='Center']


plt.figure(figsize=(20,10))
gspec = gridspec.GridSpec(nrows=4,ncols=4,wspace=0.1)
top_histogram = plt.subplot(gspec[0, 1:])
side_histogram = plt.subplot(gspec[0:, 0])
lower_right = plt.subplot(gspec[1:, 1:])


lower_right.scatter(mjguards['OWS'],mjguards['DWS'],alpha=0.5,color='r',label='Mj Guards')
lower_right.scatter(newguards['OWS'],newguards['DWS'],alpha=0.5,color='g',label='New Guards')
plt.axvline(x=8,linestyle='--',color='black')
plt.axhline(y=3,linestyle='--',color='black')
plt.yticks(np.arange(0,8,1))
plt.annotate(s="Great Defense, Great Offense", xy=(14,6.5),va="center", ha="center", size="medium"
             , bbox=dict(boxstyle="round", fc="w"))
plt.annotate(s="Great Defense", xy=(1.2,5.8),va="center", ha="center", size="medium",bbox=dict(boxstyle="round", fc="w"))
plt.annotate(s="Great Offense", xy=(15,0.1),va="center", ha="center", size="medium",bbox=dict(boxstyle="round", fc="w"))
for i in range(len(mjguards)):
    xy_text = (3, 5)
    name = mjguards.iloc[i]['Player']
    if (name == "Michael Jordan*")&(mjguards.iloc[i]['OWS']>12)&(mjguards.iloc[i]['DWS']>3):
        xy_text = (1, -14)
        plt.annotate(s= name
                , xy=(mjguards.iloc[i]['OWS']-1, mjguards.iloc[i]['DWS'])
                ,xycoords = "data", textcoords='offset points', xytext=xy_text,size='small')





top_histogram.hist(mjguards['OWS'],bins='auto',density=True,label='Mj Guards Off Win Share',color='g',alpha=0.5,stacked=True)
top_histogram.hist(newguards['OWS'],bins='auto',color='r',density=True,alpha=0.3,label='New era guards Off Win Share',stacked=True)

side_histogram.hist(mjguards['DWS'],bins='auto',orientation='horizontal',density=True,label='Mj Guards Def Win Share'
                    ,color='g',alpha=0.5,stacked=True)
side_histogram.hist(newguards['DWS'],bins='auto',orientation='horizontal',color='r',density=True,alpha=0.4
                    ,label='New era guards Def Win Share',stacked=True)


side_histogram.invert_xaxis()
side_histogram.yaxis.set_ticks_position('right')
top_histogram.yaxis.set_ticks_position('right')
lower_right.yaxis.set_ticks_position('right')
top_histogram.yaxis.set_label_position("right")

top_histogram.legend()
side_histogram.legend()
plt.xlabel('Offensive Win Share')
plt.ylabel('Defensive Win Share')
lower_right.yaxis.set_label_position("right")
plt.legend(loc=2)
plt.show()




####IN THE GRAPH ABOVE WE CAN NOTICE THAT PUTTING MJ WITH HIS STATS WITHOUT ADJUSTION HE WOULD DOMINATE
####EVERYTHING IN THE NEW ERA.
####BUT WHAT WOULD HAPPENED WHEN HIS GAME WOULD GET ADDAPTED??
####ILL START TRYING TO FIGURE OUT HIS NEW STATS.

###FIRST CREATING THE DATAFRAME OF THE MJ STATS(I AM TAKING THE TIME PERIOD 1988-1999,WITHOUT THE YEAR 1994 CUZ 
###MJ RETIRED
mjera=mjera.reset_index()
newmj=mjera[(mjera['Player']=='Michael Jordan*')&(mjera['Year']>1986)&(mjera['Year']<1999)&(mjera['Year']!=1994)]
newmj
###STARTING BY FINDING OUT THE DIFFERENCE IN ASSISTS,REBOUNDS,TURNOVERS.
###ILL CALCULATE THE MEAN ASTS,REB,TOV PRODUCED IN EACH PERIOD,CALCULATE THE DIFFERENCE AND FIND THE % OF THE DIFFERENCE
###THEN CHANGE MJ STATS ACCORDING TO THIS DIFFERENCE.
###TO FIND THE DIFFERENCE I AM GOING TO USE A FORMULA CREATED BY MYSELF,ILL DEVIDE THE PLAYERS INTO 3 CATEGORIES DEPENDING ON 
###THEIR PER(ILL USE HOLLINGERS FORMULA FOR ALLSTARS,MVPS AND NORMAL PLAYERS). ALLSTARS WILL HAVE A WEIGHT 0F 0.45(CUZ THEY ARE
###THE MOST IMPORTANT CATEGORY,THE CATEGORY CONTROLLING THE GAME)NORMAL PLAYERS WILL HAVE A WEIGHT OF 0.35 AND MVPS A WEIGHT OF
###0.20
###I AM GIVING A 80% INTO ALLSTARS AND MVPS COMBINED CUZ JORDAN IS AN MVP CALIBER PLAYER.
newmj=newmj.rename(columns={'2P_new':'2P','3PA_new':'3PA','2PA_new':'2PA','3P_new':'3P','PTS_newEra':'PTS'})


###NOW ILL WORK ON THE ASSISTS,REBOUNDS,TURNOVERS.
a=df[(df['Year']>2014)&((df['Position']=='Guard')|(df['Pos']=='SF'))&(df['G']>20)]
b=df[((df['Year']!=1994)&(df['Year']>1986))&((df['Position']=='Guard')|(df['Pos']=='SF'))&(df['Year']<1999)&(df['G']>20)]
####I AM NOT GOING TO USE POWER FORWARDS AND CENTERS CUZ I AM FOCUSING INTO GUARDS.
asd=['AST','TRB','STL','BLK','TOV']
for i in asd: ###ADDING PER GAME STATS
    a[i+'PG']=a[i]/a['G']
    b[i+'PG']=b[i]/b['G']

columns=['ASTPG','TRBPG','STLPG','BLKPG','TOVPG','PER','Year','Position','G']
a=a[columns]
b=b[columns]
def playerType(x): ###MY FUNCTION OF GETTING THE PLAYER TYPE
    if x<21:
        return ('Normal')
    if (x>21)&(x<27):
        return('Allstar')
    else:
        return('MVP')
a['PlayerType']=a['PER'].apply(playerType)
b['PlayerType']=b['PER'].apply(playerType)
a=a.groupby('PlayerType').agg({'ASTPG':np.mean,'TRBPG':np.mean,'TOVPG':np.mean,'STLPG':np.mean,'BLKPG':np.mean,'TOVPG':np.mean})
b=b.groupby('PlayerType').agg({'ASTPG':np.mean,'TRBPG':np.mean,'TOVPG':np.mean,'STLPG':np.mean,'BLKPG':np.mean,'TOVPG':np.mean})
a['Power']=[0.45,0.35,0.20]
b['Power']=[0.45,0.35,0.20]
for i in asd:
    a[i+'PG'+'MJ']=b[i+'PG']
for i in asd:
    a['Dif'+i+'PG']=((a[i+'PG']-a[i+'PGMJ'])*a['Power'])/a[i+'PGMJ']
q=[]

for i in asd:
    b=np.sum(a['Dif'+i+'PG'])
    q.append(b)
sums=pd.DataFrame(q,index=asd)
sums
####ADDING NOW THE SUMS INTO MJS STATS.
newmj['ASTPG']=newmj['AST']/newmj['G']
newmj['TRBPG']=newmj['TRB']/newmj['G']
newmj['BLKPG']=newmj['BLK']/newmj['G']
newmj['TOVPG']=newmj['TOV']/newmj['G']
newmj['STLPG']=newmj['STL']/newmj['G']

newmjast=newmj.groupby('Player').agg({'G':np.mean,'ASTPG':np.mean,'TRBPG':np.mean,'BLKPG':np.mean,'TOVPG':np.mean,'STLPG':np.mean,
                                    })

www=['AST','BLK','STL','TOV','TRB']
for i in www:
    newmjast['New'+i+'PG']=(newmjast[i+'PG']+newmjast[i+'PG']*sums.loc[i][0])
cc=['NewASTPG','NewBLKPG','NewSTLPG','NewTOVPG','NewTRBPG']
newmjstats=newmjast[cc]
newmjstats=newmjstats.rename(columns={'NewASTPG':'ASTpg','NewBLKPG':'BLKpg','NewSTLPG':'STLpg','NewTOVPG':'TOVpg'
                                      ,'NewTRBPG':'RPGpg'})

newmjstats
#NOW ILL MOVE ON INTO THE PERSONAL FOULS AND THEN INTO THE FREE THROWS OF MJ SO ILL HAVE HIS TOTAL POINTS.
###FIRST ILL START WITH THE FOULS AND THEN ILL MOVE INTO THE FREE THROWS
###ILL START WITH THE TOTAL FOULS,AND THEN ILL MOVE INTO THE FOUL BY EACH POSITION.
foulsnew=df[(df['Year']>2014)&(df['G']>20)&((df['Position']=='Guard')|(df['Pos']=='SF'))]
foulsold=df[((df['Year']>1986)&(df['Year']<1999)&(df['Year']!=1994))&(df['G']>20)&((df['Position']=='Guard')|(df['Pos']=='SF'))]
columns=['Year','Position','G','MP','PER','PF']
foulsnew=foulsnew[columns]
foulsold=foulsold[columns]
fouls=pd.merge(foulsold,foulsnew,how='outer')
fouls['PFpg']=round(fouls['PF']/fouls['G'],1)
fouls['MPpg']=round(fouls['MP']/fouls['G'],1)
fouls['PFperminute']=fouls['PFpg']/fouls['MPpg']
fouls['MinutesForFoul']=fouls['MPpg']/fouls['PFpg']
foulsg=fouls.groupby('Year').agg({'PFpg':np.mean,'PF':np.mean,'PFperminute':np.mean,'MinutesForFoul':np.mean})
###SO ILL FIND THE MEANS OF THE TWO ERAS.
foulsg=foulsg.reset_index()
a=np.mean(foulsg[foulsg['Year']<1999])
b=np.mean(foulsg[foulsg['Year']>1999])
foulsg.loc['NewMean']=b
foulsg.loc['OldMean']=a
foulsg=foulsg.drop(foulsg.index[:10])
del foulsg['Year']

foulsg.loc['Difference']=foulsg.loc['NewMean']-foulsg.loc['OldMean']
foulsg.loc['Difference%']=round(foulsg.loc['Difference']/foulsg.loc['NewMean'],3)
foulsg

####BELOW ILL CREATE A PLOT TO SHOW THESE DIFFERENCES.
####FIND THE PERCENTAGE OF DIFFERENCE,FIND THE NEW FOULS.
####SO I SEE WE HAVE 0.2% MORE PFpg.(HERE I DIDNT SEE A REASON TO FIND DIFFERENCE DEPENDING ON A PLAYERS PER.)
###LETS FIND THE PF FOR MJ
newmj['PFpg']=newmj['PF']/newmj['G']
newmj['PFpg']=newmj['PFpg']+foulsg.loc['Difference%']['PFpg']*newmj['PFpg']
newmjstats['PFpg']=np.mean(newmj['PFpg'])
newmjstats
###FREE THROWS IDEA.ILL CALCULATE THE PERCENTAGE OF DIFFERENCE IN FT AND ADD IT INTO MJ.(SAME FORMULA.)

ftold=df[((df['Year']!=1994)&(df['Year']>1986)&(df['Year']<1999))&(df['G']>20)&((df['Position']=='Guard')|(df['Pos']=='SF'))&(df['G']>20)]
ftnew=df[(df['Year']>2014)&(df['G']>20)&((df['Position']=='Guard')|(df['Pos']=='SF'))]
columns=['Year','Position','G','MP','PER','FTA']
ftold=ftold[columns]
ftnew=ftnew[columns]
ftold['FTApg']=round(ftold['FTA']/ftold['G'],1)
ftnew['FTApg']=round(ftnew['FTA']/ftnew['G'],1)
asd=[]
asdf=[]
for i in range(len(ftold.index)):
    if ftold.iloc[i]['PER']<21:
        asd.append('Normal')
    elif (ftold.iloc[i]['PER']>21)&(ftold.iloc[i]['PER']<27):
        asd.append('Allstar')
    else:
        asd.append('MVP')

for i in range(len(ftnew.index)):
    if ftnew.iloc[i]['PER']<21:
        asdf.append('Normal')
    elif (ftnew.iloc[i]['PER']>21)&(ftnew.iloc[i]['PER']<27):
        asdf.append('Allstar')
    else:
        asdf.append('MVP')

ftnew['PlayerType']=asdf        
ftold['PlayerType']=asd
ftnew=ftnew.groupby('PlayerType').agg({'FTApg':np.mean})
ftold=ftold.groupby('PlayerType').agg({'FTApg':np.mean})
ftnew=ftnew.rename(columns={'FTApg':'FTApg NewEra'})
ftnew['FTApg MjEra']=ftold['FTApg']
ftnew['Power']=[0.45,0.35,0.20]
ftnew['Difference']=ftnew['FTApg NewEra']-ftnew['FTApg MjEra']
ftnew['PowerDifference']=ftnew['Power']*ftnew['Difference']

ftnew['Difference%']=ftnew['PowerDifference']/ftnew['FTApg MjEra']
ftnew.loc['Sums']=ftnew.loc['Normal']+ftnew.loc['MVP']+ftnew.loc['Allstar']


newmj['FTApg']=newmj['FTA']/newmj['G']
newmj['FTApg']=newmj['FTApg']+newmj['FTApg']*ftnew.loc['Sums']['Difference%']
newmjstats['FTApg']=np.mean(newmj['FTApg'])
newmjstats

###NOW ILL TRY TO CALCULATE THE DIFFERENCE IN FGA,IN THREES AND TWOS,USING THE SAME FORMULA BUT EDITING IT A BIT BY ADDING
###THE USG% AS WELL.

stold=df[((df['Year']!=1994)&(df['Year']>1986)&(df['Year']<1999))&(df['G']>20)&((df['Position']=='Guard')|(df['Pos']=='SF'))&(df['G']>20)]
stnew=df[(df['Year']>2014)&(df['G']>20)&((df['Position']=='Guard')|(df['Pos']=='SF'))]
columns=['Year','Position','G','MP','PER','USG%','FGA','3PA','2PA']
stold=stold[columns]
stnew=stnew[columns]
stold['FGApg']=round(stold['FGA']/stold['G'],1)
stnew['FGApg']=round(stnew['FGA']/stnew['G'],1)
stnew['3Ppg']=round(stnew['3PA']/stnew['G'],1)
stold['3Ppg']=round(stold['3PA']/stold['G'],1)
stnew['2Ppg']=round(stnew['2PA']/stnew['G'],1)
stold['2Ppg']=round(stold['2PA']/stold['G'],1)
asd=[]
asdf=[]
for i in range(len(stold.index)):
    if stold.iloc[i]['PER']<21:
        asd.append('Normal')
    elif (stold.iloc[i]['PER']>21)&(stold.iloc[i]['PER']<27):
        asd.append('Allstar')
    else:
        asd.append('MVP')

for i in range(len(stnew.index)):
    if stnew.iloc[i]['PER']<21:
        asdf.append('Normal')
    elif (stnew.iloc[i]['PER']>21)&(stnew.iloc[i]['PER']<27):
        asdf.append('Allstar')
    else:
        asdf.append('MVP')

stnew['PlayerType']=asdf        
stold['PlayerType']=asd
###till now i ve used the same method as before.
###now ill try to create a new category depending on the USG%,since max is 41.7 and min is 8.1 ill use (41,7-8.1)/3 and ill
### have the 3 categories

asd=[]
asdf=[]
for i in range(len(stold.index)):
    if stold.iloc[i]['USG%']<19:
        asd.append('LowUSG%')
    elif (stold.iloc[i]['USG%']>19)&(stold.iloc[i]['USG%']<30):
        asd.append('MediumUSG%')
    else:
        asd.append('BigUSG%')

for i in range(len(stnew.index)):
    if stnew.iloc[i]['USG%']<19:
        asdf.append('LowUSG%')
    elif (stnew.iloc[i]['USG%']>19)&(stnew.iloc[i]['USG%']<30):
        asdf.append('MediumUSG%')
    else:
        asdf.append('BigUSG%')
stnew['PlayerUSG%']=asdf
stold['PlayerUSG%']=asd
stnewg=stnew.groupby(['PlayerType','PlayerUSG%']).agg({'FGApg':np.mean,'3Ppg':np.mean,'2Ppg':np.mean})
stoldg=stold.groupby(['PlayerType','PlayerUSG%']).agg({'FGApg':np.mean,'3Ppg':np.mean,'2Ppg':np.mean})
c=stoldg.columns
for i in c:
    stnewg[i+'MJera']=stoldg[i]
print(stnewg)
###NOW ILL TRY TO FIND THE POWERS OF EACH CATEGORY 
###FOR NORMAL AND LOWUSG% ILL ADD 0 POWER,FOR NORMAL AND MEDIUMUSG% ILL ADD 0.10 SINCE ITS THE COMMON FOR NORMAL PLAYERS TO HAVE
### MEDIUM USG%. FOR BIGUSG RATE ILL ADD 0.05 CUZ ITS BAD

###MOVING TO ALL STARS(I HAVE 0.4 OF THE PER POWER) SO ILL GIVE 0.15 TO MEDIUM USG% AND 0.25 TO BIG USG
###AND INTO MVPS I HAVE 0.35 SO IT IS 0.25 TO BIG USG% AND 0.10 TO MEDIUM USG%

stnewg['Power']=[0.25,0.20,0.25,0.1,0.05,0.05,0.1]
for i in c:
    stnewg['Dif'+i+'%']=((stnewg[i]-stnewg[i+'MJera'])*stnewg['Power'])/stnewg[i+'MJera']
columns=['Dif3Ppg%','Dif2Ppg%','DifFGApg%']
asd=[]
for i in columns:
    b=np.sum(stnewg[i])
    asd.append(b)
sums1=pd.DataFrame(asd,index=columns)
newmj['FGApg']=round(newmj['FGA']/newmj['G'],1)
newmj['3Ppg']=round(newmj['3PA']/newmj['G'],1)
newmj['2Ppg']=round(newmj['2PA']/newmj['G'],1)  
a=newmj.groupby('Player').agg({'FGApg':np.mean,'3Ppg':np.mean,'2Ppg':np.mean})
for i in a.columns:
    a[i]=a[i]+a[i]*sums1.loc['Dif'+i+'%'][0]
for i in a.columns:
    newmjstats[i]=a[i]

newmjstats
###MOVING INTO HIS SHOOTING for 2pts,3pts,ft.
###lets try to find if there is a difference with the above formula.

stold=df[((df['Year']>1986)&(df['Year']<1999)&(df['Year']!=1994))&(df['G']>20)&((df['Position']=='Guard')|(df['Pos']=='SF'))&(df['G']>20)]
stnew=df[(df['Year']>2014)&(df['G']>20)&((df['Position']=='Guard')|(df['Pos']=='SF'))]
columns=['Year','Position','G','MP','PER','USG%','3P%','2P%','FT%']
stold=stold[columns]
stnew=stnew[columns]
asd=[]
asdf=[]
for i in range(len(stold.index)):
    if stold.iloc[i]['PER']<21:
        asd.append('Normal')
    elif (stold.iloc[i]['PER']>21)&(stold.iloc[i]['PER']<27):
        asd.append('Allstar')
    else:
        asd.append('MVP')

for i in range(len(stnew.index)):
    if stnew.iloc[i]['PER']<21:
        asdf.append('Normal')
    elif (stnew.iloc[i]['PER']>21)&(stnew.iloc[i]['PER']<27):
        asdf.append('Allstar')
    else:
        asdf.append('MVP')

stnew['PlayerType']=asdf        
stold['PlayerType']=asd
###till now i ve used the same method as before.
###now ill try to create a new category depending on the USG%,since max is 41.7 and min is 8.1 ill use (41,7-8.1)/3 and ill
### have the 3 categories

asd=[]
asdf=[]
for i in range(len(stold.index)):
    if stold.iloc[i]['USG%']<19:
        asd.append('LowUSG%')
    elif (stold.iloc[i]['USG%']>19)&(stold.iloc[i]['USG%']<30):
        asd.append('MediumUSG%')
    else:
        asd.append('BigUSG%')

for i in range(len(stnew.index)):
    if stnew.iloc[i]['USG%']<19:
        asdf.append('LowUSG%')
    elif (stnew.iloc[i]['USG%']>19)&(stnew.iloc[i]['USG%']<30):
        asdf.append('MediumUSG%')
    else:
        asdf.append('BigUSG%')
stnew['PlayerUSG%']=asdf
stold['PlayerUSG%']=asd
stnewg=stnew.groupby(['PlayerType','PlayerUSG%']).agg({'3P%':np.mean,'2P%':np.mean,'FT%':np.mean})
stoldg=stold.groupby(['PlayerType','PlayerUSG%']).agg({'3P%':np.mean,'2P%':np.mean,'FT%':np.mean})
c=stoldg.columns
for i in c:
    stnewg[i+'MJera']=stoldg[i]
print(stnewg)
###NOW ILL TRY TO FIND THE POWERS OF EACH CATEGORY 
###FOR NORMAL AND LOWUSG% ILL ADD 0 POWER,FOR NORMAL AND MEDIUMUSG% ILL ADD 0.10 SINCE ITS THE COMMON FOR NORMAL PLAYERS TO HAVE
### MEDIUM USG%. FOR BIGUSG RATE ILL ADD 0.05 CUZ ITS BAD

###MOVING TO ALL STARS(I HAVE 0.4 OF THE PER POWER) SO ILL GIVE 0.15 TO MEDIUM USG% AND 0.25 TO BIG USG
###AND INTO MVPS I HAVE 0.35 SO IT IS 0.25 TO BIG USG% AND 0.10 TO MEDIUM USG%

stnewg['Power']=[0.25,0.20,0.25,0.1,0.05,0.05,0.1]

for i in c:
    stnewg['Dif'+i]=((stnewg[i]-stnewg[i+'MJera'])*stnewg['Power'])/stnewg[i+'MJera']
columns=['Dif3P%','Dif2P%','DifFT%']
asd=[]
for i in columns:
    b=np.sum(stnewg[i])
    asd.append(b)
sums2=pd.DataFrame(asd,index=columns)
a=newmj.groupby('Player').agg({'3P%':np.mean,'2P%':np.mean,'FT%':np.mean})
for i in a.columns:
    a[i]=a[i]+a[i]*sums2.loc['Dif'+i][0]
for i in a.columns:
    newmjstats[i]=a[i]

newmjstats
###HERE ILL CALCULATE MJS PPG

newmjstats['PPG']=(3*newmjstats['3Ppg']*newmjstats['3P%']+2*newmjstats['2P%']*newmjstats['2Ppg']+newmjstats['FT%']
                  *newmjstats['FTApg'])
for i in newmjstats.columns:
    newmjstats[i]=round(newmjstats[i],2)
newmjstats
###ILL USE HOLLINGER'S FORMULA TO CALCULATE THE NEW PER OF MJ.(FOUND THE FORMULA IN BASKETBALL REFERENCE)
###https://www.basketball-reference.com/about/per.html


###first ill find all the league elements(like league assists,etc)

a=df[df['Year']>2014]
a=a.groupby('Year').agg({'AST':np.sum,'FG':np.sum,'FT':np.sum,'PTS':np.sum,'FGA':np.sum,'ORB':np.sum
                        ,'TOV':np.sum,'FTA':np.sum,'TRB':np.sum,'PF':np.sum})
asd=[]
for i in a.columns:
    asd.append(round(np.mean(a[i]),2))

a.loc['Means']=asd
lg_means=pd.DataFrame(a.loc['Means'])

#CREATING THE ELEMENTS TO CALCULATE PER
factor=(2/3)-(0.5*(lg_means.loc['AST']['Means']/lg_means.loc['FG']['Means']))/(2*(lg_means.loc['FG']['Means']/lg_means.loc['FT']['Means']))
VOP=(lg_means.loc['PTS']['Means']/(lg_means.loc['FGA']['Means']-lg_means.loc['ORB']['Means']+lg_means.loc['TOV']['Means']
                        +lg_means.loc['FTA']['Means']))
DRB=(lg_means.loc['TRB']['Means']-lg_means.loc['ORB']['Means'])/lg_means.loc['TRB']['Means']

##TO CALCULATE ILL FIX THE TEAM STATS OF BULLS TO FIT INTO THE NEW ERA
## ILL USE THE SAME PERC OF DIFFERENCE
b=df[((df['Year']>1987)&(df['Year']<1999)&(df['Year']!=1994))&(df['Tm']=='CHI')]
b=b.groupby('Year').agg({'AST':np.sum,'FG':np.sum,'FG%':np.mean})

asd=[]
for i in b.columns:
    asd.append(round(np.mean(b[i]),2))

b.loc['Means']=asd
tm_means=pd.DataFrame(b.loc['Means'])


##ILL ADD THE DIFFERENCE OF ASSISTS AND FG
tm_means.loc['AST']['Means']=tm_means.loc['AST']['Means']+tm_means.loc['AST']['Means']*sums.loc['AST'][0]
tm_means.loc['FG']['Means']=tm_means.loc['FG']['Means']+tm_means.loc['FG']['Means']*(sums1.loc['DifFGApg%']*tm_means.loc['FG%']['Means'])
tm_means

##NOW ILL ADD TO newmjstats THE EXTRA ELEMENTS I NEED.
newmjstatss=pd.DataFrame()

newmjstatss['AST']=newmjstats['ASTpg']*81
newmjstatss['FG']=(newmjstats['3P%']*newmjstats['3Ppg']+newmjstats['2P%']*newmjstats['2Ppg'])*81
newmjstatss['TOV']=newmjstats['TOVpg']*81
newmjstatss['FGA']=(newmjstats['2Ppg']+newmjstats['3Ppg'])*81
newmjstatss['FTA']=newmjstats['FTApg']*81
newmjstatss['FT']=newmjstatss['FTA']*newmjstats['FT%']
newmjstatss['TRB']=newmjstats['RPGpg']*81
newmjstatss['ORB']=(np.mean(newmj['ORB'])/np.mean(newmj['TRB']))*newmjstatss['TRB']
newmjstatss['STL']=newmjstats['STLpg']*81
newmjstatss['BLK']=newmjstats['BLKpg']*81
newmjstatss['MP']=np.mean(newmj['MP'])
newmjstatss['3P']=newmjstats['3Ppg']*newmjstats['3P%']*81
newmjstatss['PF']=newmjstats['PFpg']*81
###NOW I AM READY TO CALCULATE THE PER.





a=(1/newmjstatss.loc['Michael Jordan*']['MP'])
b=newmjstatss.loc['Michael Jordan*']['3P']
c=(2/3)*newmjstatss.loc['Michael Jordan*']['AST']
d=(2-factor*(tm_means.loc['AST']['Means']/tm_means.loc['FG']['Means']))*newmjstatss.loc['Michael Jordan*']['FG']
e=(newmjstatss.loc['Michael Jordan*']['FT']*0.5*(1+(1-(tm_means.loc['AST']['Means']/tm_means.loc['FG']['Means']))+(2/3)*(tm_means.loc['AST']['Means']/tm_means.loc['FG']['Means'])))
f=VOP*newmjstatss.loc['Michael Jordan*']['TOV']
g=VOP*DRB*(newmjstatss.loc['Michael Jordan*']['FGA']-newmjstatss.loc['Michael Jordan*']['FG'])
h=VOP*0.44*(0.44+(0.56*DRB))*(newmjstatss.loc['Michael Jordan*']['FTA']-newmjstatss.loc['Michael Jordan*']['FT'])
i=VOP*(1-DRB)*(newmjstatss.loc['Michael Jordan*']['TRB']-newmjstatss.loc['Michael Jordan*']['ORB'])
k=VOP*DRB*newmjstatss.loc['Michael Jordan*']['ORB']
l=VOP*newmjstatss.loc['Michael Jordan*']['STL']
m=VOP*DRB*newmjstatss.loc['Michael Jordan*']['BLK']
n=newmjstatss.loc['Michael Jordan*']['PF']*((lg_means.loc['FT']['Means']/lg_means.loc['PF']['Means'])-0.44*(lg_means.loc['FTA']['Means']/lg_means.loc['PF']['Means'])*VOP)


uPER=a*(b+c+d+e-f-g-h+i+k+l+m-n)
uPER
###TO GO ON WITH MY CALCULATIONS ILL HAVE TO CALCULATE THE PACE 
###TO CALCULATE THE PACE I HAVE TO CALCULATE THE POSSESIONS.
###I LL TRY TO CALCULATE THE MEAN POSSESIONS OF EVERY TEAM(I AM THE USING THE FORMULA BEING USED IN NBA.STATS)
#https://fansided.com/2018/11/07/49ers-giants-worst-monday-night-football-game-ever/
a=df[df['Year']>2014] ### I AM TAKING THE MEAN OF EVERY STATS SINCE 2014
a=a.groupby(['Year','Tm']).agg({'FGA':np.sum,'FTA':np.sum,'ORB':np.sum,'DRB':np.sum,'TOV':np.sum,'FG':np.sum,'G':np.sum}).reset_index()
a=a.groupby('Tm').agg({'FGA':np.mean,'FTA':np.mean,'ORB':np.mean,'DRB':np.mean,'TOV':np.mean,'FG':np.mean,'G':np.mean}).reset_index()

b=pd.DataFrame(index=a['Tm'],columns=a['Tm'],data=None)
a=a.set_index('Tm')
asd=[]
for i in b.index:
    asdf=0
    for j in b.columns:
        b.loc[i][j]=((a.loc[i]['FGA']+a.loc[j]['FGA']+0.44*a.loc[i]['FTA']+0.44*a.loc[j]['FTA']-a.loc[i]['ORB']-
                              a.loc[j]['ORB']+a.loc[i]['TOV']+a.loc[j]['TOV'])/82)/2
        if i==j:
            b.loc[i][j]=0
        asdf=asdf+b.loc[i][j]
    asd.append(round(asdf/30,2))
b['MeanPoss']=asd
team_pos=b
team_pos

###NOW I LL MOVE TO CALCULATE THE PACE OF EVERY TEAM THE SAME WAY
a=a.reset_index()
c=pd.DataFrame(index=a['Tm'],columns=a['Tm'],data=None)
asd=[]
for i in c.index:
    asdf=0
    for j in c.columns:
        c.loc[i][j]=48*((team_pos.loc[i]['MeanPoss']+team_pos.loc[j]['MeanPoss'])/(2*(48)))
        if i==j:
            c.loc[i][j]=0
        asdf=asdf+c.loc[i][j]
    asd.append(round(asdf/30,2))
c['MeanPace']=asd
team_pace=c
team_pace=team_pace.drop('TOT')
###NOW CALCULATING THE pace adjustment.

lg_pace=np.mean(team_pace['MeanPace'])
team_pace['PaceAdj']=lg_pace/team_pace['MeanPace']
lg_pace
         
    
###I have to calculate the pace abj for mjera and find the difference
###then fix it
###then calculate it
a=df[(df['Year']>1987)&(df['Year']<1999)&(df['Year']!=1994)] ### I AM TAKING THE MEAN OF EVERY STATS SINCE 2014
a=a.groupby(['Year','Tm']).agg({'FGA':np.sum,'FTA':np.sum,'ORB':np.sum,'DRB':np.sum,'TOV':np.sum,'FG':np.sum,'G':np.sum}).reset_index()
a=a.groupby('Tm').agg({'FGA':np.mean,'FTA':np.mean,'ORB':np.mean,'DRB':np.mean,'TOV':np.mean,'FG':np.mean,'G':np.mean}).reset_index()

b=pd.DataFrame(index=a['Tm'],columns=a['Tm'],data=None)
a=a.set_index('Tm')
asd=[]
for i in b.index:
    asdf=0
    for j in b.columns:
        b.loc[i][j]=((a.loc[i]['FGA']+a.loc[j]['FGA']+0.44*a.loc[i]['FTA']+0.44*a.loc[j]['FTA']-a.loc[i]['ORB']-
                              a.loc[j]['ORB']+a.loc[i]['TOV']+a.loc[j]['TOV'])/82)/2
        if i==j:
            b.loc[i][j]=0
        asdf=asdf+b.loc[i][j]
    asd.append(round(asdf/30,2))
b['MeanPoss']=asd

team_pos1=b
team_pos1['CHI']['MeanPoss']=np.mean(team_pos['MeanPoss'])
a=a.reset_index()
c=pd.DataFrame(index=a['Tm'],columns=a['Tm'],data=None)
asd=[]
for i in c.index:
    asdf=0
    for j in c.columns:
        c.loc[i][j]=48*((team_pos1.loc[i]['MeanPoss']+team_pos1.loc[j]['MeanPoss'])/(2*(48)))
        if i==j:
            c.loc[i][j]=0
        asdf=asdf+c.loc[i][j]
    asd.append(round(asdf/30,2))
c['MeanPace']=asd
team_pace1=c
lg_pace1=np.mean(team_pace1['MeanPace'])
lg_paceDif=(lg_pace-lg_pace1)/lg_pace1 ### calculating the league pace diff%
####NOW ALL ADD IT INTO THE CHICAGO BULLS LEAGUE PACE DIFF%

team_pace1.loc['CHI']['MeanPace']
###AND ILL FIND TE PACE ADJUSMENT OF THE CHI BULLS WITH THE NEW LG PACE
pace_adj=team_pace1.loc['CHI']['MeanPace']/lg_pace
team_pace1.loc['CHI']['MeanPace']=team_pace1.loc['CHI']['MeanPace']+team_pace1.loc['CHI']['MeanPace']*lg_paceDif
team_pace1.loc['CHI']['MeanPace']
pace_adj=team_pace1.loc['CHI']['MeanPace']/lg_pace
pace_adj=np.mean(team_pace['PaceAdj'])#I cant adjust the bulls pace into the new eras pace
# so i am giving them the mean paceAdj
team_pos1['CHI']['MeanPoss']
###CALCULATING THE aPER(ACCORDING TO HOLLINGER FORMULA)
aPER=pace_adj*uPER
aPER
###TO FINALY CALCULATE PER I HAVE TO CALCULATE THE MEAN aPER OF THE LEAGUE,I HAVE THE PACE_ADJ OF EVERY TEAM,NOW ILL JUST
###HAVE TO FIND THE uPER OF EVERY PLAYER.
###BEFORE THE DEF I HAVE TO CALCULATE THE team means
#a1=mp a2=3P  a3=ast  a4=team_means_assists a5=tm_means_fg  a6=fg  a7=ft  a8=tov a9=fga a10=fg a11=fta a12=trb a13=orb
# a14=stl a15=blk a16=pf
team=df[df['Year']>2014]
team=team.groupby(['Tm','Year']).agg({'AST':np.sum,'FG':np.sum})
team=team.rename(columns={'AST':'Tm_AST','FG':'Tm_FG'})
df2=df[df['Year']>2014]


def calculateuPER(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16):
    a=(1/a1)
    b=a2
    c=(2/3)*a3
    d=(2-factor*(a4/a5))*a6
    e=(a7*0.5*(1+(1-(a4/a5))+(2/3)*(a4/a5)))
    f=VOP*a8
    g=VOP*DRB*(a9-a10)
    h=VOP*0.44*(0.44+(0.56*DRB))*(a11-a7)
    i=VOP*(1-DRB)*(a12-a13)
    k=VOP*DRB*a13
    l=VOP*a14
    m=VOP*DRB*a15
    n=a16*((lg_means.loc['FT']['Means']/lg_means.loc['PF']['Means'])-0.44*(lg_means.loc['FTA']['Means']/lg_means.loc['PF']['Means'])*VOP)
    
    asd=a*(b+c+d+e-f-g-h+i+k+l+m-n)
    return(asd)
df2=df2.sort_values('Tm')
df2=df2.reset_index()
df2=df2.set_index('Tm')
df2=df2.drop('TOT')
df2=df2.reset_index()
df2=df2.set_index(['Tm','Year'])
qq=[]
ss=[]
for i in df2.index:
    a=team.loc[i]['Tm_AST']
    qq.append(a)
    
    ss.append(team.loc[i]['Tm_FG'])
df2['Tm_AST']=qq
df2['Tm_FG']=ss
df2=df2.reset_index()
df2=df2.set_index('Player')

df2['uPER']=calculateuPER(df2['MP'],df2['3P'],df2['AST'],df2['Tm_AST'],df2['Tm_FG'],df2['FG'],df2['FT'],df2['TOV'],
                         df2['FGA'],df2['FG'],df2['FTA'],df2['TRB'],df2['ORB'],df2['STL'],df2['BLK'],df2['PF'])
df2=df2.reset_index()
df2=df2.set_index('Tm')

#a1=mp a2=3P  a3=ast  a4=team_means_assists a5=tm_means_fg  a6=fg  a7=ft  a8=tov a9=fga a10=fg a11=fta a12=trb a13=orb
# a14=stl a15=blk a16=pf
ss=[]
for i in df2.index:
    a=team_pace.loc[i]['PaceAdj']
    ss.append(a) 
df2['Pace_adj']=ss
df2=df2.reset_index()
df2=df2.set_index('Player')
df2['aPER']=df2['uPER']*df2['Pace_adj']
 ### TO COMPLETE THE FORMULA ILL CALCULATE AVERAGE aPER USING PLAYERS MP AS WEIGHTS(AS THE FORMULA IN BASKETBALL REFERENCE SAYS)
lg_aPER=np.average(df2['aPER'],weights=df2['MP'])
####I HAVE EVERYTHING!!!
###LETS FIND JORDANS PER.
###
MJ_PER=aPER*(15/lg_aPER)
MJ_PER
### ILL SEE IF MY FORMULA IS CORRECT AND TRY THE ERROR IN MY ESTIMATION 
df2['PER1']=df2['aPER']*(15/lg_aPER)
estimation_error=np.mean(df2['PER1']-df2['PER'])/np.mean(df2['PER'])

newmjstats['PER']=MJ_PER
###ITS AN ERROR THAT I ASSUME IS BECAUSE IVE CONSIDERED NOT JUST A YEAR BUT ALL THE YEARS AFTER 2014.
estimation_error
###ILL TRY TO CALCULATE WIN SHARES NOW. ILL USE THE FORMULA IN BASKETBALL REFERENCE.:
#https://www.basketball-reference.com/about/ratings.html
#and https://www.basketball-reference.com/about/ws.html
newmjstatss['PTS']=newmjstats['PPG']*81
newmjstatss

###STARTING WITH THE STATS I NEED FROM EVERY TEAM.
dfteam=df[((df['Year']>1987)&(df['Year']<1999))&(df['Year']!=1994)]
dfteam=dfteam[dfteam['Tm']=='CHI']
dfteam=dfteam.groupby('Year').agg({'MP':np.sum,'AST':np.sum,'FG':np.sum,'TRB':np.sum,
                                     'PTS':np.sum,'FT':np.sum,'FGA':np.sum,
                                     'ORB%':np.mean,'ORB':np.sum,'FTA':np.sum,
                                    'TOV':np.sum,'3P':np.sum,'STL':np.sum,
                                  'BLK':np.sum,'PF':np.sum,'DRB':np.sum,'3P':np.sum,
                                  '2P':np.sum,'3P%':np.mean,'2P%':np.mean,'2PA':np.sum,
                                  '3PA':np.sum,'FT%':np.mean})
asd=[]

for i in dfteam.columns:
    
    asd.append(np.mean(dfteam[i]))

dfteam.loc['Means']=asd

####THEN I NEED TO FIND OPPOMENT STATS
####SO ILL GET THE MEAN VALUE OF EVERY TEAM IN ALL THESE YEARS EXCEPT THE CHI.
dfva=df[(df['Year']>2014)]
dfva=dfva.groupby(['Year','Tm']).agg({'TRB':np.sum,'ORB':np.sum,'FGA':np.sum,
                                     'FG':np.sum,'TOV':np.sum,'FTA':np.sum,'BLK':np.sum,
                                     'FT':np.sum,'MP':np.sum,'PTS':np.sum,'FT%':np.mean,'STL':np.sum,
                                     '2PA':np.sum,'2P%':np.mean,'3PA':np.sum,'3P%':np.mean,'AST':np.sum,'PF':np.sum}).reset_index()
dfva=dfva.groupby('Year').agg({'TRB':np.mean,'ORB':np.mean,'FGA':np.mean,
                                     'FG':np.mean,'TOV':np.mean,'FTA':np.mean,'BLK':np.mean,
                                     'FT':np.mean,'MP':np.mean,'PTS':np.mean,'FT%':np.mean,'STL':np.mean,
                                     '2PA':np.mean,'2P%':np.mean,'3PA':np.mean,'3P%':np.mean,'AST':np.mean,'PF':np.mean}).reset_index()

dfteam
###NOW I HAVE ALL THE VARIABLES,SO I AM STARTING CALCULATING THE OFF.WINSHARE
###ILL HAVE TO ADJUCT THE CHICAGO BULLS STATS INTO THE NEW ERA.
dfvava=df[(df['Year']>1988)&(df['Year']<1999)&(df['Year']!=1994)]
dfvava=dfvava.groupby(['Year','Tm']).agg({'TRB':np.sum,'ORB':np.sum,'FGA':np.sum,
                                     'FG':np.sum,'TOV':np.sum,'FTA':np.sum,'BLK':np.sum,
                                     'FT':np.sum,'MP':np.sum,'PTS':np.sum,'FT%':np.mean,'STL':np.sum,
                                     '2PA':np.sum,'2P%':np.mean,'3PA':np.sum,'3P%':np.mean,'AST':np.sum,'PF':np.sum}).reset_index()
dfvava=dfvava.groupby('Year').agg({'TRB':np.mean,'ORB':np.mean,'FGA':np.mean,
                                     'FG':np.mean,'TOV':np.mean,'FTA':np.mean,'BLK':np.mean,
                                     'FT':np.mean,'MP':np.mean,'PTS':np.mean,'FT%':np.mean,'STL':np.mean,
                                     '2PA':np.mean,'2P%':np.mean,'3PA':np.mean,'3P%':np.mean,'AST':np.mean,'PF':np.mean}).reset_index()
dfvava['DRB']=dfvava['TRB']-dfvava['ORB']
dfvava.loc['Means']=np.mean(dfvava[dfvava.columns])
dfva['DRB']=dfva['TRB']-dfva['ORB']
dfva.loc['Means']=np.mean(dfva[dfva.columns])
sumsums=pd.DataFrame(columns=dfvava.columns,data=None)
asd=[]
for i in dfva.columns:
    a=(dfva.loc['Means'][i]-dfvava.loc['Means'][i])/dfvava.loc['Means'][i]
    asd.append(a)
    
sumsums.loc['Means']=asd
sumsums
###NOW THAT IVE CALCULATED THE DIFFERENCES ILL FIND THE NEW TEAM STATS.

dfteam['AST']=dfteam['AST']+dfteam['AST']*sums.loc['AST'][0]
dfteam['ORB']=dfteam['TRB']-dfteam['DRB']
dfteam['TOV']=dfteam['TOV']+dfteam['TOV']*sums.loc['TOV'][0]
dfteam['3PA']=dfteam['3PA']+dfteam['3PA']*sums1.loc['Dif3Ppg%'][0]
dfteam['DRB']=dfteam['TRB']-dfteam['ORB']
dfteam['STL']=dfteam['STL']+dfteam['STL']*sums.loc['STL'][0]
dfteam['BLK']=dfteam['BLK']+dfteam['BLK']*sums.loc['BLK'][0]
dfteam['2PA']=dfteam['2PA']+dfteam['2PA']*sums1.loc['Dif2Ppg%'][0]
dfteam['FGA']=dfteam['FGA']+dfteam['FGA']*sums1.loc['DifFGApg%'][0]
dfteam['3P%']=dfteam['3P%']+dfteam['3P%']*sums2.loc['Dif3P%'][0]
dfteam['2P%']=dfteam['2P%']+dfteam['2P%']*sums2.loc['Dif2P%'][0]
dfteam['3P']=dfteam['3PA']*dfteam['3P%']
dfteam['2P']=dfteam['2PA']*dfteam['2P%']
dfteam['PTS']=dfteam['PTS']+dfteam['PTS']*sumsums.loc['Means']['PTS']


Opponent_TRB=dfva.loc['Means']['TRB']
Opponent_ORB=dfva.loc['Means']['ORB']
Opponent_FG=dfva.loc['Means']['FG']
Opponent_FGA=dfva.loc['Means']['FGA']
Opponent_TOV=dfva.loc['Means']['TOV']
Opponent_FTA=dfva.loc['Means']['FTA']
Opponent_FT=dfva.loc['Means']['FT']
Opponent_MP=dfva.loc['Means']['MP']
Opponent_PTS=dfva.loc['Means']['PTS']

qAST=((newmjstatss.loc['Michael Jordan*']['MP']/(dfteam.loc['Means']['MP']/5))*
      (1.14*((dfteam.loc['Means']['AST']-newmjstatss.loc['Michael Jordan*']['AST'])/
             dfteam.loc['Means']['FG'])))+((((dfteam.loc['Means']['AST']/dfteam.loc['Means']['MP'])*
                                           newmjstatss.loc['Michael Jordan*']['MP']*5-newmjstatss.loc['Michael Jordan*']['AST'])
                                          /((dfteam.loc['Means']['FG']/dfteam.loc['Means']['MP'])*newmjstatss.loc['Michael Jordan*']['MP']*5-
                                           newmjstatss.loc['Michael Jordan*']['FG']))*(1-(newmjstatss.loc['Michael Jordan*']['MP']/
                                                                                         (dfteam.loc['Means']['MP']/5))))
FG_Part=newmjstatss.loc['Michael Jordan*']['FG']*(1-0.5*((
        newmjstatss.loc['Michael Jordan*']['PTS']-newmjstatss.loc['Michael Jordan*']['FT'])
                                                        /(2*newmjstatss.loc['Michael Jordan*']['FGA']))*qAST)

AST_Part=0.5*(((dfteam.loc['Means']['PTS']-dfteam.loc['Means']['FT'])-
             (newmjstatss.loc['Michael Jordan*']['PTS']-newmjstatss.loc['Michael Jordan*']['FT']))
              /(2*(dfteam.loc['Means']['FGA']-newmjstatss.loc['Michael Jordan*']['FGA'])))*newmjstatss.loc['Michael Jordan*']['AST']

FT_Part=(1-(1-(newmjstatss.loc['Michael Jordan*']['FT']/
              newmjstatss.loc['Michael Jordan*']['FTA']))**2)*0.4*newmjstatss.loc['Michael Jordan*']['FTA']

Team_Scoring_Poss=dfteam.loc['Means']['FG']+(1-(1-(
        dfteam.loc['Means']['FT']/dfteam.loc['Means']['FTA']))**2)*dfteam.loc['Means']['FTA']*0.4

Team_ORB=dfteam.loc['Means']['ORB']/(dfteam.loc['Means']['ORB']+
                                    (Opponent_TRB-Opponent_ORB))

Team_Play=Team_Scoring_Poss/(dfteam.loc['Means']['FGA']+dfteam.loc['Means']['FTA']*
                             0.4+dfteam.loc['Means']['TOV'])

Team_ORB_Weight=((1-Team_ORB)*Team_Play)/((1-Team_ORB)*
                                         Team_Play+Team_ORB*(1-Team_Play))

ORB_Part=newmjstatss.loc['Michael Jordan*']['ORB']*Team_ORB_Weight*Team_Play


###SO MOVING INTO THE SCORING POSSESSIONS:
ScPoss=(FG_Part+AST_Part+FT_Part)*(1-(dfteam.loc['Means']['ORB']/Team_Scoring_Poss)
                                  *Team_ORB_Weight*Team_Play)+ORB_Part
#NOW FOR THE MISSED FG,MISSED FT POSESSIONS:
FGxPoss=(newmjstatss.loc['Michael Jordan*']['FGA']-newmjstatss.loc['Michael Jordan*']['FG'])*(1-1.07*Team_ORB)
FTxPoss=((1-(newmjstatss.loc['Michael Jordan*']['FT']/newmjstatss.loc['Michael Jordan*']['FTA']))**2)*0.4*newmjstatss.loc['Michael Jordan*']['FTA']

##SO THE TOTAL POSSESIONS ARE:
TotPoss=ScPoss+FGxPoss+FTxPoss+newmjstatss.loc['Michael Jordan*']['TOV']

#MOVING INTO THE POINTS PRODUCED.
#FIRST CALCULATING:
PProd_FG_Part=2*(newmjstatss.loc['Michael Jordan*']['FG']+0.5*
                newmjstatss.loc['Michael Jordan*']['3P'])*(1-0.5*((
        newmjstatss.loc['Michael Jordan*']['PTS']-newmjstatss.loc['Michael Jordan*']['FT'])
                                                                 /(2*newmjstatss.loc['Michael Jordan*']['FGA']))*qAST)
PProd_AST_Part=2*((dfteam.loc['Means']['FG']-newmjstatss.loc['Michael Jordan*']['FG']+
                  0.5*(dfteam.loc['Means']['3P']-newmjstatss.loc['Michael Jordan*']['3P']))/
                 (dfteam.loc['Means']['FG']-newmjstatss.loc['Michael Jordan*']['FG']))*0.5*(((
        dfteam.loc['Means']['PTS']-dfteam.loc['Means']['FT'])-(newmjstatss.loc['Michael Jordan*']['PTS']-
                                                                                            newmjstatss.loc['Michael Jordan*']['FT']))/
                                                                                            (2*(dfteam.loc['Means']['FGA']-newmjstatss.loc['Michael Jordan*']['FGA']
                                                                                             )))*newmjstatss.loc['Michael Jordan*']['AST']
PProd_ORB_Part=newmjstatss.loc['Michael Jordan*']['ORB']*Team_ORB_Weight*Team_Play*(
dfteam.loc['Means']['PTS']/(dfteam.loc['Means']['FG']+(1-(1-(dfteam.loc['Means']['FT']/dfteam.loc['Means']['FTA']))**2)
                           *0.4*dfteam.loc['Means']['FTA']))
##FINALY:
PProd=(PProd_FG_Part+PProd_AST_Part+newmjstatss.loc['Michael Jordan*']['FT'])*(
1-(dfteam.loc['Means']['ORB']/Team_Scoring_Poss)*Team_ORB_Weight*Team_Play)+PProd_ORB_Part

##MOVING INTO MARGINAL OFFENSE 
###I HAVE TO CALCULATE POINTS PER POSSESIONS,I HAVE POSSESIONS OF EVERY TEAM,SO ILL FIND THE MEAN
###AND THEN THE MEAN POINTS AND DEVIDE THEM
lg_poss=np.mean(team_pos['MeanPoss']) ###(TOTAL POSSESIONS OF EVER TEAM PER GAME)
df99=df[df['Year']>2014].groupby(['Year','Tm']).agg({'PTS':np.sum})
df99['PTSpg']=df99['PTS']/82
Points_PG=np.mean(df99['PTSpg'])
lg_points_per_poss=Points_PG/lg_poss #(I AM MULTYPLYING BY 2 BECAUSE POINTS PG IS FOR ONE TEAM AND POSSESIONS FOR TWO TEAMS,SO I NEED IT BALANCED)
###SO I HAVE:
marginal_off=PProd-0.92*(lg_points_per_poss)*TotPoss
marginal_off
###MOVING INTO THE MARGINAL POINTS PER WINS:
marginal_pts_per_win=0.32*(Points_PG)*pace_adj
###SO I HAVE THE WIN SHARE OF THE PLAYER:
OWS=marginal_off/marginal_pts_per_win
OWS
#DEFENSIVE WIN SHARE.USING THE FORMULA IN BASKETBAL REFERENCE I HAVE:

DOR=Opponent_ORB/(Opponent_ORB+dfteam.loc['Means']['DRB'])
DFG=Opponent_FG/Opponent_FGA
FMwt=(DFG*(1-DOR))/(DFG*(1-DOR)+(1-DFG)*DOR)

Stops1=newmjstatss.loc['Michael Jordan*']['STL']+newmjstatss.loc['Michael Jordan*']['BLK']*FMwt*(
1-1.07*DOR)+DRB*(1-FMwt)

Stops2=(((Opponent_FGA-Opponent_FG-dfteam.loc['Means']['BLK'])/dfteam.loc['Means']['MP'])*FMwt*(1-1.07*DOR)+(
        (Opponent_TOV-dfteam.loc['Means']['STL'])/dfteam.loc['Means']['MP'])) *newmjstatss.loc['Michael Jordan*']['MP']+(
    newmjstatss.loc['Michael Jordan*']['PF']/dfteam.loc['Means']['PF'])* 0.4*Opponent_FTA*(1-(Opponent_FT/Opponent_FTA))**2
       
Stops=Stops1+Stops2

##Calculating the Stop% which is the rate of a player forcing a stop.
Stop=(Stops*Opponent_MP)/(team_pos1.loc['CHI']['MeanPoss']*82*newmjstatss.loc['Michael Jordan*']['MP'])
            
###MOVING ON ON CALCULATING THE VARIABLES FOR THE DRTG
Team_Defensive_Rating=100*(Opponent_PTS/82)/(team_pos1['CHI']['MeanPoss'])
D_PTS_per_ScPoss=Opponent_PTS/(Opponent_FG+(1-(1-(Opponent_FT/Opponent_FTA))**2)*Opponent_FTA*0.4)

###AND NOW THE DRtg
DRtg=Team_Defensive_Rating+0.2*(100*D_PTS_per_ScPoss*(1-Stop)-Team_Defensive_Rating)


#MOVING ON FOR THE DWS
team_def_pos=dfteam.loc['Means']['FGA']-dfteam.loc['Means']['ORB']+dfteam.loc['Means']['TOV']+(0.4*dfteam.loc['Means']['FTA'])
Marg_Def=(newmjstatss.loc['Michael Jordan*']['MP']/dfteam.loc['Means']['MP'])*team_def_pos*(1.08*(lg_points_per_poss)-(DRtg/100))


DWS=Marg_Def/marginal_pts_per_win
DWS
##PUTTING ALL THE STATS TOGETHER AND ADDING AN EXTRA COUPLE OF BASICS STATS
newmjstatss['DRB']=newmjstatss['TRB']-newmjstatss['ORB']
newmjstatss['2P']=newmjstats['2Ppg']*81
newmjstatss['3P%']=newmjstats['3P%']
newmjstatss['2P%']=newmjstats['2P%']
newmjstatss['FT%']=newmjstats['FT%']
newmjstatss['FG%']=newmjstatss['FG']/newmjstatss['FGA']
newmjstatss['PER']=MJ_PER
newmjstatss['OWS']=OWS
newmjstatss['DWS']=DWS
newmjstatss['WS']=OWS+DWS
newmjstatss['3PAr']=newmjstatss['3P']/newmjstatss['FGA']
newmjstatss['USG%']=100*((newmjstatss.loc['Michael Jordan*']['FGA']+0.44*newmjstatss.loc['Michael Jordan*']['FTA']+
                         newmjstatss.loc['Michael Jordan*']['TOV'])*(dfteam.loc['Means']['MP']/5))/(newmjstatss.loc['Michael Jordan*']['MP']*(dfteam.loc['Means']['FGA']+
                                                                                                                                 0.44*dfteam.loc['Means']['FTA']+
                                                                                                                                 dfteam.loc['Means']['TOV']))
TSA=newmjstatss.loc['Michael Jordan*']['FGA']+0.44*newmjstatss.loc['Michael Jordan*']['FTA']
newmjstatss['TS%']=(newmjstatss.loc['Michael Jordan*']['PTS']/(2*TSA))
newmjstatss['G']=81
###TIME TO COMPARE THE NEW STATS TO THE OTHER PLAYERS.ILL CREATE A DF OF THEM.
fin=df[df['Year']>2014].reset_index()
finback=df[(df['Year']>1987)&(df['Year']!=1994)&(df['Year']<1999)].reset_index()
d={x:np.mean for x in newmjstatss.columns}
fin=fin.groupby('Player').agg(d)
finback=finback.groupby('Player').agg(d)
fin.loc['Michael Jordan']=newmjstatss.loc['Michael Jordan*']
finbest=fin[(fin['G']>50)&(fin['PER']>22)]
finbackbest=finback[(finback['G']>50)&(finback['PER']>22)]
###STARTING WITH POINTS ASSISTS TRB
finbest['PTSformula']=finbest['PTS']/np.max(finbest['PTS'])
finbest['TRBformula']=finbest['TRB']/np.max(finbest['TRB'])
finbest['ASTformula']=finbest['AST']/np.max(finbest['AST'])
finbackbest['PTSformula']=finbackbest['PTS']/np.max(finbackbest['PTS'])
finbackbest['TRBformula']=finbackbest['TRB']/np.max(finbackbest['TRB'])
finbackbest['ASTformula']=finbackbest['AST']/np.max(finbackbest['AST'])
finbest=finbest.sort_values('PTS')
finbackbest=finbackbest.sort_values('PTS')
finbackbest=finbackbest.reset_index()
finbest=finbest.reset_index()


fig=plt.figure(figsize=(22,11))
ax=fig.add_subplot(121)
a=finbest.plot('Player','PTSformula',kind='barh',ax=ax,figsize=(14,12),color='r')
b=finbest.plot('Player','ASTformula',kind='barh',ax=ax,figsize=(14,12),color='g',align='edge')
plt.tick_params(top=False,bottom=False,right=False,labelbottom=False)
plt.legend()

ax1=fig.add_subplot(122)
a=finbackbest.plot('Player','PTSformula',kind='barh',ax=ax1,figsize=(14,12),color='r',label='Points')
b=finbackbest.plot('Player','ASTformula',kind='barh',ax=ax1,figsize=(14,12),color='g',align='edge',label='Assist')
plt.tick_params(top=False,bottom=False,right=False,labelbottom=False)
ax1.yaxis.tick_right()

plt.show()
#MJ HAS THE MOST POINTS,LETS SEE IF POINTS ARE IMPORTANT IN WIN SHARES
fin['PPG']=fin['PTS']/fin['G']
finback['PPG']=finback['PTS']/finback['G']
finpg=fin[fin['PPG']>15]##ILL ONLY USE PLAYERS WITH MORE THAN 15 PPG
finbackpg=finback[finback['PPG']>15]
finpg=finpg.reset_index()
finbackpg=finbackpg.reset_index()

fig=plt.figure(figsize=(25,15))

ax=fig.add_subplot(121)

finpg.plot('WS','PPG',kind='scatter',figsize=(20,10),ax=ax)
for i in range(len(finpg)):
    if finpg.iloc[i]['WS']>6:
        xy_text = (1, 3)
        name = finpg.iloc[i]['Player']
    
        plt.annotate(s= name
            , xy=(finpg.iloc[i]['WS']-0.5, finpg.iloc[i]['PPG'])
            ,xycoords = "data", textcoords='offset points', xytext=xy_text,size='small')

        
        
ax1=fig.add_subplot(122)
finbackpg.plot('WS','PPG',kind='scatter',figsize=(20,10),ax=ax1)
for i in range(len(finbackpg)):
    if finbackpg.iloc[i]['WS']>6:
        xy_text = (1, 3)
        name = finbackpg.iloc[i]['Player']
    
        plt.annotate(s= name
            , xy=(finbackpg.iloc[i]['WS']-0.5, finbackpg.iloc[i]['PPG'])
            ,xycoords = "data", textcoords='offset points', xytext=xy_text,size='small')


plt.show()
#MOVING ON OFFENSIVE WIN SHARES AND DEFENSIVE WIN SHARES
finws=fin.sort_values('WS',ascending=False)[:100]
finbackws=finback.sort_values('WS',ascending=False)[:100]
finws=finws.reset_index()
finbackws=finbackws.reset_index()

fig=plt.figure(figsize=(25,15))


ax=fig.add_subplot(121)

a=finws.plot('OWS','DWS',kind='scatter',figsize=(20,10),ax=ax)
a.axvline(x=8,color='black')
a.axhline(y=3.5,color='black')
plt.xticks(np.arange(0,15,1))

for i in range(len(finws)):
    if (finws.iloc[i]['OWS']>8):
        xy_text = (1, 3)
        name = finws.iloc[i]['Player']
    
        plt.annotate(s= name
            , xy=(finws.iloc[i]['OWS']-0.5, finws.iloc[i]['DWS'])
            ,xycoords = "data", textcoords='offset points', xytext=xy_text,size='small')

        
for i in range(len(finws)):   
    if finws.iloc[i]['DWS']>3.5:
        xy_text = (1, 3)
        name = finws.iloc[i]['Player']
    
        plt.annotate(s= name
            , xy=(finws.iloc[i]['OWS']-0.5, finws.iloc[i]['DWS'])
            ,xycoords = "data", textcoords='offset points', xytext=xy_text,size='small')   


ax1=fig.add_subplot(122)
a=finbackws.plot('OWS','DWS',kind='scatter',figsize=(20,10),ax=ax1)
a.axvline(x=8,color='black')
a.axhline(y=3.5,color='black')
plt.xticks(np.arange(0,15,1))

for i in range(len(finbackws)):
    if (finbackws.iloc[i]['OWS']>8):
        xy_text = (1, 3)
        name = finbackws.iloc[i]['Player']
    
        plt.annotate(s= name
            , xy=(finbackws.iloc[i]['OWS']-0.5, finbackws.iloc[i]['DWS'])
            ,xycoords = "data", textcoords='offset points', xytext=xy_text,size='small')

        
for i in range(len(finbackws)):   
    if finbackws.iloc[i]['DWS']>3.5:
        xy_text = (1, 3)
        name = finbackws.iloc[i]['Player']
    
        plt.annotate(s= name
            , xy=(finbackws.iloc[i]['OWS']-0.5, finbackws.iloc[i]['DWS'])
            ,xycoords = "data", textcoords='offset points', xytext=xy_text,size='small')  


        
        
plt.show()
###USAGE RATE AND TRUE SHOOTING
finus=fin[fin['G']>50]
finus=finus.sort_values('USG%',ascending=False)[:100]
finus=finus.reset_index()
finbackus=finback[finback['G']>50]
finbackus=finbackus.sort_values('USG%',ascending=False)[:100]
finbackus=finbackus.reset_index()

fig=plt.figure(figsize=(25,15))



ax=fig.add_subplot(121)
a=finus.plot('USG%','TS%',kind='scatter',figsize=(20,10),ax=ax)
for i in range(len(finus)):
    if (finus.iloc[i]['USG%']>28):
        xy_text = (1, 3)
        name = finus.iloc[i]['Player']
    
        plt.annotate(s= name
            , xy=(finus.iloc[i]['USG%']-0.5, finus.iloc[i]['TS%'])
            ,xycoords = "data", textcoords='offset points', xytext=xy_text,size='small')
        
        
ax1=fig.add_subplot(122)
a=finbackus.plot('USG%','TS%',kind='scatter',figsize=(20,10),ax=ax1)
for i in range(len(finbackus)):
    if (finbackus.iloc[i]['USG%']>28):
        xy_text = (1, 3)
        name = finbackus.iloc[i]['Player']
    
        plt.annotate(s= name
            , xy=(finbackus.iloc[i]['USG%']-0.5, finbackus.iloc[i]['TS%'])
            ,xycoords = "data", textcoords='offset points', xytext=xy_text,size='small')



plt.show()
###PER AND MVPS:
finper=fin[fin['G']>50]
finper=finper.sort_values('PER',ascending=False)[:15]
finper=finper.reset_index()
finbackper=finback[finback['G']>50]
finbackper=finbackper.sort_values('PER',ascending=False)[:15]
finbackper=finbackper.reset_index()

fig=plt.figure(figsize=(25,15))



ax=fig.add_subplot(121)
ax=finper.plot('Player','PER',figsize=(20,10),kind='bar',ax=ax)
plt.xticks(rotation=20)
plt.yticks(np.arange(0,30,2))

ax1=fig.add_subplot(122)
ax=finbackper.plot('Player','PER',figsize=(20,10),kind='bar',ax=ax1)
plt.xticks(rotation=20)
plt.yticks(np.arange(0,30,2))


plt.show()
####JORDANS PER WHOULD BE NEARLY THE SAME IN THIS ERA AND AND HIS ERA,THAT MEANS HE WOULD STILL BE A MVP-SUPERSTAR
####HE WOULD ALSO LEAD THE LEAGUE IN POINTS,MEANING HE WOULD BE THE SCORING MACHINE HE WAS,BUT SCORING LESS
####THIS MUST BE CUZ HE WOULD SHOT MORE 3s AND HIS % ISNT GREAT(COMPARING IT WITH HIS AUTOMATIC MID RANGE SHOT)
####E WONT BE THE SAME DOMINANT PLAYER HE WAS,HE WILL HAVE A LOT OF COMPETITION WITH GREAT PLAYERS


####I CAN ALSO NOTICE HIS DWS ARE GETTING LOWER,MJ WAS A DEFENSIVE PLAYER OF THE YEAR,BUT IN THE NEW ERA HE WONT DOMINATE ON D
####MAYBE CUZ PLAYERS ARE BETTER OFFENSIVELY,MAYBE BECAUSE DEFENSE HAS CHANGE IN THIS ERA
####HE WOULD BE A JAMES HARDEN TYPE PLAYER(MAYBE A BETTER VERSION),A SCORING MACHINE LEADING HIS TEAM
#####BUT WITH BETTER PER,ALMOST THE SAME WS,MJ WOULD HAVE MORE OWS AND LESS DWS
####WOULD HE WIN AN MVP??JARDEN WITH OVER 26 PER WON SO MJ WITH HIGHEST PER WOULD ALSO WIN
####AS FOR CHAMPION DEPENDS ON THE TEAM HE HAVE.
####NOT THE PURE DOMINANCE HE HAS BACK IN HIS ERA



####I CAN ALSO NOTICE THAT SUPERSTAR PLAYERS ARE HAVING MORE USG% NOW ,THAT MEANS BALL GETS IN SUPERASTARS HANDS A LOT MORE
####AND THEY DO CONTRIBUTE WITH THEIR GREAT TS%
####I CAN ALSO NOTICE THAT PLAYERS IN THE OLD ERA HAVE WAY MORE DWS,MORE PLAYERS WITH A LOT DWS AND BIGGEST DWS NUMBERS
####SO DEFENSE WAS A FACTOR BACK THEN AND NOWDAYS NOT SO MUCH,MAYBE THATS WHY MJ HAS A LOWER DWS
###NOW SOME MORE DIFFERENCES ON THE WAY THE GAME IS BEING PLAYED
####DIFFERENCES ON THE WAY GAME IS BEING PLAYED

from matplotlib.colors import ListedColormap
cmap = ListedColormap(['y',  'orange','r'])

cc=['USG%','PTS','AST','TRB','2P','3P','WS','PER','STL','BLK','TS%','OWS','DWS']
data=fin[cc]
data1=finback[cc]
correlations=data.corr()
correlations1=data1.corr()
fig=plt.figure(figsize=(25,12))
ax=fig.add_subplot(121)
cax=ax.matshow(correlations,vmin=0,vmax=1,cmap=cmap)
fig.colorbar(cax)
ticks=np.arange(0,13,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(cc)
ax.set_yticklabels(cc)

ax1=fig.add_subplot(122)
cax1=ax1.matshow(correlations1,vmin=0,vmax=1,cmap=cmap)
fig.colorbar(cax1)
ticks=np.arange(0,13,1)
ax1.set_xticks(ticks)
ax1.set_yticks(ticks)
ax1.set_xticklabels(cc)
ax1.set_yticklabels(cc)

plt.show()
####BY SEEING AT THE COLORMAPS WE CAN NOTICE THAT THE USG% IS SO MUCH MORE IMPORTANT IN THE NEW ERA THAN IN THE OLD ERA
####THAT MEANS TO BE A GREAT A PLAYER YOU HAVE TO GET THE BALL FAR MORE THAN YOU USED TO


####ALSO 3PS ARE ALSO A DIFFERENCE,IN TOTAL POINTS SO IN EVERYTHING(TAKING INTO CONSIDERATION THAT POINTS ARE THE MOST IMPORTANT)


####BLOCKS ALSO HELP A LOT THE PER OF THE PLAYERS
####SUPRISING STEALS ARE BOOSTING 3POINTERS,MAYBE BECAUSE WE HAVE MORE 3&D PLAYERS(ROLE PLAYERS THAT PLAY GREAT D,STRECH TH FLOOR
####AND HIT THE OPEN THREES)
####THE ASSUMPTION IS CORRECT BECAUSE DWS ALSO DEPENDS MORE ON THE 3P,AND NOT SO MUCH ON TS%(CUZ SHOOTERS HAVE LESS FG%)


####TS% IS NOT SO MEANINGFULL,MAYBE BECAUSE IT WAS A BIGMAN-ERA AND BIGMAN HAD TS%(AS THE ABOVE GRAPH SAYS)

####OWS DEPENDS LESS ON THE ASSISTS AND MORE ON THE USG%,SO THE MORE U HAVE THE BALL THE MORE OWS U HAVE

####BLOCKS HAVE MORE IMPACT ON THE GAME,SINCE GUARDS ARE MORE ATHLETIC AND HELP MORE ON DEFENSE
def calculateLeaguePace(i,t,y,a1,a2,a3,a4,a5,a6,a7):   
    a=df[df[y]==i] ### I AM TAKING THE MEAN OF EVERY STATS SINCE 2014
    a=a.groupby([y,t]).agg({a1:np.sum,a2:np.sum,a3:np.sum,a4:np.sum,a5:np.sum,a6:np.sum,a7:np.sum}).reset_index()
    a=a.groupby(t).agg({a1:np.mean,a2:np.mean,a3:np.mean,a4:np.mean,a5:np.mean,a6:np.mean,a7:np.mean}).reset_index()

    b=pd.DataFrame(index=a[t],columns=a[t],data=None)
    a=a.set_index(t)
    asd=[]
    for i in b.index:
        asdf=0
        for j in b.columns:
            b.loc[i][j]=((a.loc[i][a1]+a.loc[j][a1]+0.44*a.loc[i][a2]+0.44*a.loc[j][a2]-a.loc[i][a3]-
                              a.loc[j][a3]+a.loc[i][a5]+a.loc[j][a5])/82)/2
            if i==j:
                b.loc[i][j]=0
            asdf=asdf+b.loc[i][j]
        asd.append(round(asdf/30,2))
    b['MeanPoss']=asd
    team_pos=b
    a=a.reset_index()
    c=pd.DataFrame(index=a[t],columns=a[t],data=None)
    asd=[]
    for i in c.index:
        asdf=0
        for j in c.columns:
            c.loc[i][j]=48*((team_pos.loc[i]['MeanPoss']+team_pos.loc[j]['MeanPoss'])/(2*(48)))
            if i==j:
                c.loc[i][j]=0
            asdf=asdf+c.loc[i][j]
        asd.append(round(asdf/30,2))
    c['MeanPace']=asd
    team_pace=c
    team_pace=team_pace.drop('TOT')
    lg_pace=np.mean(team_pace['MeanPace'])
    return(lg_pace)

qq=[]
year=[]
for i in range(1988,1999):
    s=calculateLeaguePace(i,'Tm','Year','FGA','FTA','ORB','DRB','TOV','FG','G',)
    qq.append(s)
    year.append(i)
for i in range(2014,2017):
    s=calculateLeaguePace(i,'Tm','Year','FGA','FTA','ORB','DRB','TOV','FG','G',)
    qq.append(s)
    year.append(i)



####ILL NOW CHECK THE PACE OF THE GAME AS WELL AS THE FG,3P SHOOTS AND THE % OF THE SHOTS THAT ARE THREES.

predict=df[(((df['Year']>1988)&(df['Year']<1999))|(df['Year']>2014))]

predict=predict.groupby(['Year','Tm']).agg({'FGA':np.sum,'3PA':np.sum,'3P%':np.mean})
predict['3P/FGA']=predict['3PA']/predict['FGA']
predict=predict.sort_values('3PA',ascending=False)
predict=predict.reset_index()

predict=predict.set_index('Tm')
q=predict.groupby('Year').agg({'FGA':np.mean,'3PA':np.mean,'3P/FGA':np.mean})
q['3PA']=round(q['3PA'],2)
q['FGA']=round(q['FGA'],2)
q=q.reset_index()

plt.figure(figsize=(20,10))
plt.subplot(4,1,1)
ax=plt.plot(q['Year'],q['FGA'],'-o',c='r',label='Average Field Goals Attempts per Year')
plt.legend()

plt.subplot(4,1,2)
ax1=plt.plot(q['Year'],q['3PA'],'-o',c='b',label='Average 3pts Attempts per Year')
plt.legend()

plt.subplot(4,1,3)
ax3=plt.plot(q['Year'],q['3P/FGA'],'-o',c='y',label='Average % of Field goal being a Three per Year')
plt.legend()


plt.subplot(4,1,4)
plt.plot(year,qq,'-o',c='y',label='Average Pace per Season')
plt.legend()
plt.yticks()
  

plt.show()
####SO THE GAME IS A LOT FASTER(BY LOOKING AT THE PACE)WE HAVE A LOT MORE POSSESSIONS
####WE HAVE MORE 3POINT ATTEMPTS BUT THE SAME AMOUNT OF SHOOTS,SO WE CONCLUDE THAT TEAMS TEND TO SHOOT MORE 3S
####SINCE MJS PER AND POINTS  ARE NEARLY THE SAME WE SEE HE HAS DEVELOPED A GOOD 3POINT SHOOT,THAT MEANS HE ADDAPTED IN THE NEW ERA.
####WE SEE HE HAS SLIGHTLY MORE WS SO HIS BETTER 3POINT SHOOT IS HELPING HIS TEAM WIN.
####ALSO BECAUSE OF THE GUARDS GETTING A LOT BETTER AND SHOOTING MORE 3S HIS DWS ARE GETTING LOWER.