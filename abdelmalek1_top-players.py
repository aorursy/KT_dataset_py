#importing helping libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import os
from IPython.display import display
%matplotlib inline
#Loading the dataset
events = pd.read_csv('../input/events.csv')
data_ginf = pd.read_csv('../input/ginf.csv')
df=data_ginf.merge(events,how='left')
df.head()
df.info()
new=dict()
with open('../input/dictionary.txt','r') as f:
    data=f.read()
data=data.split('\n\n\n')
for i in range(len(data)):
    if data[i]:
        variable_name = data[i].split('\n')[0]
        values = data[i].split('\n')[1:]
        new[variable_name]={int(s.split('\t')[0]):s.split('\t')[1] for s in values}
        print(data[i])
for name in new:
    df[name]=df[name].map(new[name])
bundesliga=df[df['country']=='germany']
ligue1=df[df['country']=='france']
laliga=df[df['country']=='spain']
premiereleague=df[df['country']=='england']
seriea=df[df['country']=='italy']
print('Bundes Liga data shape:',bundesliga.shape)
print('Ligue 1 data shape:',ligue1.shape)
print('La Liga data shape:',laliga.shape)
print('Premiere League:',premiereleague.shape)
print('Serie A data shape:',seriea.shape)
def top_scorers(data):
    goals=data.loc[data['is_goal']==1&(data['situation']!='Own goal')] #excluding own goals(we are looking for strikers who can score in the opponent's net)
    goals=goals.groupby('player')['is_goal'].sum().reset_index().rename(columns={'is_goal':'G'}).sort_values(by='G',ascending=False)
    goals=goals[['player','G']].set_index('player')
    return goals
player_tp=top_scorers(df)
print('G : Goals')
player_tp[:20]
def pointgraph(data,x,s):
    plt.figure(figsize=(12,8))
#     data=top_scorers(data)
    ax=sns.scatterplot(x=data[x],y=data.index,s=700,alpha=.7)
    for i,j in enumerate(data[x]):
        ax.text(j-2,i-0.2,int(j),color='white')
    plt.title(s)
    plt.tight_layout()
    plt.show()
pointgraph(player_tp[:20],'G','Top 20 Scorers')
def GPM(data):
    x=data[data['situation']!='Own goal']
    y=x.groupby(['id_odsp','player'])['is_goal'].sum().reset_index().rename(columns={'id_odsp':'Matches','is_goal':'G'})
    xy=y.groupby('player').agg({'Matches':'count','G':"sum"})
    xy['GPM']=xy['G']/xy['Matches']
    xy=xy[xy['Matches']>xy['Matches'].max()*0.25]
#     print(xy['Matches'].max()*0.25)
    xy.sort_values(by='GPM',ascending=False)
    return xy.sort_values(by='GPM',ascending=False)

print('G : Goals')
print('GPM : Goals Per Match')
player_gpm=GPM(df)
player_gpm[:20]
def twin_barplot(data1,x1,y1,s1,data2,x2,y2,s2):
    plt.figure(figsize=(20,10))

    plt.subplot(121)
    ax=sns.barplot(x=x1,y=y1,data=data1)
    for i,j in enumerate(data1[x1][:20]):
        ax.text(0.5,i,j,weight='bold')
    plt.title(s1)
    plt.ylabel("")
    plt.subplot(122)
    plt.subplots_adjust(wspace=.5)
    ax=sns.barplot(x=x2,y=y2,data=data2)
    for i,j in enumerate(player_gpm[x2][:20]):
        ax.text(0.01,i,j,weight='bold')
    plt.title(s2)
twin_barplot(player_tp[:20],'G',player_tp.index[:20],'Goals',player_gpm[:20],'GPM',player_gpm.index[:20],'Goals Per Match')
def NPGPM(data):
    x=data[(data['situation']!='Own goal')&(data['location']!='Penalty spot')]
    y=x.groupby(['id_odsp','player'])['is_goal'].sum().reset_index().rename(columns={'id_odsp':'Matches','is_goal':'NPG'})
#     print(y[y['player']=='sergio aguero'])
    xy=y.groupby('player').agg({'Matches':'count','NPG':"sum"})
    xy['NPGPM']=xy['NPG']/xy['Matches']
    xy=xy[xy['Matches']>31]
#     print(xy['Matches'].max()*0.25)
    
    return xy.sort_values(by='NPGPM',ascending=False)
print('NPG : Non-Penalty Goals')
print('NPGPM : Non-Penalty Goals Per Match')
player_npg=NPGPM(df)
player_npg[:20]
def double_bargraph(data,s):
#     print(data)
    ax=data.plot(kind='barh',figsize=(20,20),edgecolor='k',linewidth=1)
    plt.title(s)
    plt.legend(loc='best',prop={'size':40})
    for i,j in enumerate(data.iloc[:,1]):
        ax.text(0.5,i,j,weight='bold')
    for i,j in enumerate(data.iloc[:,0]):
        ax.text(0.5,i-0.2,j,weight='bold',color='white')
xx=pd.concat([player_tp,player_npg],axis=1).fillna(0)
double_bargraph(xx.sort_values(by='G',ascending=False)[['G','NPG']][:20],'Goals Vs. Non-Penalty Goals')

def ExpG(data):
    x=data[(data['location']!='Penalty spot')&(data['event_type2']!='Own goal')&(data['event_type']=='Attempt')]
    y=x.groupby(['player','id_odsp']).agg({'is_goal':'sum','event_type':'count'}).reset_index()
    y['total']=y['is_goal']/y['event_type']
    y=y.groupby('player').agg({'is_goal':'sum','total':'mean','event_type':'sum','id_odsp':'count'})
    y['total2']=y['event_type']/y['id_odsp']
    y['GPM']=y['is_goal']/y['id_odsp']
    y=y[y['is_goal']>18]
    y.columns=['NPG','Avg GPA','Attempts','Matches','APM','GPM']
    return y
print('NPG : Non-Penalty Goals')
print('Avg GPA : Average Goal Per Attempt')
print('APM : Attempt Per Match')
print('GPM : Goal Per Match')

ExpG(df).sort_values(by='Attempts',ascending=False)[:20]

def bar(data,x,y,s ):
    fig=plt.figure(figsize=(15,15))
    ax=sns.barplot(x=x,y=y,data=data)
    plt.title(s)
    for i,j in enumerate(data[x]):
        ax.text(0.01,i,j,weight='bold')
player_expg=ExpG(df).sort_values(by='Avg GPA',ascending=False)[:20]
bar(player_expg,'Avg GPA',player_expg.index,'Average Goals Per Match')        


def GPL(data,colors,labels):
    plt.figure(figsize=(15,12))
    plt.xticks(list(range(10)))
    plt.xlabel('Goals Per Match')
#     plt.legend(loc='best',prop={'size':40})
    for d,c,s in zip(data,colors,labels):
        d=d.groupby('id_odsp')['is_goal'].sum()
        sns.kdeplot(d,shade=True,color=c,label=s)
        plt.axvline(d.mean(),linestyle='dashed',color=c,label=(s+' Mean'))
#FOR the honor of League winners this year, i changed the colors to be the color of the winner teams shirts
GPL([bundesliga,laliga,ligue1,seriea,premiereleague],['r','w','g','k','b'],['BundesLiga','LaLiga','Ligue1','SerieA','PremiereLeague'])
# top scorer in PL
def pointgraph(data,x,s):
    plt.figure(figsize=(12,8))
#     data=top_scorers(data)
    ax=sns.scatterplot(x=data[x],y=data.index,s=700,alpha=.7)
    for i,j in enumerate(data[x]):
        ax.text(j-.5,i-0.2,int(j),color='white') #we will overide the original function just to update the x position of text 
    plt.title(s)
    plt.tight_layout()
    plt.show()
def league_repr(data,n):
    tp=top_scorers(data)
    gpm=GPM(data)
    npgpm=NPGPM(data)
    xx=pd.concat([tp,npgpm],axis=1).fillna(0)
    expg=ExpG(data)
    
    pointgraph(tp[:n],'G','Top Scorers')
    twin_barplot(tp[:n],'G',tp.index[:n],'Goals',gpm[:n],'GPM',gpm.index[:n],'Goals Per Match')
    double_bargraph(xx[['G','NPG']].sort_values(by='G',ascending=False)[:n],'Goals Vs.Non-Penalty Goals')
    bar(expg.sort_values(by='Avg GPA',ascending=False)[:n],'Avg GPA',expg.sort_values(by='Avg GPA',ascending=False).index[:n],'Average Goals Per Attempt')
    print('sorted by number of attempts')
    display(expg.sort_values(by='Attempts',ascending=False)[:n])
# pl_tp=top_scorers(premiereleague)
# pl_gpm=GPM(premiereleague)
# pl_xx=pd.concat([top_scorers(premiereleague),NPGPM(premiereleague)],axis=1).fillna(0)
# pl_expg=ExpG(premiereleague)
# pointgraph(pl_tp[:20],'G','Top Scorers in Premiere League')
# twin_barplot(pl_tp[:20],'G',pl_tp.index[:20],'Goals',pl_gpm[:20],'GPM',pl_gpm.index[:20],'Goals Per Match')
# double_bargraph(pl_xx[['G','NPG']].sort_values(by='G',ascending=False)[:20],'Non-Penalty')
# bar(pl_expg.sort_values(by='Avg GPA',ascending=False)[:20],'Avg GPA',pl_expg.sort_values(by='Avg GPA',ascending=False).index[:20],'Average Goals Per Attempt')
# pl_expg.sort_values(by='Attempts',ascending=False).head(20)

league_repr(premiereleague,20)

league_repr(ligue1,20)
league_repr(seriea,20)
league_repr(laliga,20)
league_repr(bundesliga,20)
