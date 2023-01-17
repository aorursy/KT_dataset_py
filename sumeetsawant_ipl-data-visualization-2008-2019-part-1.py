import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
base_color=sns.color_palette()[8]
sns.set(style="whitegrid")
%matplotlib inline 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
matches=pd.read_csv('/kaggle/input/ipldata//matches.csv')
print("Matches shape{}".format(matches.shape))


matches.head()

#Checking null values 
matches.isnull().sum()
matches[matches.city.isnull()]
matches.city.fillna('Dubai ',inplace=True)
matches.replace('Rising Pune Supergiant','Rising Pune Supergiants',inplace=True)

matches.replace('Delhi Daredevils','Delhi Capitals',inplace=True)

#Modifying Data so as to plot second graph
team_1=matches.groupby(['team1']).count()
team_2=matches.groupby(['team2']).count()
total_matches=team_1['id']+team_2['id']
total_matches=pd.DataFrame(total_matches,columns=['id'])
total_matches.reset_index(inplace=True)

total_matches.replace('Delhi Daredevils','Delhi Capitals',inplace=True)
total_matches.replace('Rising Pune Supergiant','Rising Pune Supergiants',inplace=True)

plt.figure(figsize=(25,10))
plt.subplot(1,2,1)
splot=sns.countplot(data=matches,x='season',color=base_color)

plt.title('Matches Across Seasons',fontsize=30)
plt.ylabel('Count',fontsize=30)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Seasons',fontsize=20)

# add annotations
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), \
                   ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points',fontsize=20)
    
    
plt.subplot(1,2,2)

splot=sns.barplot(data=total_matches,x='team1',y='id',color=base_color,order=total_matches.sort_values(by=['id'],ascending=False)['team1'])

plt.title('Matches Played by each team',fontsize=30)
plt.ylabel('Count',fontsize=30)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Teams',fontsize=20)

# add annotations
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), \
                   ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points',fontsize=20)
plt.figure(figsize=(15,10))

splot=sns.countplot(data=matches,x='city',color=base_color,order=matches['city'].value_counts().index)

plt.title('Matches Played in each city',fontsize=30)
plt.ylabel('Count',fontsize=30)
plt.xlabel('City',fontsize=30)
plt.xticks(rotation=90,fontsize=20)

# add annotations
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), \
                   ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points',fontsize=20)
plt.figure(figsize=(20,8))
splot=sns.countplot(data=matches,x='venue',color=base_color,order=matches['venue'].value_counts().index)
plt.xticks(rotation=90,fontsize=20);
plt.title('Games at Each Venue',fontsize=15);
plt.ylabel('Count',fontsize=20);
plt.xlabel('Stadium Names',fontsize=20);

for p in splot.patches:
    splot.annotate(format((p.get_height()), ".0f"), (p.get_x() + p.get_width() / 2., p.get_height()), \
                   ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points',fontsize=10)
#Creating a columns for Toss Win percent in total match dataframe which we created above 

total_matches['percent_toss_wins']=(matches.groupby('toss_winner').count()['toss_decision'].values/total_matches['id'].values)*100


plt.figure(figsize=(15,10))
#base_color=sns.color_palette()[7]
splot=sns.barplot(data=total_matches,x='team1',y='percent_toss_wins',color=base_color,order=total_matches.sort_values(by=['percent_toss_wins'],ascending=False)['team1'])

plt.title('% Toss won by each team',fontsize=30)
plt.ylabel('Percent',fontsize=30)
plt.xlabel('Team',fontsize=30)
plt.xticks(rotation=90,fontsize=20)

# add annotations
for p in splot.patches:
    splot.annotate(format(p.get_height(), ".1f"), (p.get_x() + p.get_width() / 2., p.get_height()), \
                   ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points',fontsize=20)
plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
ticks=[0,50,100,150,200,250,300,350]
labels=[(x/matches.shape[0])*100 for x in ticks];
labels=['%.1f' % elem for elem in labels ]
splot=sns.countplot(data=matches,x='toss_decision',color=base_color)
plt.yticks(ticks,labels);
plt.xlabel('Toss Decision',fontsize=20);
plt.ylabel('% Percent',fontsize=20);
plt.xticks(fontsize=20);

for p in splot.patches:
    splot.annotate(format((p.get_height()/matches.shape[0])*100, ".1f"), (p.get_x() + p.get_width() / 2., p.get_height()), \
                   ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points',fontsize=20)
    
plt.subplot(1,2,2)
sns.countplot(x='season',hue='toss_decision',data=matches)
plt.xlabel('Toss Decision',fontsize=20);
plt.xticks(rotation=90,fontsize=20);
plt.ylabel('Count',fontsize=20);
plt.figure(figsize=(10,5))
sns.countplot(x='toss_winner',hue='toss_decision',data=matches)
plt.xlabel('Toss Decision',fontsize=20);
plt.xticks(rotation=90,fontsize=20);
plt.ylabel('Count',fontsize=20);
Team_list=['Mumbai Indians','Kings XI Punjab', 'Chennai Super Kings','Royal Challengers Bangalore','Kolkata Knight Riders','Delhi Capitals','Rajasthan Royals',                
            'Sunrisers Hyderabad' ,'Deccan Chargers', 'Pune Warriors', 'Rising Pune Supergiants','Gujarat Lions', 'Kochi Tuskers Kerala']


total_toss_wins=matches.groupby(['toss_winner']).count()                                             #counts for team winning the toss 
match_wins=matches[matches.winner==matches.toss_winner].groupby(['toss_winner','winner']).count()  # Counts for team winning the toss and winnng the match 
percent_wins={}

for Team in Team_list:
    percent_wins.update({Team : (match_wins.loc[Team]['id']/total_toss_wins.loc[Team]['id']).values[0]*100} )
    

toss_and_game_win=pd.DataFrame.from_dict(percent_wins,orient='index',columns=['percent']).reset_index()
plt.figure(figsize=(15,10))
#base_color=sns.color_palette()[7]
splot=sns.barplot(data=toss_and_game_win,x='index',y='percent',color=base_color,order=toss_and_game_win.sort_values(by=['percent'],ascending=False)['index'])

plt.title(' Toss and game won by each team Percent',fontsize=30)
plt.ylabel('Percent',fontsize=30)
plt.xlabel('Team',fontsize=30)
plt.xticks(rotation=90,fontsize=20)

# add annotations
for p in splot.patches:
    splot.annotate(format(p.get_height(), ".1f"), (p.get_x() + p.get_width() / 2., p.get_height()), \
                   ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points',fontsize=20)
# Dropping Null values for umpire 
matches.dropna(subset=['umpire1'],inplace=True,axis=0)
fig4, ax = plt.subplots(figsize=(12, 10))
ax.set_yticklabels(ax.get_yticklabels(), ha="right", fontsize=12, weight='bold');
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, weight='bold');
cbar_kws = {'orientation':"horizontal", 'pad':0.08, 'aspect':50};
top=10
sns.heatmap(data=matches.groupby(['umpire1','umpire2']).size().to_frame('count').sort_values(by='count',ascending=False).head(top),annot=True,cmap='RdPu',ax=ax,cbar=False);

## More umpire can be plotted but I am restricting it to top 10 . You can just change top parameter in the cell 
#Making a dataframe for plotting top 10 player of match award winners 
Player_of_match=pd.DataFrame(matches.groupby('player_of_match').count()['id'].sort_values(ascending=False).head(10),columns=['id']).reset_index()

plt.figure(figsize=(10,10));
ticks=[0,3,6,9,12,15,18,21]
#base_color=sns.color_palette()[5]
sns.barplot(y='id',x='player_of_match',data=Player_of_match,color=base_color);
plt.xticks(rotation=90);
plt.yticks(ticks,ticks);
plt.xticks(fontsize=15);
plt.title('Won Player of Match title',fontsize=20);
plt.ylabel('Count',fontsize=20);
plt.xlabel('Players',fontsize=20);
def head_to_head(df,team1,team2):
    """
    Function which computes the head to head winner between two teams 
    
    Input : Dataframe , str of team 1 and team 2 names 
    eg: headtohead(matches,'Mumbai Indians','Delhi Capitals')
    
    output : Prints a table for head to head winner
    
    
Team names can be selected among 

Mumbai Indians                 
Kings XI Punjab                 
Chennai Super Kings             
Royal Challengers Bangalore     
Kolkata Knight Riders       
Delhi Capitals                  
Rajasthan Royals                
Sunrisers Hyderabad             
Deccan Chargers                 
Pune Warriors                   
Rising Pune Supergiants         
Gujarat Lions                   
Kochi Tuskers Kerala             
    
    """
    
    
    wins=df[(((df['team1']==team1) | (df['team2']==team1)) & ((df['team1']==team2)|(df['team2']==team2)))]
    print(wins.groupby('winner').count()['id'])
                                                         
head_to_head(matches,'Mumbai Indians','Delhi Capitals')
head_to_head(matches,'Mumbai Indians','Kolkata Knight Riders')

plt.figure(figsize=(10,8))
g=sns.scatterplot(x='team1',y='team2',data=matches.sort_values(by='win_by_wickets',ascending=False,).head(100),hue='win_by_wickets', palette='tab10');
plt.xticks(rotation=90,fontsize=15);
plt.yticks(fontsize=15);
g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1);
fig=plt.figure(figsize=(20,10))
axes=plt.subplot(1,2,2)
g=sns.scatterplot(x='season',y='win_by_wickets',data=matches.sort_values(by='win_by_wickets',ascending=False,).head(100),hue='winner', palette='bright',legend=False);
plt.xticks(rotation=90,fontsize=15);
plt.yticks([10,9,8],fontsize=15);

plt.xlabel('Season',fontsize=15);
plt.ylabel('win_by_wickets',fontsize=15);
plt.title('Top 100 wins batting second vs season',fontsize=15);

plt.subplot(1,2,1)
g=sns.scatterplot(x='season',y='win_by_runs',data=matches.sort_values(by='win_by_runs',ascending=False,).head(100),hue='winner', palette='tab20');
plt.xticks(rotation=90,fontsize=15);
plt.yticks(fontsize=15);
plt.xlabel('Season',fontsize=15);
plt.ylabel('win_by_runs',fontsize=15);
plt.title('Top 100 wins batting fist vs season',fontsize=15);
g.legend(loc='center left', bbox_to_anchor=(2.25, 0.5), ncol=1);


