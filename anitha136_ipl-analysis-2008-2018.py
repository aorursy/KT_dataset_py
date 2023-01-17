
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns               # Provides a high level interface for drawing attractive and informative statistical graphics
%matplotlib inline
sns.set()
from subprocess import check_output

import warnings                                            # Ignore warning related to pandas_profiling
warnings.filterwarnings('ignore') 


def annot_plot(ax,w,h):                                    # function to add data to plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for p in ax.patches:
        ax.annotate('{0:.1f}'.format(p.get_height()), (p.get_x()+w, p.get_height()+h))
import os
print(os.listdir("../input"))

data_path = "../input/"
matches_data = pd.read_csv(data_path+"matches.csv")
matches_data.head(5)
matches_data.shape                                                    # This will print the number of rows and comlumns of the Data Frame
matches_data.columns                                            # This will print the names of all columns.
matches_data.info()                                                   # This will give Index, Datatype and Memory information
matches_data.describe()
matches_data.isnull().sum()
matches_data.drop('umpire3', axis=1, inplace=True)  
matches_data.columns.unique()
#Replacing Rising Pune Supergiant with Rising Pune Supergiants
matches_data.replace( 'Rising Pune Supergiant', 'Rising Pune Supergiants',inplace = True)
matches_data.head(2)
matches_data['city'].fillna( matches_data['venue'].apply(lambda x: x[:5]),inplace = True)
matches_data[matches_data['city']== 'Dubai']
matches_data[matches_data['winner'].isnull()]
matches_data.replace( 'Bengaluru', 'Bangalore',inplace = True)
matches_data['city'].unique()

matches_data.columns
# display the seasons
matches_data['season'].unique()  
# display the team names in IPL
matches_data['team1'].unique() 
#No. of matches held each season

fig = plt.figure()
ax = fig.add_subplot(111)
ax=matches_data.groupby("season")["id"].count().plot(kind="line",title="Matches per season", marker='d',color=['blue'],figsize=(10,3)) 
plt.ylabel("No. of Matches")



max_times_winner = matches_data.groupby('season')['winner'].value_counts()

count=1
fig = plt.figure()

groups=max_times_winner.groupby('season')
for year,group in groups:
    ax = fig.add_subplot(4,3,count)
    ax.set_title(year)
    ax=group[year].plot(kind="bar",figsize=(10,15),width=0.8)
    count=count+1
    plt.xlabel('')
    plt.yticks([])
    plt.ylabel('Matches Won')
    
    totals = []
    for i in ax.patches:
        totals.append(i.get_height())
    total = sum(totals)
    for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
        ax.text(i.get_x()+0.2, i.get_height()-1.5,s= i.get_height(),color="white",fontweight='bold')
    
    
plt.tight_layout()
plt.show()


matches_played = matches_data['team1'].value_counts()+ matches_data['team2'].value_counts()
matches_played

matches_won = matches_data.groupby('winner').count()
matches_won["id"]
matches_won = matches_data.groupby('winner').count()
matches_won

total_matches = matches_data['team1'].value_counts()+ matches_data['team2'].value_counts()
total_matches

matches_won['Total matches']=total_matches
matches_won[["Total matches","result"]].sort_values(by=["Total matches"],ascending=False).plot.bar(stacked=True,figsize=(7,3))
match_succes_rate = (matches_won["id"]/total_matches)*100
#print(match_succes_rate)

data = match_succes_rate.sort_values(ascending = False)
plt.figure(figsize=(7,3))
ax = sns.barplot(x = data.index, y = data, palette="Set2");
plt.ylabel('Succes rate of each team')
plt.xticks(rotation=80)
annot_plot(ax,0.08,1)

season_winner = matches_data.groupby('season')['season','winner'].tail(1)
season_winner.sort_values(by="season",ascending=True)
season_winner.groupby('winner').count().plot.bar(figsize=(5,3))
maximum_runs = matches_data.sort_values('win_by_runs', ascending = False)[:5].head(5)
maximum_runs[['season','winner','win_by_runs']]
min_runs = matches_data[matches_data['win_by_runs'] == 1]
min_runs['winner'].value_counts()
plt.figure(figsize=(8,5))

sns.swarmplot(y='win_by_runs',x='winner',data=matches_data)
plt.tight_layout()
plt.xticks(rotation=90)
plt.title('Best Defending Teams')
plt.show()
max_wickets=matches_data[matches_data['win_by_wickets']==10]
max_wickets['winner'].value_counts()
matches_data[['season','winner','win_by_wickets']][matches_data['win_by_wickets'] ==1]
plt.figure(figsize=(8,5))
sns.swarmplot(y='win_by_wickets',x='winner',data=matches_data)
plt.xticks(rotation=80)
plt.title('Best Chasing Team')
plt.show()
plt.figure(figsize=(5,3))

ax =matches_data['player_of_match'].value_counts()[:10].plot.bar()
plt.title('Top 10 high performing Players')
annot_plot(ax,0.08,1)
toss_won = matches_data['toss_winner'].value_counts()
toss_win_rate = (toss_won/matches_played)*100
data = toss_win_rate.sort_values(ascending = False)
plt.figure(figsize=(5,3))
ax = sns.barplot(x = data.index, y = data, palette="Set2");
plt.ylabel('Toss win rate of each team')
plt.xticks(rotation=90)
annot_plot(ax,0.08,1)
plt.title('Toss winning success rate of each team')
toss=matches_data['toss_decision'].value_counts()
labels=np.array(toss.index)
sizes = toss.values
colors = ['gold', 'lightskyblue']

# Plot
plt.figure(figsize=(5,3))
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True,startangle=90)

plt.title('Toss Decision of all the matches')
plt.axis('equal')
plt.show()
plt.figure(figsize=(8,3))
ax =sns.countplot(x='season',hue='toss_decision',data=matches_data,palette="Set2")
plt.ylabel('Toss Decision frequency')
plt.title('Toss Decision across seasons')
annot_plot(ax,0.08,1)
plt.figure(figsize=(5,3))
sns.boxplot(x="toss_winner", y="season", hue = 'toss_decision', data=matches_data)
plt.xlabel('toss_winner ')
plt.xticks(rotation=90)
plt.title('Toss decision by each team')
tosswin_win = matches_data['id'][matches_data['toss_winner'] == matches_data['winner']].count()
total_matches=matches_data['id'].count()
Success_rate = ((matches_data[matches_data['toss_winner'] == matches_data['winner']].count())/(total_matches))*100

print("Number of matches in which Toss winner is the game winner is :",tosswin_win, "out of",total_matches," ie.,", Success_rate["id"],"%" )

tosswin_winner = matches_data['toss_decision'][matches_data['toss_winner'] == matches_data['winner']].value_counts()
labels=np.array(tosswin_winner.index)
sizes = tosswin_winner.values
colors = ['gold', 'lightskyblue']

plt.figure(figsize=(5,3))
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True,startangle=90)

plt.title('Toss decision of toss winner to win the game')
plt.axis('equal')
plt.show()
plt.figure(figsize=(5,3))

ax=matches_data['city'].value_counts()[:10].plot.bar()
plt.title('Top 10 Cities to hold match')
plt.xticks(rotation=70)
annot_plot(ax,0.08,1)
a=matches_data.groupby(['winner','city']).size().reset_index(name='win_counts')
a=a.sort_values("win_counts",ascending=False)
a.groupby("winner").head(1)
#top 10 venue to hold max number of matches
plt.figure(figsize=(5,3))
venue=matches_data.groupby('venue')["id"].count()
ax =venue.sort_values(ascending=False).head(10).plot.bar(figsize=(5,3))
plt.title('Top 10 venue to hold matches')
plt.xticks(rotation=90)
annot_plot(ax,0.08,1)
venue_suit_for =matches_data[matches_data['toss_winner'] == matches_data['winner']]
sns.countplot(x='venue',hue='toss_decision',data=venue_suit_for)
plt.xlabel('Venue ')
plt.title('Venue is Best Suited for')
plt.xticks(rotation=90)

result=matches_data['result'].value_counts().tolist()
names='Normal - '+str(result[0]), 'Tie - '+str(result[1]), 'No result - '+str(result[2]), 

fig, ax = plt.subplots(figsize=(3.5,3.5))  
# Create a pieplot
explode = (0, 0.01, 0.01)
ax1,text=ax.pie(result,labeldistance=2,explode=explode,radius=0.1, startangle=180,colors=['skyblue','green','red'])
#plt.show()
ax.axis('equal')
ax.set_title("Match Results") 

# add a circle at the center
my_circle=plt.Circle( (0,0), 0.07, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.legend(ax1, names,  bbox_to_anchor=(.9,.8), loc=2)
plt.tight_layout()
plt.show()
toss=matches_data['dl_applied'].value_counts()
labels=np.array(toss.index)
sizes = toss.values
colors = ['gold', 'lightskyblue']

# Plot
plt.figure(figsize=(5,3))
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True,startangle=90)

plt.title('Dl applied of all the matches')
plt.axis('equal')
plt.show()
plt.figure(figsize=(5,3))
ax=sns.countplot(matches_data.city[matches_data.dl_applied==1])
plt.ylabel('Dl applied count City wise')
plt.xticks(rotation=80)
