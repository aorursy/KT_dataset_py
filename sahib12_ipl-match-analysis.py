# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns# data visualization

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
delivery_data=pd.read_csv('/kaggle/input/ipl/deliveries.csv') # Bowl by Bowl data



match_data=pd.read_csv('/kaggle/input/ipl/matches.csv')# IPL(Indian premier League) Match Data from 2008 and 2016.
# print(match_data.shape)

match_data.head()

# match_data.loc[match_data.season==2017]
biggest_win=match_data['win_by_runs'].max()

teams_biggest_win=[]



f=match_data.loc[match_data['win_by_runs']==biggest_win]



teams_biggest_win.append((str(f['season'].tolist()[0])))



teams_biggest_win.append((str(f['venue'].tolist()[0])))



teams_biggest_win.append(str(f['team1'].tolist()[0]))



teams_biggest_win.append(str(f['team2'].tolist()[0]))





teams_biggest_win.append(str(f['winner'].tolist()[0]))





tie_matches_no=match_data.loc[match_data['result']=='tie']



f=match_data.loc[match_data['dl_applied']==1]

season_number=f['season'].value_counts()

dl_stadium=f['venue'].value_counts()
sns.set_style("darkgrid")# to make background with grid

ls=match_data['venue'].value_counts().sort_values(ascending=False)

ls=ls[:7]

plt.figure(figsize=(20,6))

Most_Played =sns.barplot(ls.index, ls.values, alpha=0.8)



plt.title('Most Played venue')

plt.ylabel('Count', fontsize=12)

plt.xlabel('Name of the stadiums', fontsize=15)

Most_Played.set_xticklabels(rotation=30,labels=ls.index,fontsize=10)

plt.show()

man_of_match=match_data['player_of_match'].value_counts()

man_of_match=man_of_match[:10]

sns.set_style("darkgrid")

plt.figure(figsize=(20,6))

man_of_matches=sns.barplot(man_of_match.index, man_of_match.values, alpha=0.8,palette='winter')

plt.title('Most Player Of The Match')

plt.ylabel('Count', fontsize=12)

plt.xlabel('player_of_match', fontsize=12)

man_of_matches.set_xticklabels(rotation=30,labels=man_of_match.index,fontsize=15)

plt.show()
plt.figure(figsize=(20,6))

season=sns.countplot(x='season',data=match_data)

plt.xlabel('Season',fontsize=20)
ump=pd.concat([match_data['umpire1'],match_data['umpire2']]).value_counts().sort_values(ascending=False)

ump=ump[:7]

plt.figure(figsize=(10,5))

Most_umpired =sns.barplot(x=ump.index, y=ump.values, alpha=0.9)



plt.title('Favorite umpire')

plt.ylabel('Count', fontsize=12)

plt.xlabel('Name of the Umpire', fontsize=15)

Most_umpired.set_xticklabels(rotation=50,labels=ump.index,fontsize=20)

plt.show()





Delhi_stadium=match_data.loc[(match_data['venue']=='Feroz Shah Kotla') ]

Delhi_stadium_win_by_runs=Delhi_stadium[Delhi_stadium['win_by_runs']>0]# As it is  win by runs this depicts Team batting First Has Won the match

slices=[len(Delhi_stadium_win_by_runs),len(Delhi_stadium)-len(Delhi_stadium_win_by_runs)]

labels=['Batting first','Batting Second']

plt.pie(slices,labels=labels,startangle=90,shadow=1,explode=(0,0.4),autopct='%1.2f%%',colors=['#bf00ff','#66CDAA'])

plt.show()
Kolkata_stadium=match_data.loc[(match_data['venue']=='Eden Gardens') ]

Kolkata_stadium_win_by_runs=Kolkata_stadium[Kolkata_stadium['win_by_runs']>0]# As it is  win by runs this depicts Team batting First Has Won the match

slices=[len(Kolkata_stadium_win_by_runs),len(Kolkata_stadium)-len(Kolkata_stadium_win_by_runs)]

labels=['Batting first','Batting Second']

plt.pie(slices,labels=labels,startangle=90,shadow=1,explode=(0,0.4),autopct='%1.2f%%',colors=['#00bfff','#00ff00'])

plt.show()
Mumbai_stadium=match_data.loc[(match_data['venue']=='Wankhede Stadium') ]

Mumbai_stadium_win_by_runs=Mumbai_stadium[Mumbai_stadium['win_by_runs']>0]# As it is  win by runs this depicts Team batting First Has Won the match

slices=[len(Mumbai_stadium_win_by_runs),len(Mumbai_stadium)-len(Mumbai_stadium_win_by_runs)]

labels=['Batting first','Batting Second']

plt.pie(slices,labels=labels,startangle=90,shadow=1,explode=(0,0.4),autopct='%1.2f%%',colors=['#00bfff','#00ff00'])

plt.show()
M_Chinnaswamy_Stadium=match_data.loc[(match_data['venue']=='M Chinnaswamy Stadium') ]

M_Chinnaswamy_Stadium_win_by_runs=M_Chinnaswamy_Stadium[M_Chinnaswamy_Stadium['win_by_runs']>0]# As it is  win by runs this depicts Team batting First Has Won the match

slices=[len(M_Chinnaswamy_Stadium_win_by_runs),len(M_Chinnaswamy_Stadium)-len(M_Chinnaswamy_Stadium_win_by_runs)]

labels=['Batting first','Batting Second']

plt.pie(slices,labels=labels,startangle=90,shadow=1,explode=(0,0.4),autopct='%1.2f%%',colors=['#99ff99','#ffcc99'])

plt.show()


Chennai_stadium=match_data.loc[(match_data['venue']=='MA Chidambaram Stadium, Chepauk') ]

Chennai_stadium_win_by_runs=Chennai_stadium[Chennai_stadium['win_by_runs']>0]# As it is  win by runs this depicts Team batting First Has Won the match

slices=[len(Chennai_stadium_win_by_runs),len(Chennai_stadium)-len(Chennai_stadium_win_by_runs)]

labels=['Batting first','Batting Second']

plt.pie(slices,labels=labels,startangle=90,shadow=1,explode=(0,0.4),autopct='%1.2f%%',colors=['#00ffbf','#00ff00'])

plt.show()
match_2017=match_data[match_data['season']==2017]

df=match_2017[match_2017['toss_winner']==match_2017['winner']]

slices=[len(df),(59-len(df))]# because Toal 59 matches were played in 2017

labels=['yes','no']

plt.pie(slices,labels=labels,startangle=90,shadow=True,explode=(0,0.05),autopct='%1.2f%%',colors=['#99ff99','#ffcc99'])

plt.show()
dfs=match_data[(match_data['toss_decision']=='field') & (match_data['season']==2017) ]

labels=['Bat','Field']

slices=[59-len(dfs),len(dfs)]

plt.pie(slices,labels=labels,startangle=90,shadow=1,explode=(0.2,0.4),autopct='%1.2f%%',colors=['#ff9999','#66b3ff'])

plt.show()


dfd=match_2017[(match_2017['toss_decision']=='field') & (match_2017['toss_winner']==match_2017['winner'])]

labels=['Fielding First Side Won','Fielding Second Side Won']

slices=[len(dfd),59-len(dfd)]

plt.pie(slices,labels=labels,startangle=90,shadow=1,explode=(0.2,0.4),autopct='%1.2f%%',colors=['#ff9999','#66b3ff'])

plt.show()
def comparator(team1):

    teams=list(match_data.team1.unique())# you can take team2 here also 

    teams.remove(team1)

    opponents=teams.copy()

    mt1=match_data[((match_data['team1']==team1)|(match_data['team2']==team1))]

    for i in opponents:

        mask = (((mt1['team1']==i)|(mt1['team2']==i)))&((mt1['team1']==team1)|(mt1['team2']==team1))# each time comparing each opponent team and the team we are looking for

#         print(mask)

        mt2 = mt1.loc[mask, 'winner'].value_counts().to_frame().T# to_frame to convert to DataFrame and T is used to Transpose

        print(mt2)

comparator('Mumbai Indians')
print(delivery_data.shape)

print(delivery_data.columns)
delivery_data.batting_team.unique()
delivery_data.head()


match_data.loc[match_data.season==2017].shape# this shows 59 matches were played in season 2017



most_maidens=delivery_data.groupby(['match_id','inning','over'])



# most_maidens.first()

list_of_most_maidens=[]

for match in range(1,60): # to iterate over each match total 59 matches

    for inning in range(1,3):# to iterate over each innings there are 2 innings in a match

        for over in range(1,21):# to iterate over maximum 20 overs in an innings

            

            try:# try block beacuse not every inning or every match is perfectly divided into 2 innings of 20 overs each in some matches due to rain or some other reason overswere reduced 

                #or even innings was dismissed

                if ((most_maidens.get_group((match,inning,over))['wide_runs'].sum()>0) + 

                    (most_maidens.get_group((match,inning,over))['noball_runs'].sum()>0)==most_maidens.get_group((match,inning,over))['total_runs'].sum()):

                    bowler=list(most_maidens.get_group((match,inning,over))['bowler'].unique())[0]

                    list_of_most_maidens.append(bowler)

                    

                    

                else:

                    pass

                

            except:

                continue





from statistics import mode 

def most_common(List): 

    return(mode(List)) 

    

print(str(most_common(list_of_most_maidens))+ " bowled most " + str(list_of_most_maidens.count(most_common(list_of_most_maidens))) + " maiden overs in 2017")                      
