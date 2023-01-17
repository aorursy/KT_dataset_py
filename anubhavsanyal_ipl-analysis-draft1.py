# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
players_data=pd.read_excel('/kaggle/input/ipl-data-set/Players.xlsx')
players_data
batting_hand=players_data.iloc[:,2]

lh=0
rh=0
for i in range(0,len(batting_hand)):
    if(batting_hand[i]=='Right_Hand'):
        rh=rh+1
    else:
        lh=lh+1
print(lh/len(batting_hand))
print(rh/len(batting_hand))
left=[1,2]
data_hand=[rh,lh]
labels=['Right Hand Batsman','Left Hand Batsman']
plt.bar(left,data_hand,tick_label=labels,width=0.3,color=['blue','green'])
plt.xlabel('Dominant Hand of the Batsman')
plt.ylabel('Number of Players')
plt.title('Dominant Hand vs Number of Players')
plt.plot()
players_data['Country'].fillna('No country',inplace=True)
players_data
player_country=players_data.iloc[:,4]
player_country.unique()
india=0
england=0
south_africa=0
australia=0
bangladesh=0
srilanka=0
westindies=0
newzealand=0
pakistan=0
netherlands=0
zimbabwea=0
for i in range(0,len(player_country)):
    if(player_country[i]=='India'):
        india=india+1
    elif(player_country[i]=='England'):
        england=england+1
    elif(player_country[i]=='South Africa'):
        south_africa=south_africa+1
    elif(player_country[i]=='Australia'):
        australia=australia+1
    elif(player_country[i]=='Bangladesh'):
        bangladesh=bangladesh+1
    elif(player_country[i]=='Sri Lanka'):
        srilanka=srilanka+1
    elif(player_country[i]=='West Indies'):
        westindies=westindies+1
    elif(player_country[i]=='New Zealand'):
        newzealand=newzealand+1
    elif(player_country[i]=='Pakistan'):
        pakistan=pakistan+1
    elif(player_country[i]=='Netherlands'):
        netherlands=netherlands+1
    else:
        zimbabwea=zimbabwea+1
left=[1,2,3,4,5,6,7,8,9,10,11]
data_country=[india,england,south_africa,australia,bangladesh,srilanka,westindies,newzealand,pakistan,netherlands,zimbabwea]
data_label=['India','England','South Africa','Australia','Bangladesh','Sri Lanka','West Indies','Newzealnd','Pakistan','Netherlands','Zimbabwea']
plt.bar(left,data_country,tick_label=data_label,width=0.8)
plt.xticks(rotation=90)
plt.xlabel('Countries')
plt.ylabel('Number of Players')
plt.title('Countries vs Number of Players')
plt.plot()
plt.show()
import seaborn as sns
plt.figure(figsize=(7,7))
plt.xticks(rotation=90,fontsize=14)
plt.xlabel('Countries')
plt.ylabel('Number of Players')
plt.title('Countries vs Number of Players')
sns.countplot(player_country)
plt.figure(figsize=(10,10))
plt.xticks(rotation=90,fontsize=16)
sns.countplot(players_data['Bowling_Skill'])
plt.title('Bowling Skills of Different Players')
match_data=pd.read_csv('/kaggle/input/ipl-data-set/matches.csv')
match_data

toss=match_data.iloc[:,7]
decision_field=0
decision_bat=0
for i in range(0,len(toss)):
    if toss[i]=='bat':
        decision_bat=decision_bat+1
    else:
        decision_field=decision_field+1
decision_bat_per=decision_bat/len(toss)
decision_field_per=(decision_field)/len(toss)
print(decision_bat_per)
print(decision_field_per)
plt.figure(figsize=(3,3))
plt.xticks(rotation=90,fontsize=14)
sns.countplot(toss)
plt.title('Toss vs Choice')
toss_win=match_data.iloc[:,6]
match_win=match_data.iloc[:,10]
count_match_winners=0
for i in range(0,len(toss_win)):
    if(toss_win[i]==match_win[i]):
        count_match_winners=count_match_winners+1
count_match_winners_per=(count_match_winners)/len(toss_win)
count_match_winners_per

plt.figure(figsize=(8,8))
plt.xticks(rotation=90,fontsize=14)
sns.countplot(match_data['city'])
plt.title('Most common venues of IPL')
strike_rate=pd.read_csv('/kaggle/input/ipl-data-set/most_runs_average_strikerate.csv')
strike_rate
strike_rate=strike_rate.iloc[0:10,:]
strike_rate
plt.figure(figsize=(5,5))
plt.xticks(rotation=90,fontsize=14)
plt.scatter(strike_rate['batsman'],strike_rate['strikerate'],color='red')
plt.plot()
plt.show()
plt.figure(figsize=(5,5))
plt.xticks(rotation=90,fontsize=14)
plt.bar(strike_rate['batsman'],strike_rate['average'],width=0.6)
plt.plot()
plt.show()
delivery_data=pd.read_csv('/kaggle/input/ipl-data-set/deliveries.csv')














