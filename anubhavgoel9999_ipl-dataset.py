# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
matches=pd.read_csv('../input/ipl-data-set/matches.csv')

players=pd.read_excel('../input/ipl-data-set/Players.xlsx')

team=pd.read_csv('../input/ipl-data-set/most_runs_average_strikerate.csv')
players.replace('Right_hand','Right_Hand',inplace=True)

matches.replace('Rising Pune Supergiant','Rising Pune Supergiants',inplace=True)

matches.replace('Delhi Daredevils','Delhi Capitals',inplace=True)

import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(players['Batting_Hand'])
plt.xticks(rotation=90)

sns.countplot(players['Country'])
matches.isnull().any()

matches.drop(['umpire1','umpire2','umpire3'],axis=1,inplace=True)

I=matches[matches['winner'].isnull()==True].index

matches.drop(I,inplace=True)

I2=matches[matches['city'].isnull()==True].index

matches.drop(I2,inplace=True)
matches['Season'].unique()
S_2008=matches[matches['Season']=='IPL-2008']

S_2009=matches[matches['Season']=='IPL-2009']

S_2010=matches[matches['Season']=='IPL-2010']

S_2011=matches[matches['Season']=='IPL-2011']

S_2012=matches[matches['Season']=='IPL-2012']

S_2013=matches[matches['Season']=='IPL-2013']

S_2014=matches[matches['Season']=='IPL-2014']

S_2015=matches[matches['Season']=='IPL-2015']

S_2016=matches[matches['Season']=='IPL-2016']

S_2017=matches[matches['Season']=='IPL-2017']

S_2018=matches[matches['Season']=='IPL-2018']

S_2019=matches[matches['Season']=='IPL-2019']
Finals=pd.concat([S_2008.loc[S_2008['id']==S_2008['id'].max()],S_2009.loc[S_2009['id']==S_2009['id'].max()],S_2010.loc[S_2010['id']==S_2010['id'].max()],S_2011.loc[S_2011['id']==S_2011['id'].max()],

S_2012.loc[S_2012['id']==S_2012['id'].max()],

S_2013.loc[S_2013['id']==S_2013['id'].max()],S_2014.loc[S_2014['id']==S_2014['id'].max()],S_2015.loc[S_2015['id']==S_2015['id'].max()],

S_2016.loc[S_2016['id']==S_2016['id'].max()],S_2017.loc[S_2017['id']==S_2017['id'].max()],S_2018.loc[S_2018['id']==S_2018['id'].max()],

S_2019.loc[S_2019['id']==S_2019['id'].max()]])

Finals
len(matches[matches['toss_winner']==matches['winner']])/len(matches) * 100
temp1=matches[matches['toss_winner']==matches['winner']]

temp2=matches.groupby('toss_winner')['winner'].count()

temp1=temp1.groupby('toss_winner')['winner'].count()

temp=temp1/temp2*100

plt.xticks(rotation=90)

plt.plot(temp)

temp
len(Finals[Finals['toss_winner']==Finals['winner']])/len(Finals) * 100
Del=pd.read_csv('../input/ipl-data-set/deliveries.csv')

Del
Stat=Del.groupby('batting_team')['total_runs'].sum()
print(len(matches[matches['team1']=='Chennai Super Kings']) + len(matches[matches['team2']=='Chennai Super Kings']))

print(len(matches[matches['team1']=='Mumbai Indians']) + len(matches[matches['team2']=='Mumbai Indians']))

print(len(matches[matches['team1']=='Kolkata Knight Riders']) + len(matches[matches['team2']=='Kolkata Knight Riders']))

print(len(matches[matches['team1']=='Rajasthan Royals']) + len(matches[matches['team2']=='Rajasthan Royals']))

print(len(matches[matches['team1']=='Royal Challengers Bangalore']) + len(matches[matches['team2']=='Royal Challengers Bangalore']))

print(len(matches[matches['team1']=='Kings XI Punjab']) + len(matches[matches['team2']=='Kings XI Punjab']))

print(len(matches[matches['team1']=='Sunrisers Hyderabad']) + len(matches[matches['team2']=='Sunrisers Hyderabad']))

print(len(matches[matches['team1']=='Delhi Capitals']) + len(matches[matches['team2']=='Delhi Capitals']))
print('chennai average score :',  Stat['Chennai Super Kings']/162)

print('Mumbai average score :',  Stat['Mumbai Indians']/184)

print('Kolkata average score :',  Stat['Kolkata Knight Riders']/177)

print('Rajasthan average score :',  Stat['Rajasthan Royals']/144)

print('Banglore average score :',  Stat['Royal Challengers Bangalore']/175)

print('punjab average score :',  Stat['Kings XI Punjab']/175)

print('Hyderabad average score :',  Stat['Sunrisers Hyderabad']/106)

print('Delhi average score :',  Stat['Delhi Daredevils']/157)
a=matches.groupby('player_of_match')['player_of_match'].count()

A=pd.DataFrame(a)

A.rename(columns={'player_of_match':'Number of MOM'},inplace=True)

A.sort_values("Number of MOM",inplace=True,ascending=False)

A=A.reset_index()

A=A.iloc[:5,:]

sns.barplot(A['player_of_match'],A['Number of MOM'])
plt.figure(figsize=[10,10])

plt.xticks(rotation=90)

sns.countplot('venue',data=matches)

plt.show()
Per=matches.pivot_table(index=['winner','Season'],aggfunc='size')

Per=pd.DataFrame(Per)

Per.rename(columns={0:'wins'},inplace=True)

Per.reset_index(inplace=True)

C=Per[Per['winner']=='Chennai Super Kings']

plt.xticks(rotation=90)

plt.title('Chennai Super Kings')

plt.scatter('Season','wins',data=C)

plt.plot('Season','wins',data=C)
R=Per[Per['winner']=='Royal Challengers Bangalore']

plt.xticks(rotation=90)

plt.title('Royal Challengers Bangalore')

plt.scatter('Season','wins',data=R)

plt.plot('Season','wins',data=R)
M=Per[Per['winner']=='Mumbai Indians']

plt.xticks(rotation=90)

plt.title('Mumbai Indians')

plt.scatter('Season','wins',data=M)

plt.plot('Season','wins',data=M)
K=Per[Per['winner']=='Kolkata Knight Riders']

plt.xticks(rotation=90)

plt.title('Kolkata Knight Riders')

plt.scatter('Season','wins',data=K)

plt.plot('Season','wins',data=K)
Kl=Per[Per['winner']=='Kings XI Punjab']

plt.xticks(rotation=90)

plt.title('Kings XI Punjab')

plt.scatter('Season','wins',data=Kl)

plt.plot('Season','wins',data=Kl)

R=Per[Per['winner']=='Rajasthan Royals']

plt.xticks(rotation=90)

plt.title('Rajasthan Royals')

plt.scatter('Season','wins',data=R)

plt.plot('Season','wins',data=R)



MS_Stat=Del[Del['batsman']=='MS Dhoni']

MS_Strike=pd.DataFrame(MS_Stat.groupby('over')['total_runs'].mean()).reset_index()

MS_Strike['total_runs']=MS_Strike['total_runs']*100

sns.scatterplot(MS_Strike['over'],MS_Strike['total_runs'])

plt.title('MS DHONI')

plt.ylabel('Strike Rate')
V_Stat=Del[Del['batsman']=='V Kohli']

V_Strike=pd.DataFrame(V_Stat.groupby('over')['total_runs'].mean()).reset_index()

V_Strike['total_runs']=V_Strike['total_runs']*100

sns.scatterplot(V_Strike['over'],V_Strike['total_runs'])

plt.ylabel('Strike Rate')

plt.title('VIRAT KOHLI')
R_Stat=Del[Del['batsman']=='SK Raina']

R_Strike=pd.DataFrame(R_Stat.groupby('over')['total_runs'].mean()).reset_index()

R_Strike['total_runs']=R_Strike['total_runs']*100

sns.scatterplot(R_Strike['over'],R_Strike['total_runs'])

plt.ylabel('Strike Rate')

plt.title('SURESH RAINA')