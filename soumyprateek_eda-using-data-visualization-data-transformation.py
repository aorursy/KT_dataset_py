

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


matches=pd.read_csv('../input/indian-premier-league-csv-dataset/Match.csv')
teams=pd.read_csv('../input/indian-premier-league-csv-dataset/Team.csv')
player=pd.read_csv('../input/indian-premier-league-csv-dataset/Player.csv')
results=matches[['Season_Id','Team_Name_Id','Opponent_Team_Id','IS_Result','Match_Winner_Id']]


results['null']=results['Match_Winner_Id'].notnull()
results=results[results['null']==True]
results.drop(['null','IS_Result'],inplace=True,axis=1)


#results.count()=577,574 3 null values 
results['Match_Winner_Id']=results['Match_Winner_Id'].astype(int)
results
teams.set_index('Team_Id',inplace=True)

teams
results['Team_Name_Id']=results['Team_Name_Id'].map(teams['Team_Short_Code'])
results['Opponent_Team_Id']=results['Opponent_Team_Id'].map(teams['Team_Short_Code'])
results['Match_Winner_Id']=results['Match_Winner_Id'].map(teams['Team_Short_Code'])
results.columns=['Season','Team','Opponent','Winner']
results
'''grouped=results.groupby('Season')
for key, item in grouped:

    print(grouped.get_group(key))
    print("\n")'''
kkr1=results[results['Team']=='KKR']
kkr2=results[results['Opponent']=='KKR']

kkr=pd.concat([kkr1,kkr2])

csk1=results[results['Team']=='CSK']
csk2=results[results['Opponent']=='CSK']

csk=pd.concat([csk1,csk2])

dd1=results[results['Team']=='DD']
dd2=results[results['Opponent']=='DD']

dd=pd.concat([kkr1,kkr2])

mi1=results[results['Team']=='MI']
mi2=results[results['Opponent']=='MI']

mi=pd.concat([kkr1,kkr2])

rcb1=results[results['Team']=='RCB']
rcb2=results[results['Opponent']=='RCB']

rcb=pd.concat([rcb1,rcb2])

rr1=results[results['Team']=='RR']
rr2=results[results['Opponent']=='RR']

rr=pd.concat([rr1,rr2])

kxip1=results[results['Team']=='KXIP']
kxip2=results[results['Opponent']=='KXIP']

kxip=pd.concat([kxip1,kxip2])

srh1=results[results['Team']=='SRH']
srh2=results[results['Opponent']=='SRH']

srh=pd.concat([srh1,srh2])




print(kkr,mi,csk,rr,dd,rcb,srh,kxip)
%matplotlib notebook
import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(12,6))
sns.countplot(x='Country', data=player)
plt.xticks(rotation='vertical')
plt.title(' number of players from each country ')
plt.xlabel('country')
plt.ylabel('number of players')
plt.show()

plt.figure(figsize=(12,6))
sns.countplot( x='Batting_Hand',data=player)
plt.xticks(rotation='vertical')
plt.title(' player info')
plt.xlabel('country')
plt.ylabel('number of players')
plt.show()

plt.figure(figsize=(12,6))
sns.countplot(x='Bowling_Skill', data=player)
plt.xticks(rotation='vertical')
plt.title(' player stats ')
plt.xlabel('skill')
plt.ylabel('number of players')
plt.show()

plt.figure(figsize=(12,6))
sns.countplot(x='Venue_Name', data=matches)
plt.xticks(rotation='vertical')
plt.title('venue statistics')
plt.xlabel('venue')
plt.ylabel('number of matches')
plt.show()
import matplotlib.gridspec as gs
plt.figure(figsize=(18,12))
gspec=gs.GridSpec(6,4)
p1=plt.subplot(gspec[0:1,0:])
sns.countplot(x='Country', data=player)
plt.xticks(rotation='vertical')
plt.title(' number of players from each country ')
plt.xlabel('country')
plt.ylabel('number of players')
plt.show()


p2=plt.subplot(gspec[2:3,0])

sns.countplot( x='Batting_Hand',data=player)

plt.title(' player batting style')
plt.xlabel('country')
plt.ylabel('number of players')
plt.show()


p3=plt.subplot(gspec[2:3,1:])
sns.countplot(x='Bowling_Skill', data=player)
plt.xticks(rotation=45)
plt.title(' player bowlin style')
plt.xlabel('skill')
plt.ylabel('number of players')
plt.show()



p4=plt.subplot(gspec[4:,0:])
sns.countplot(x='Venue_Name', data=matches)
plt.xticks(rotation='vertical')
plt.title('venue statistics')
plt.xlabel('venue')
plt.ylabel('number of matches')
plt.show()



plt.figure(figsize=(12,6))
sns.countplot(x='Winner', data=results)
plt.xticks(rotation='vertical')
plt.title(' total matches won by each team ')
plt.xlabel('Teams')
plt.ylabel('Matches won')
plt.show()
