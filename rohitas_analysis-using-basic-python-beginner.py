import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from pylab import savefig

%matplotlib inline
def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,

                '%d' % int(height),

                ha='center', va='bottom')
BALL_BY_BALL = "D:\EC\EC\IPL\Ball_by_Ball.csv"

MATCHES = "D:\EC\EC\IPL\Match.csv"

PLAYER_MATCH = "D:\EC\EC\IPL\Player_Match.csv"

PLAYER = "D:\EC\EC\IPL\Player.csv"

SEASON = "D:\EC\EC\IPL\Season.csv"

TEAMS = "D:\EC\EC\IPL\Team.csv"
deliveries = pd.read_csv(BALL_BY_BALL);

matches = pd.read_csv(MATCHES);

player_match = pd.read_csv(PLAYER_MATCH);

player = pd.read_csv(PLAYER);

season = pd.read_csv(SEASON);

teams = pd.read_csv(TEAMS);
print("Deliveries Dataset")

print(deliveries.columns.values)

print('_'*80)

print("Matches Dataset")

print(matches.columns.values)

print('_'*80)

print("Player Match Dataset")

print(player_match.columns.values)

print('_'*80)

print("Player Dataset")

print(player.columns.values)

print('_'*80)

print("Season Dataset")

print(season.columns.values)

print('_'*80)

print("Teams Dataset")

print(teams.columns.values)

print('_'*80)


print("Deliveries Dataset")

print(deliveries.head())

print('_'*80)

print("Matches Dataset")

print(matches.head())

print('_'*80)

print("Player Match Dataset")

print(player_match.head())

print('_'*80)

print("Player Dataset")

print(player.head())

print('_'*80)

print("Season Dataset")

print(season.head())

print('_'*80)

print("Teams Dataset")

print(teams.head())

print('_'*80)
print("Deliveries Dataset")

print(deliveries.info())

print('_'*80)

print("Matches Dataset")

print(matches.info())

print('_'*80)

print("Player Match Dataset")

print(player_match.info())

print('_'*80)

print("Player Dataset")

print(player.info())

print('_'*80)

print("Season Dataset")

print(season.info())

print('_'*80)

print("Teams Dataset")

print(teams.info())

print('_'*80)
deliveries["Team_Batting"] = pd.merge(deliveries, teams[["Team_Id","Team_Short_Code"]], how='left', left_on='Team_Batting_Id',right_on='Team_Id')["Team_Short_Code"]

deliveries["Team_Bowling"] = pd.merge(deliveries, teams[["Team_Id","Team_Short_Code"]], how='left', left_on='Team_Bowling_Id',right_on='Team_Id')["Team_Short_Code"]

deliveries["Striker"] = pd.merge(deliveries, player[["Player_Id","Player_Name"]], how='left', left_on='Striker_Id',right_on='Player_Id')["Player_Name"]

deliveries["Non_Striker"] = pd.merge(deliveries, player[["Player_Id","Player_Name"]], how='left', left_on='Non_Striker_Id',right_on='Player_Id')["Player_Name"]

deliveries["Bowler"] = pd.merge(deliveries, player[["Player_Id","Player_Name"]], how='left', left_on='Bowler_Id',right_on='Player_Id')["Player_Name"]

deliveries["Bowler"] = pd.merge(deliveries, player[["Player_Id","Player_Name"]], how='left', left_on='Bowler_Id',right_on='Player_Id')["Player_Name"]
deliveries.drop(["Team_Batting_Id","Team_Bowling_Id","Player_dissimal_Id","Fielder_Id","Bowler_Id","Non_Striker_Id","Striker_Id"], axis=1, inplace=True)

print(deliveries.head())
print(deliveries.head())
matches["Team_Name"] = pd.merge(matches, teams[["Team_Id","Team_Short_Code"]], how='left', left_on='Team_Name_Id',right_on='Team_Id')["Team_Short_Code"]

matches["Opponent_Team"] = pd.merge(matches, teams[["Team_Id","Team_Short_Code"]], how='left', left_on='Opponent_Team_Id',right_on='Team_Id')["Team_Short_Code"]

matches["Toss_Winner"] = pd.merge(matches, teams[["Team_Id","Team_Short_Code"]], how='left', left_on='Toss_Winner_Id',right_on='Team_Id')["Team_Short_Code"]

matches["Match_Winner"] = pd.merge(matches, teams[["Team_Id","Team_Short_Code"]], how='left', left_on='Match_Winner_Id',right_on='Team_Id')["Team_Short_Code"]

matches["Man_Of_The_Match"] = pd.merge(matches, player[["Player_Id","Player_Name"]], how='left', left_on='Man_Of_The_Match_Id',right_on='Player_Id')["Player_Name"]

matches.drop(["Team_Name_Id","Opponent_Team_Id","Toss_Winner_Id","Match_Winner_Id","Man_Of_The_Match_Id","Second_Umpire_Id","First_Umpire_Id"], axis=1, inplace=True)
print(matches.head())
matches["type"]="pre-qualifier"



for year in range(1,3):

    final_match_index = matches[matches['Season_Id']==year][-1:].index.values[0]

    matches = matches.set_value(final_match_index, "type", "final")

    matches = matches.set_value(final_match_index-1, "type", "semifinal-2")

    matches = matches.set_value(final_match_index-2, "type", "semifinal-1")

    



for year in range(3,4):

    final_match_index = matches[matches['Season_Id']==year][-1:].index.values[0]

    matches = matches.set_value(final_match_index, "type", "final")

    matches = matches.set_value(final_match_index-1, "type", "3rd-place")

    matches = matches.set_value(final_match_index-2, "type", "semifinal-2")

    matches = matches.set_value(final_match_index-3, "type", "semifinal-1")

    



for year in range(4,10):

    final_match_index = matches[matches['Season_Id']==year][-1:].index.values[0]

    matches = matches.set_value(final_match_index, "type", "final")

    matches = matches.set_value(final_match_index-1, "type", "qualifier-2")

    matches = matches.set_value(final_match_index-2, "type", "eliminator")

    matches = matches.set_value(final_match_index-3, "type", "qualifier-1")  
matches.groupby(["type"])["Match_Id"].count()

print((matches['Man_Of_The_Match'].value_counts()).idxmax(),' : has most man of the match awards')

print(((matches['Match_Winner']).value_counts()).idxmax(),': has the highest number of match wins')
k=sns.countplot(x='Season_Id', data=matches)

plt.show()

k1=k.get_figure()

k1.savefig("matsea1.png")
plt.figure(figsize=(12,6))

k=sns.countplot(y='Venue_Name', data=matches)

plt.yticks(rotation='horizontal')

plt.show()

k1=k.get_figure()

k1.savefig("stadiums.png")
plt.figure(figsize=(12,6))

k=sns.countplot(x='Match_Winner', data=matches)

plt.xticks(rotation='vertical')

plt.show()

k1=k.get_figure()

k1.savefig("matsea.png")

a=((matches['IS_Superover'].sum()))

b=((matches['IS_Result'].sum()))

c=((matches['Is_DuckWorthLewis'].sum()))

print("Superover Matches:",a)

print("Result Matches:",b)

print("DuckWorthLewis Matches:",c)

labels = 'SuperOver','Results','DuckWorthLewis'

sizes = [a,b,c]

colors = ['lightblue','gold','lightgreen']

explode = (0,0.1,0)

plt.title('NATURE OF MATCHES')

plt.legend(labels,loc=3)



plt.pie(sizes,colors=colors,labels=labels,

autopct='%1.1f%%', shadow=True, startangle=140)







plt.figure(figsize=(12,6))

k=sns.countplot(x='Win_Type', data=matches)

plt.xticks(rotation='vertical')

plt.show()

k1=k.get_figure()

k1.savefig("matsea.png")
plt.figure(figsize=(12,6))

k=sns.countplot(x='Toss_Decision', data=matches)

plt.xticks(rotation='vertical')

plt.show()

k1=k.get_figure()

k1.savefig("matsea.png")

#Toss Analysis

k=((matches['Toss_Winner']==matches['Match_Winner'])).sum()

print("Toss_Won",k) 

s=((matches['Toss_Winner']!=matches['Match_Winner'])).sum()

print("Toss_Lost",s)

import matplotlib.pyplot as plt

labels = 'Toss_Won','Toss_lost'

sizes = [k,s]

colors = ['gold','lightskyblue']

explode = (0.1, 0)

plt.title('WIN PERCENTAGE')



pie=plt.pie(sizes,labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=140)





plt.axis('equal')









#player analysis

p=((deliveries['Striker'].value_counts())).idxmax()

p1=((deliveries['Striker'].value_counts())).idxmin()

k=((deliveries['Bowler'].value_counts())).idxmax()

k1=((deliveries['Bowler'].value_counts())).idxmin()

m=(deliveries['Striker']==p).sum()

m1=(deliveries['Striker']==p1).sum()

s=(deliveries['Bowler']==k).sum()

s1=(deliveries['Bowler']==k1).sum()

print("Most Number of balls faced-->",p,":",m)

print("Least Number of balls faced-->",p1,":",m1)

print("Most Number of balls bowled-->",k,":",s)

print("Least Number of balls bowled-->",k1,":",s1)

l=deliveries['Bowler']=="AC Gilchrist"

dk=deliveries[l]

print(dk.Striker)

l1=deliveries['Striker']=="DP Vijaykumar"

dk1=deliveries[l1]

print(dk1.Bowler)
#Extra Types

plt.figure(figsize=(12,6))

sns.countplot(x='Extra_Type', data=deliveries)

plt.xticks(rotation='vertical')

plt.show()
k=pd.read_csv("D:\EC\EC\IPL\Player.csv")

plt.figure(figsize=(12,6))

l=sns.countplot(x='Country', data=k,palette="Set1")

plt.xticks(rotation=340)

plt.show()

p=k['Country']=="Netherlands"

s=k[p]

s.head()

k1=l.get_figure()

k1.savefig("playcount.png")



k=pd.read_csv("E:\kct\IPL\Player.csv")

plt.figure(figsize=(12,6))

sns.countplot(x='Batting_Hand', data=k,palette="Set2")

plt.xticks(rotation=30)

plt.show()

k=pd.read_csv("E:\kct\IPL\Player.csv")

plt.figure(figsize=(12,6))

sns.countplot(x='Bowling_Skill', data=k,palette="deep")

plt.xticks(rotation=100)

plt.show()
#Types of wickets

a=(deliveries['Dissimal_Type']=="caught").sum()

b=(deliveries['Dissimal_Type']=="run out").sum()

c=(deliveries['Dissimal_Type']=="bowled").sum()

d=(deliveries['Dissimal_Type']=="stumped").sum()

explode = (0.8,0,0,0)

print("caught:",a,"||run out:",b,"||bowled:",c,"||stumped:",d)

labels='caught','run out','bowled','stumped'

sizes=[a,b,c,d]

colors=['lightblue','lightgreen','pink','gold']

plt.pie(sizes,labels=labels, colors=colors,autopct='%1.1f%%',shadow=True, startangle=140)
sns.boxplot(data=player)
#Team Analysis

l=teams['Team_Short_Code'].tolist()

g=input("Enter team name:")

for x in l:

    if x==g:

        t1=deliveries.Team_Bowling==g

        df=deliveries[t1]

        a1=df['Striker'].value_counts().idxmax()

        print("Most balls faced against",g,":",a1)

        t4=deliveries.Team_Batting==g

        df4=deliveries[t4]

        a4=df4['Bowler'].value_counts().idxmax()

        print("Most balls bowled against",g,":",a4)

        t3=deliveries.Team_Batting==g

        df2=deliveries[t3]

        a3=df2['Striker'].value_counts().idxmax()

        print("Most balls faced for",g,":",a3)

        t2=deliveries.Team_Bowling==g

        df1=deliveries[t2] 

        a2=df1['Bowler'].value_counts().idxmax()

        print("Most balls bowled for",g,":",a2)

       



        



#Venue Analysis

o=matches['Venue_Name'].unique()

m=o.tolist()

keys=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]

i=0

dic={}

for key in keys:

    dic[key]=m[i]

    i+=1

print(dic)

p=int(input("Enter the Stadium Id"))

k=matches['Venue_Name']==dic[p]

k1=matches[k]

k2=k1.Toss_Decision=="field"

k3=k1[k2]

c=(k3['Toss_Winner']==k3['Match_Winner']).sum()

k4=k1.Toss_Decision=="bat"

k5=k1[k4]

v=(k5['Toss_Winner']==k5['Match_Winner']).sum()

print("\nAt",dic[p],":\n")

print('Field_First:',c,'||','Bat_First:',v,"\n")

labels='Field_First','Bat_First'

sizes=[c,v]

plt.pie(sizes,labels=labels,colors=colors,

autopct='%1.1f%%', shadow=True, startangle=140)

if(c==v):

    print('same win percentage')

elif(c>v):

    print('You should Field first!')

else:

    print('You should Bat first!')

#H2H Analysis

print(teams['Team_Short_Code'].tolist())

x=input("Enter Team1 name:")

y=input("Enter Team2 name :")

t1=matches['Team_Name']==x

t2=matches[t1]

o1=t2['Opponent_Team']==y

o2=t2[o1]

l1=o2['Match_Winner']==x

l2=o2["Match_Winner"]==y

s1=l1.sum()

s2=l2.sum()

#-------------------

t11=matches['Team_Name']==y

t22=matches[t11]

o11=t22['Opponent_Team']==x

o22=t22[o11]

l11=o22['Match_Winner']==x

l22=o22["Match_Winner"]==y

s11=l11.sum()

s22=l22.sum()

sum1=s1+s11

sum2=s2+s22

print(x,":",sum1)

print(y,":",sum2)

if(sum1>sum2):

    print(x,"has more wins!")

elif((sum1==0 and sum2==0)):

    print("They have not faced each other")

elif(sum1==sum2):

    print("They have equal number of wins")

else:

    print(y,"has more wins!")

labels=x,y

sizes=[sum1,sum2]

colors=['blue','yellow']

plt.pie(sizes,labels=labels, colors=colors,

autopct='%1.1f%%',shadow=True, startangle=140)

#Basic Information

Matches=matches['Match_Id'].count()

print("No.of matches played:",Matches)

Players=player['Player_Id'].count()

print("No. of Players played:",Players)

Balls=deliveries['Ball_Id'].count()

print("No of balls bowled:",Balls)

deliveries['Extra_Runs'].replace(' ', 0,inplace=True)

convert_dict = {'Extra_Runs': int }

deliveries= deliveries.astype(convert_dict)

Extra=deliveries['Extra_Runs'].sum()

print("No.of Extras conceded:",Extra)

k=matches['Venue_Name'].unique()

k1=k.tolist()

No_of_venues=len(k1)

print("No of venues hosted:",No_of_venues)

