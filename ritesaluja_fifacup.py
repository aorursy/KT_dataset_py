# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from termcolor import colored

import warnings
warnings.filterwarnings("ignore")
# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/fifa-world-cup/WorldCups.csv")
df1 = pd.read_csv("../input/fifa-world-cup/WorldCupMatches.csv")
df2 = pd.read_csv("../input/fifa-world-cup/WorldCupPlayers.csv")
cup2018teams = pd.read_csv("../input/coparussiajogos/Cup.Russia.Teams.csv")
cup2018matches = pd.read_csv("../input/coparussiajogos/Cup.Russia.Matches.csv")
#Data Prep - Adding 2018 Worldcup data 
df["Attendance"]=df["Attendance"].str.replace('.','')
#list(df)
datatemp = {'Year':2018,'Country':'Russia','Winner':'France','Runners-Up':'Croatia','Third':'Belgium',\
            'Fourth':'England', 'GoalsScored':169,'QualifiedTeams':32,'MatchesPlayed':64,'Attendance':3031768} 
df = df.append(datatemp,ignore_index=True)
df["Winner"].replace(to_replace="Germany FR", value="Germany",inplace=True)
df["Runners-Up"].replace(to_replace="Germany FR", value="Germany",inplace=True)
df["Third"].replace(to_replace="Germany FR", value="Germany",inplace=True)
df["Fourth"].replace(to_replace="Germany FR", value="Germany",inplace=True)
#FiFa Worldcup Attendances by Year
sns.set_style("darkgrid")
att = df.groupby("Year")["Attendance"].sum().reset_index()
att["Year"] = att["Year"].astype(int)
att["Attendance"] = att["Attendance"].astype(int)
plt.figure(figsize=(12,7))

sns.barplot(att["Year"],att["Attendance"],linewidth=1,edgecolor="k"*len(att),palette="Blues_d")
plt.grid(True)
plt.title("Attendence by year",color='b')
plt.show()
#preparing winner data
Winnerdata = {}
listingYears = []
for i in df["Winner"]:
    listingYears.append(list(df[df["Winner"]==i]["Year"]))

j = 0
for k in df["Winner"]:
    if k not in Winnerdata.keys():
        Winnerdata[k] = listingYears[j]
    j = j + 1


#getting required Winner data onto list for plot
names = list(Winnerdata.keys())
values = list(Winnerdata.values())

#  ++updated with France 2018 worldcup 
fig_size = plt.rcParams["figure.figsize"]
# Set figure width to 12 and height to 9
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

for j in range(8):
    plt.bar(j, len(values[j]),tick_label=names[j],hatch="-")
    j = j +1
plt.xticks(range(0,8),names)
plt.title("Countries Winning FIFA WorldCup till 2018",color='blue')
plt.xlabel("Countries", color = 'green')
plt.ylabel("Wins",color = 'green')

x=0 
for i in  range(len(names)):
    y = 0.72
    for j in range(len(values[i])):
        plt.text(x, y, ""+str(values[i][j]), color='black', va='center', fontweight='bold',horizontalalignment='center')
        y = y + 0.77
    x = x+1


    
plt.show()
#All eyes on France
totalMatches = (df1[df1["Home Team Name"]=="France"]["Home Team Name"].value_counts() + df1[df1["Away Team Name"]=="France"]["Away Team Name"].value_counts())
GF = (df1[df1["Home Team Name"]=="France"]["Home Team Goals"].sum() + df1[df1["Away Team Name"]=="France"]["Away Team Goals"].sum())
GA = (df1[df1["Home Team Name"]=="France"]["Away Team Goals"].sum() + df1[df1["Away Team Name"]=="France"]["Home Team Goals"].sum())
print("In FiFa WorldCup to time \nFrance played                  %d Matches"%(totalMatches))
print("France Scored                  %d Goals"%(GF))
print("Goals conceded Against France  %d"%(GA))
print("On an Average France Scored    %.2f Goals"%(GF/totalMatches))
print("France hosted WorldCup twice and won once out of those two instances")


print("On 4th May, 1980 - Croatia leave Yugoslavia to become the Nation")
#All eyes on Croatia
totalMatches = (df1[df1["Home Team Name"]=="Croatia"]["Home Team Name"].value_counts() + df1[df1["Away Team Name"]=="Croatia"]["Away Team Name"].value_counts())
GF = (df1[df1["Home Team Name"]=="Croatia"]["Home Team Goals"].sum() + df1[df1["Away Team Name"]=="Croatia"]["Away Team Goals"].sum())
GA = (df1[df1["Home Team Name"]=="Croatia"]["Away Team Goals"].sum() + df1[df1["Away Team Name"]=="Croatia"]["Home Team Goals"].sum())
print("\nIn FiFa WorldCup to time \nCroatia played                 %d Matches"%(totalMatches))
print("Croatia Scored                 %d Goals"%(GF))
print("Goals Conceded Against Croatia %d"%(GA))
print("On an Average Croatia Scored   %.2f Goals"%(GF/totalMatches))
print("Croatia never got the opportunity to host the WorldCup. \nPoint to note in 1998 when France won the WorldCup Croatia was Third on rank!")
#Cleaning data
df1["Home Team Name"] = df1["Home Team Name"].str.replace('rn">United Arab Emirates',"United Arab Emirates")
df1["Home Team Name"] = df1["Home Team Name"].str.replace("C�te d'Ivoire","Côte d’Ivoire")
df1["Home Team Name"] = df1["Home Team Name"].str.replace('rn">Republic of Ireland',"Republic of Ireland")
df1["Home Team Name"] = df1["Home Team Name"].str.replace('rn">Bosnia and Herzegovina',"Bosnia and Herzegovina")
df1["Home Team Name"] = df1["Home Team Name"].str.replace('rn">Serbia and Montenegro',"Serbia and Montenegro")
df1["Home Team Name"] = df1["Home Team Name"].str.replace('rn">Trinidad and Tobago',"Trinidad and Tobago")
df1["Home Team Name"] = df1["Home Team Name"].str.replace("Soviet Union","Russia")
df1["Home Team Name"] = df1["Home Team Name"].str.replace("Germany FR","Germany")

df1["Away Team Name"] = df1["Away Team Name"].str.replace('rn">United Arab Emirates',"United Arab Emirates")
df1["Away Team Name"] = df1["Away Team Name"].str.replace("C�te d'Ivoire","Côte d’Ivoire")
df1["Away Team Name"] = df1["Away Team Name"].str.replace('rn">Republic of Ireland',"Republic of Ireland")
df1["Away Team Name"] = df1["Away Team Name"].str.replace('rn">Bosnia and Herzegovina',"Bosnia and Herzegovina")
df1["Away Team Name"] = df1["Away Team Name"].str.replace('rn">Serbia and Montenegro',"Serbia and Montenegro")
df1["Away Team Name"] = df1["Away Team Name"].str.replace('rn">Trinidad and Tobago',"Trinidad and Tobago")
df1["Away Team Name"] = df1["Away Team Name"].str.replace("Germany FR","Germany")
df1["Away Team Name"] = df1["Away Team Name"].str.replace("Soviet Union","Russia")
df1 = df1.dropna(how='all')
#Team Encounters
def who_won(x,y):
    wt = df1[df1["Away Team Name"]==y]
    wt = wt[wt["Home Team Name"]==x]
    wx = wy = d = j = 0 
    
    HTG = np.array(wt["Home Team Goals"].astype(int))
    ATG = np.array(wt["Away Team Goals"].astype(int))
    
    for i in wt["Home Team Name"]:
        if HTG[j] > ATG[j]:
            wx = wx + 1 
        elif HTG[j] < ATG[j]:
            wy = wy + 1
        else:
            d = d + 1
        j = j + 1
            
    return (wx,wy,d)

def plotingPie(WinX, WinY, D,x,y):
    plt.figure(figsize=(9,9))
    the_grid = GridSpec(1, 2)
    labels = 'Wins', 'Draws', 'Loss'
    fracs1 = [WinX, D, WinY]
    fracs2 = [WinY, D, WinX ]
    explode = (0, 0.05, 0)
    plt.subplot(the_grid[0, 0], aspect=1)
    plt.title("Matches Outcome for %s"%(x))
    plt.pie(fracs1, labels=labels, autopct='%1.0f%%', shadow=True,
                                           colors  = ["lawngreen","royalblue","tomato"],
                                           wedgeprops={"linewidth":2,"edgecolor":"white"})
    circ = plt.Circle((0,0),.7,color="white")
    plt.gca().add_artist(circ)
    
    plt.subplot(the_grid[0, 1], aspect=1)
    plt.title("Matches Outcome for %s"%(y))
    plt.pie(fracs2, explode=explode, labels=labels, autopct='%1.0f%%', shadow=True,
                                           colors  = ["lawngreen","royalblue","tomato"],
                                           wedgeprops={"linewidth":2,"edgecolor":"white"})
    circ = plt.Circle((0,0),.7,color="white")
    plt.gca().add_artist(circ)
    plt.show()
            
def teamEncounter(x,y):
    (wx1,wy1,d1) = who_won(x,y)
    (wy2,wx2,d2) = who_won(y,x)
    WinX = wx1 + wx2
    WinY = wy1 + wy2
    Draws = d1 + d2
    n = WinX + WinY + Draws
    print("Of %d Encounter(s) the two team had in WorldCup (till 2014)- \nTeam %s won %d Matches against %s"%(n,x,WinX,y))
    print("Team %s won %d Matches against %s"%(y,WinY,x))
    print("Matches drawn ",Draws) 
    plotingPie(WinX,WinY,Draws,x,y)
    
        
teamEncounter('France','Croatia')
teamEncounter('Belgium','England')
print(colored(cup2018teams.groupby("Team")["Goals"].sum().sort_values(ascending=False).head(10),'blue'))
#Data prep for MaxFinals
temp1 = df.groupby("Runners-Up")["Runners-Up"].count()
temp2 = df.groupby("Winner")["Winner"].count()
MaxFinals = list(temp2.index.values)
r = {}
MaxF = 0
Team =""
for i in MaxFinals:
    if i not in list(temp1.index.values):
        temp1[i] = 0
    r[i] = int(temp2[i]) + int(temp1[i])
    if MaxF < int(r[i]):
        MaxF = int(r[i])
        Team = i
        

print(colored("Out of All FiFa WorldCup Winners till 2018: ",'blue')+Team+\
              " has played Maximum Finals "+colored(MaxF,'red') + colored(" of 21",'red') )
    
HighAttTeam = {}
for i in range(64):
    if cup2018matches["Home Team"][i] not in HighAttTeam.keys():
        HighAttTeam[cup2018matches["Home Team"][i]] = cup2018matches["Attendance"][i]
    else:
        HighAttTeam[cup2018matches["Home Team"][i]] = HighAttTeam[cup2018matches["Home Team"][i]] + cup2018matches["Attendance"][i]
        
    if cup2018matches["Away Team"][i] not in HighAttTeam.keys():
        HighAttTeam[cup2018matches["Away Team"][i]] = cup2018matches["Attendance"][i] 
    else:
        HighAttTeam[cup2018matches["Away Team"][i]] = HighAttTeam[cup2018matches["Away Team"][i]] + cup2018matches["Attendance"][i]
import operator
sortedHighAttTeam = sorted(HighAttTeam.items(), key=operator.itemgetter(1), reverse = True)
print("Team whose Matches witness Maximum Attendance in FIFA WorldCup 2018:"+colored(sortedHighAttTeam[0],'green'))
foulplay = cup2018teams
foulplay.fillna(0,inplace=True)
foulplay.head(1)
#combination of Fouls, Red-Cards and Yellow-Cards
foulplay["FoulScore"] = 1*foulplay["Yellow Cards"] + 2*foulplay["Red Cards"] + 0.1*foulplay["Fouls Committed"]
foulplay=foulplay.sort_values("FoulScore",ascending=False).reset_index()
print(colored("Top 3 Teams with Foul Play in 2018",'blue'))
print("1.",foulplay["Team"][0])
print("2.",foulplay["Team"][1])
print("3.",foulplay["Team"][2])

