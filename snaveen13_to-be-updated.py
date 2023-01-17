# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/matches.csv")

data.describe()
data = data.drop('umpire3',axis=1)

data.describe()
data.isnull().sum()
data[pd.isnull(data['city'])]
data[data['result'] == 'no result']
data['city'] = data['city'].fillna("Dubai")

data = data.fillna('-')
data.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',

                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',

                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors']

                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW'],inplace=True)
teams = ['KKR', 'CSK','DD', 'RCB','RR', 'KXIP','DC', 'MI','PW', 'KTK','SRH', 'RPS','GL']

home_win = []

away_win = []

print("HOME MATCHES WIN PERCENTAGE")

print("========================")

for i in teams:

    home = (data['team1'] == i).sum()

    winners = ((data['winner'] == i) & (data['team1'] == i)).sum()

    home_win.append((winners / home)*100)

    print(i ,":%.1f"%((winners / home)*100)+'%')

    

print("")



print("AWAY MATCHES WIN PERCENTAGE")

print("========================")

for i in teams:

    away = (data['team2'] == i).sum()

    winners = ((data['winner'] == i) & (data['team2'] == i)).sum()

    away_win.append((winners / away)*100)

    print(i ,":%.1f"%((winners / away)*100)+'%')
# Pie chart shows home win percentage where CSK has highest of 12% and RPS has 0%.

plt.pie(home_win, labels=teams,autopct='%1.1f%%', shadow=True, startangle=140)

plt.title("Home win percentage by each team")

plt.show()
# Pie chart shows home win percentage where GL has highest of 11% and PW has 3.2%.

plt.pie(away_win, labels=teams,autopct='%1.1f%%', shadow=True, startangle=140)

plt.title("Away win percentage by each team")

plt.show()
sns.barplot(y = data['city'].value_counts().index, x = data['city'].value_counts())

plt.title("Total number of matches played at each venue")

plt.xlabel("number of games played")

plt.ylabel("City")

plt.show()
data['winner'].value_counts().values[:13]
xx = sns.barplot(x = data['winner'].value_counts().index[:13], y = data['winner'].value_counts()[:13])

## Add text at top of bars

text(x = xx, y = data['winner'].value_counts().values[:13], 

     label = data['winner'].value_counts().values[:13], pos = 3, cex = 0.8, col = "red")

## Add x-axis labels 

##axis(1, at=xx, labels=dat$fac, tick=FALSE, las=2, line=-0.5, cex.axis=0.5)



plt.title("Total number of Wins")

plt.xlabel('Number of wins')

plt.ylabel('Teams')

plt.show()
sns.countplot(x = 'toss_decision',data = data)

plt.title("Toss Decision")

plt.show()
sns.countplot(x = 'dl_applied',data = data)

plt.title("DL Method")

plt.show()
sns.barplot(x = data['toss_winner'].value_counts().index, y = data['toss_winner'].value_counts())

plt.title("Toss winner")

sns.factorplot(x="toss_winner", col="toss_decision",data=data, kind="count", size=6, aspect=.8)

plt.show()



## Gujarat Lions and Royal challengers Bangalore are likely to field whenever they win the toss.
toss_win_win = data[(data['winner']==data['toss_winner'])]

sns.countplot(x = 'toss_winner',data = toss_win_win)

plt.title("Toss and game winner")

sns.factorplot(x="toss_winner", col="toss_decision",data=toss_win_win, kind="count", size=7, aspect=.8)

plt.show()



## Gujarat, Kochi and Pune have won games only when they choose fielding.
sns.countplot(x = 'toss_decision',data = toss_win_win)

plt.title("Overall toss and match winners in terms of decision")

plt.show()



## Fielding first is dominated in general
sns.countplot(x = 'toss_decision',data = toss_win_win, hue = 'season')

plt.title("Toss and match winners in terms of decision year wise")

plt.show()



##2016 has the least wins with team who chose to bat first
for i in teams:

    top_players = data[data['winner'] == i].player_of_match.value_counts()

    sns.barplot(x = top_players[:3].index,y = top_players[:3])

    plt.title(i + " top players")

    plt.show()
years = [2008,2009,2010,2011,2012,2013,2014,2015,2016]

man_of_series = []

print("Most of the matches each year")

print("===========================")

for i in years:

    man_of_series = data[data['season'] == i].player_of_match.value_counts().index[0]

    print(i, ":", man_of_series)
for i in years:

    champ = data[data['season'] == i].winner.values[len(data[data['season'] == i])-1]

    print(i,"Champions :", champ)
ump1 = data.umpire1.unique()

ump2 = data.umpire2.unique()

umps = list(ump1)+list(ump2)

umps = list(set(umps))
ump_2008 = [(len(data[(data['season'] == 2008) & ((data['umpire1'] == i) | (data['umpire2'] == i))])) for i in umps]

ump_2009 = [(len(data[(data['season'] == 2009) & ((data['umpire1'] == i) | (data['umpire2'] == i))])) for i in umps]

ump_2010 = [(len(data[(data['season'] == 2010) & ((data['umpire1'] == i) | (data['umpire2'] == i))])) for i in umps]

ump_2011 = [(len(data[(data['season'] == 2011) & ((data['umpire1'] == i) | (data['umpire2'] == i))])) for i in umps]

ump_2012 = [(len(data[(data['season'] == 2012) & ((data['umpire1'] == i) | (data['umpire2'] == i))])) for i in umps]

ump_2013 = [(len(data[(data['season'] == 2013) & ((data['umpire1'] == i) | (data['umpire2'] == i))])) for i in umps]

ump_2014 = [(len(data[(data['season'] == 2014) & ((data['umpire1'] == i) | (data['umpire2'] == i))])) for i in umps]

ump_2015 = [(len(data[(data['season'] == 2015) & ((data['umpire1'] == i) | (data['umpire2'] == i))])) for i in umps]

ump_2016 = [(len(data[(data['season'] == 2016) & ((data['umpire1'] == i) | (data['umpire2'] == i))])) for i in umps]
umps_years = [[umps, ump_2008, ump_2009]]

umpire_match = pd.DataFrame({'Umpire Names':umps, 'Yr2008':ump_2008, 'Yr2009':ump_2009, 'Yr2010':ump_2010

                            , 'Yr2011':ump_2011, 'Yr2012':ump_2012, 'Yr2013':ump_2013

                            , 'Yr2014':ump_2014, 'Yr2015':ump_2015, 'Yr2016':ump_2016})

umpire_match.head()
umpire_match['Total_matches'] = umpire_match.iloc[:,1:].sum(axis = 1)

umpire_match.sort_values(by = "Total_matches", ascending= False, inplace= True)

umpire_match.head()
sns.barplot(x = umpire_match['Total_matches'][:10], y = umpire_match['Umpire Names'][:10])

plt.xlabel("Number of games")

plt.title("Top 10 umpires in IPL")

plt.show()
year_column = ['Yr2008', 'Yr2009','Yr2010','Yr2011','Yr2012','Yr2013','Yr2014','Yr2015','Yr2016',]

for i in year_column:

    top_2 = umpire_match.sort_values(by = i,ascending = False)['Umpire Names'][:2].values

    print("")

    print("Top 2 umpires in", i)

    print("========================")

    for elem in top_2:

        print(elem)