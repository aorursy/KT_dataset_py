import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
file = '../input/ipl-matches-data-from-2011-to-2019/IPL 2.csv'

df = pd.read_csv(file)

df.head()
%matplotlib inline

plt.rcParams['figure.figsize']=30,15
a = df['season'].value_counts()

sns.countplot(x = 'season',data = df,

              palette = 'icefire_r',

              order = df['season'].value_counts()[:10].index)

plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.xticks(fontsize =25,rotation = 65)

plt.yticks(fontsize = 25)

plt.xlabel("Year",fontsize = 30)

plt.ylabel("Number of matches",fontsize = 30)

plt.title("Number of matches per year",fontsize = 50)

print(a)
d = df['team1'].value_counts()

e = df['team2'].value_counts()

f = e + d

f= f[['Chennai Super Kings','Delhi Capitals','Kings XI Punjab','Kolkata Knight Riders',

       'Mumbai Indians','Rajasthan Royals','Royal Challengers Bangalore',

       'Sunrisers Hyderabad']]

f
f = f[['Chennai Super Kings','Delhi Capitals','Kings XI Punjab','Kolkata Knight Riders',

       'Mumbai Indians','Rajasthan Royals','Royal Challengers Bangalore',

       'Sunrisers Hyderabad']]

f.plot(kind = 'bar',color = ['yellow','mediumblue','red','darkmagenta',

                                     'midnightblue','palevioletred','firebrick','darkorange'])

plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.xticks(fontsize =25,rotation = 65)

plt.yticks(fontsize = 25)

plt.xlabel("IPL Team",fontsize = 30)

plt.ylabel("Total Matches",fontsize = 30)

plt.title("Total Matches per team",fontsize = 50)
b = df['winner'].value_counts()

sns.countplot(x = 'winner',data = df,

              palette = 'CMRmap',

              order = df['winner'].value_counts()[:8].index)

plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.xticks(fontsize =25,rotation = 65)

plt.yticks(fontsize = 25)

plt.xlabel("IPL Team",fontsize = 30)

plt.ylabel("Number of wins",fontsize = 30)

plt.title("Number of wins per team",fontsize = 50)

print(b)
b = b[['Chennai Super Kings','Delhi Capitals','Kings XI Punjab','Kolkata Knight Riders',

       'Mumbai Indians','Rajasthan Royals','Royal Challengers Bangalore',

       'Sunrisers Hyderabad']]

b
wr = b/f * 100

wr
z = df['winner'].value_counts()

z = dict(z)

z
df
c = df['player_of_match'].value_counts()[:20]

sns.countplot(x = 'player_of_match',data = df,

              palette = 'plasma_r',

              order = df['player_of_match'].value_counts()[:20].index)

plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.xticks(fontsize =25,rotation = 65)

plt.yticks(fontsize = 25)

plt.xlabel("Player name",fontsize = 30)

plt.ylabel("Number of PoTM awards in the IPL",fontsize = 30)

plt.title("Number of PoTM awards per player",fontsize = 50)

plt.axis([-0.5,21,0,23])

print(c)
wr.plot(kind = 'bar',color =  ['yellow','mediumblue','red','darkmagenta','midnightblue',

                              'palevioletred','firebrick','darkorange'])

plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.xticks(fontsize =25,rotation = 65)

plt.yticks(fontsize = 25)

plt.ylim = (0,100)

plt.xlabel("IPL Team",fontsize = 30)

plt.ylabel("Winning Rate",fontsize = 30)

plt.title("Winnning rate per team",fontsize = 50)
h = df['city'].value_counts()

sns.countplot(x = 'city',data = df,

              palette = 'terrain',

              order = df['city'].value_counts()[:10].index)

plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.xticks(fontsize =25,rotation = 65)

plt.yticks(fontsize = 25)

plt.xlabel("City",fontsize = 30)

plt.ylabel("Number of matches",fontsize = 30)

plt.title("Number of matches per city",fontsize = 50)
i = df['toss_winner'].value_counts()

sns.countplot(x = 'toss_winner',data = df,

              palette = 'twilight',

              order = df['toss_winner'].value_counts().index)

plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.xticks(fontsize =25,rotation = 65)

plt.yticks(fontsize = 25)

plt.xlabel("IPL Team",fontsize = 30)

plt.ylabel("Number of wins",fontsize = 30)

plt.title("Number of wins per team",fontsize = 50)

print(i)
k = df['win_by_runs'].value_counts()

k[:20]
twdf = df[df['toss_winner']==df['winner']]

twdf.head()
a2 = twdf['winner'].value_counts()

a2.plot(kind = 'bar',color = 'mediumaquamarine')

plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.xticks(fontsize =25,rotation = 65)

plt.yticks(fontsize = 25)

plt.xlabel("IPL Team",fontsize = 30)

plt.ylabel("Wins with winning toss",fontsize = 30)

plt.title("Winnning with winning toss per team",fontsize = 50)

a2
b = a2/f

b = b *100

b1 = b[['Chennai Super Kings','Delhi Capitals','Kings XI Punjab','Kolkata Knight Riders',

       'Mumbai Indians','Rajasthan Royals','Royal Challengers Bangalore',

       'Sunrisers Hyderabad']]

b1.plot(kind = 'bar',color = ['yellow','mediumblue','red','darkmagenta','midnightblue',

                              'palevioletred','firebrick','darkorange'])

plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.xticks(fontsize =25,rotation = 65)

plt.yticks(fontsize = 25)

plt.ylim = (0,100)

plt.xlabel("IPL Team",fontsize = 30)

plt.ylabel("Wins with winning toss",fontsize = 30)

plt.title("Winnning with winning toss per team",fontsize = 50)
b1
wdf = df[['winner','toss_winner','toss_decision','win_by_runs','win_by_wickets']]

wdf.head()
w1 = wdf['win_by_runs'].value_counts(0)

w1
wa2 = w1.head(11)

wa2 = wa2.tail(10)

wa2
wa2.plot(kind = 'bar',color = ['black','maroon','chocolate','darkorange','yellowgreen',

                               'mediumseagreen','darkcyan','fuchsia','deeppink','crimson'])

plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.xticks(fontsize =25,rotation = 0)

plt.yticks(fontsize = 25)

plt.xlabel("Number of Runs",fontsize = 30)

plt.ylabel("Number of Matches",fontsize = 30)

plt.title("Number of Matches won on runs",fontsize = 50)
w2 = wdf['win_by_wickets'].value_counts(0)

w2

wa1 = w2.tail(10)
wa1.plot(kind = 'bar',color = ['black','maroon','chocolate','darkorange','yellowgreen',

                               'mediumseagreen','darkcyan','fuchsia','deeppink','crimson'])

plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.xticks(fontsize =25,rotation = 360)

plt.yticks(fontsize = 25)

plt.xlabel("Number of Wickets",fontsize = 30)

plt.ylabel("Number of Matches",fontsize = 30)

plt.title("Number of Matches won on wickets",fontsize = 50)
wdf['win_by_wickets'].value_counts(0)
df.shape
mwdf = df[(df['win_by_wickets']==0)]
mwdf = mwdf[['winner']]
mwdf #Runs
mw = mwdf['winner'].value_counts()

mw
mwp = mw/f *100

mwp = mwp1

mwp1
mwp1 = mwp[['Chennai Super Kings','Delhi Capitals','Kings XI Punjab','Kolkata Knight Riders',

       'Mumbai Indians','Rajasthan Royals','Royal Challengers Bangalore',

       'Sunrisers Hyderabad']]

mwp1.plot(kind = 'bar',color = ['yellow','mediumblue','red','darkmagenta',

                               'midnightblue','palevioletred','firebrick','darkorange'])

plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.xticks(fontsize =25,rotation = 60)

plt.yticks(fontsize = 25)

plt.axis([-0.5,7.5,0,100])

plt.xlabel("IPL Team",fontsize = 30)

plt.ylabel("Percentage of Matches won chasing",fontsize = 30)

plt.title("Percentage of Matches won on chasing per Team",fontsize = 50)
mrdf = df[(df['win_by_runs']==0)]
mrm = mrdf['winner']
mrm
mr = mrdf['winner'].value_counts()
mr
mw
f
mrp = mr/f *100

mrp1
mrp1 = mrp[['Chennai Super Kings','Delhi Capitals','Kings XI Punjab','Kolkata Knight Riders',

       'Mumbai Indians','Rajasthan Royals','Royal Challengers Bangalore',

       'Sunrisers Hyderabad']]

mrp1.plot(kind = 'bar',color = ['yellow','mediumblue','red','darkmagenta','midnightblue',

                                'palevioletred','firebrick','darkorange'])

plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.xticks(fontsize =25,rotation = 60)

plt.yticks(fontsize = 25)

plt.axis([-0.5,7.5,0,100])

plt.xlabel("IPL Team", fontsize = 30)

plt.ylabel("Percentage of Matches won by defending team",fontsize = 30)

plt.title("Percentage of Matches won by defending per team",fontsize = 50)
419 + 350
mwdf.shape
mrdf.shape
df.shape
rwdf = df[['win_by_runs','win_by_wickets','winner']]
rwdf
rwdf['win_by_runs'].groupby(rwdf['winner']).sum()
rwdf['win_by_wickets'].groupby(rwdf['winner']).sum()