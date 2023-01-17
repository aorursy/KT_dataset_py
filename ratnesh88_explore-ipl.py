import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

deliveries = pd.read_csv("../input/deliveries.csv", sep=",")

matches = pd.read_csv("../input/matches.csv", sep=",")

print(deliveries.columns)

deliveries.info()
matches.info()
total_matches = matches.groupby('team1').size() + matches.groupby('team2').size()

print(total_matches)

total_matches.plot("bar",figsize=(12,6),color='gold')
matches['winner'].value_counts().plot("pie",figsize=(10,10))
matches['toss_winner'].value_counts().plot("pie",figsize=(10,10))
matches['city'].value_counts().plot("pie",figsize=(10,10))
matches['venue'].value_counts().plot("pie",figsize=(10,10))
matches['dl_applied'].value_counts()
d  = deliveries['dismissal_kind'].value_counts()

d
o = deliveries.loc[deliveries['dismissal_kind']=='obstructing the field']

o
d.plot("pie",figsize=(10,10))
deliveries['is_super_over'].value_counts()
o = deliveries.loc[deliveries['is_super_over']==1]

o
o['batsman'].value_counts().plot('bar',figsize=(12,8),title="Batsman of Superovers")
o['bowler'].value_counts().plot('bar',figsize=(12,8),title="Bowlers of Superovers")
temp =deliveries.groupby('batting_team').sum()
df2 = pd.DataFrame(np.array(temp[['wide_runs','bye_runs', 'legbye_runs', 'noball_runs', 'penalty_runs'

       ]]),columns=['wide_runs','bye_runs', 'legbye_runs', 'noball_runs', 'penalty_runs'], index=temp.index)

p =df2.plot.bar(figsize=(12,8),stacked = True)

p.set_title("Extra runs division batting teamwise",color='b',fontsize=40)

p.set_xlabel("Batting team name",color='g',fontsize=20)

p.set_ylabel("Frequency",color='g',fontsize=20)
temp =deliveries.groupby('bowling_team').sum()

df2 = pd.DataFrame(np.array(temp[['wide_runs','bye_runs', 'legbye_runs', 'noball_runs', 'penalty_runs'

       ]]),columns=['wide_runs','bye_runs', 'legbye_runs', 'noball_runs', 'penalty_runs'], index=temp.index)

p =df2.plot.bar(figsize=(12,8),stacked = True)

p.set_title("Extra runs division bowling teamwise",color='b',fontsize=40)

p.set_xlabel("Bowling team name",color='g',fontsize=20)

p.set_ylabel("Frequency",color='g',fontsize=20)
runs = deliveries['total_runs'] - deliveries['extra_runs']



p = runs.value_counts().plot('bar',figsize=(12,8))

p.set_title("Run form",color='b',fontsize=40)

p.set_xlabel("Run type",color='g',fontsize=20)

p.set_ylabel("Frequency",color='g',fontsize=20)