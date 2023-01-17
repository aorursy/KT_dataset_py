import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
train_years = [i for i in range(2010,2014)]
test_years = [i for i in range(2014,2018)]
teams = pd.read_csv('../input/Teams.csv')
events = []
for y in train_years:
    events.append(pd.read_csv('../input/Events_'+str(y)+'.csv'))
events[0].head()
compact = pd.read_csv('../input/NCAATourneyCompactResults.csv')
compact.head()
compact.Season.unique()
import seaborn as sns
import matplotlib.pyplot as plt
import math
f,axa = plt.subplots(2,2,figsize=(25,35))

sns.set(font_scale = 2)
wthist = compact.groupby(['WTeamID','Season'],
                           as_index=False).sum().sort_values(by=['WScore'],
                         ascending=False)
for i in range(4):
    yw = wthist[wthist['Season']==train_years[i]]
    yw = pd.merge(yw,teams,left_on='WTeamID',right_on='TeamID')
    sns.barplot(x=yw['TeamName'],y=yw['WScore'],orient='v',ax=axa[math.floor(i/2)][(i%2)])
    axa[math.floor(i/2)][(i%2)].set_title(str(train_years[i]))
    axa[math.floor(i/2)][(i%2)].tick_params(axis='x', rotation=90)
    #axa[math.floor(i/2)][(i%2)].labels.set_visible(False)




players = []
for y in train_years:
    players.append(pd.read_csv('../input/Players_'+str(y)+'.csv'))
players[0].head()
coaches = pd.read_csv('../input/TeamCoaches.csv')
coaches[coaches['Season']==2010][0:10]
#coaches.head()
import seaborn as sns
import matplotlib.pyplot as plt
import math

f,axa = plt.subplots(2,2,figsize=(25,35))
sns.set(font_scale = 2)
wthist = compact.groupby(['WTeamID','Season'],
                           as_index=False).sum().sort_values(by=['WScore'],
                         ascending=False)
for i in range(4):
    yw = wthist[wthist['Season']==train_years[i]]

    yw = pd.merge(yw,coaches,left_on=['WTeamID','Season'],right_on=['TeamID','Season'])
    yw = yw.groupby(['CoachName'],
                           as_index=False).sum().sort_values(by=['WScore'],
                         ascending=False)[0:10]
    sns.barplot(x=yw['CoachName'],y=yw['WScore'],orient='v',ax=axa[math.floor(i/2)][(i%2)])
    axa[math.floor(i/2)][(i%2)].set_title(str(train_years[i]))
    axa[math.floor(i/2)][(i%2)].tick_params(axis='x', rotation=90)
seeds = pd.read_csv('../input/NCAATourneySeeds.csv')
seeds.groupby(['Seed','TeamID'],as_index=False).count().head()

sub = pd.read_csv('../input/SampleSubmissionStage1.csv')
sub.head()