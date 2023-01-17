# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("white")
plt.style.use('seaborn')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

#print(os.listdir("../input/NFL-Punt-Analytics-Competition"))
#print(os.listdir("../input/nflplaybyplay2009to2016"))
# Any results you write to the current directory are saved as output.
df_play_by_play=pd.read_csv('../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2018 (v5).csv',low_memory=False)
print('we imported the kaggle play by play Dataset')
df_play_by_play_dist=df_play_by_play[df_play_by_play['yardline_100']>-1]
df_play_by_play_dist=df_play_by_play_dist.reset_index()
drivesDistance=[]
currentDrive=1
driveBegin=0
puntDIsts=[]

for i in tqdm(range(len(df_play_by_play_dist)-1)):
    if df_play_by_play_dist['play_type'][i]=='kickoff':
        if i==0:
            currentDrive=df_play_by_play_dist['drive'][i]
            driveBegin=df_play_by_play_dist['yardline_100'][i+1]
        else:
            if df_play_by_play_dist['yardline_100'][i-1]>-1:
                drivesDistance.append(driveBegin-df_play_by_play_dist['yardline_100'][i-1])
            else:
                drivesDistance.append(driveBegin-df_play_by_play_dist['yardline_100'][i-2])
            currentDrive=df_play_by_play_dist['drive'][i]
            driveBegin=df_play_by_play_dist['yardline_100'][i+1]
    elif currentDrive==df_play_by_play_dist['drive'][i]:
        pass
    else:
        if df_play_by_play['yardline_100'][i-1]>-1:
            drivesDistance.append(driveBegin-df_play_by_play_dist['yardline_100'][i-1])
        else:
            drivesDistance.append(driveBegin-df_play_by_play_dist['yardline_100'][i-2])
        currentDrive=df_play_by_play_dist['drive'][i]
        driveBegin=df_play_by_play_dist['yardline_100'][i]

drivesDistance.append(driveBegin-df_play_by_play_dist['yardline_100'][i+1])


for i in tqdm(range(len(df_play_by_play_dist))):
    if df_play_by_play_dist['play_type'][i]=='punt':
        puntDIsts.append(df_play_by_play_dist['yardline_100'][i]-(100-df_play_by_play_dist['yardline_100'][i+1]))



print('On average, each drive makes a team go '+ str(int(np.mean(drivesDistance)))+' yards further' )
print('On average, each punt give the ball to the opposite team '+ str(int(np.mean(puntDIsts))) +' yards away')
print('So by punting (when you are less than 40 yards to the goal) you expect to get back the ball and a first down '+str(int(np.mean(puntDIsts)-np.mean(drivesDistance)))+ ' yards further' )
print('Punting is very advantaging for an offense')


df_play_by_play_fourth=df_play_by_play_dist[df_play_by_play_dist['down']==4]
df_play_by_play_fourth['5LineYard']=df_play_by_play_fourth['yardline_100'].apply(lambda x: int(x/5))
df_play_by_play_fourth=df_play_by_play_fourth.reset_index()
nb_punt_dist=[0]*20
nb_fieldgoal_dist=[0]*20
nb_fourth_dist=[0]*20
nb_pass_run_kneel_fourth_dist=[0]*20
ratio_punt_dist=[]
ratio_fieldgoal_dist=[]
ratio_fieldgoal_or_punt_dist=[]
ratio_punt_per_pass_run_kneel_play_dist=[]
dist=[]
for i in tqdm(range(len(df_play_by_play_fourth))):
    nb_fourth_dist[df_play_by_play_fourth['5LineYard'][i]]+=1
    if df_play_by_play_fourth['play_type'][i]=='punt':
        nb_punt_dist[df_play_by_play_fourth['5LineYard'][i]]+=1
    elif df_play_by_play_fourth['play_type'][i]=='field_goal':
        nb_fieldgoal_dist[df_play_by_play_fourth['5LineYard'][i]]+=1
    elif df_play_by_play_fourth['play_type'][i]=='qb_kneel' or df_play_by_play_fourth['play_type'][i]=='run' or df_play_by_play_fourth['play_type'][i]=='pass':
        nb_pass_run_kneel_fourth_dist[df_play_by_play_fourth['5LineYard'][i]]+=1

for i in range(20):
    ratio_punt_per_pass_run_kneel_play_dist.append(nb_punt_dist[i]/(nb_punt_dist[i]+nb_pass_run_kneel_fourth_dist[i]))
    ratio_punt_dist.append(nb_punt_dist[i]/nb_fourth_dist[i])
    ratio_fieldgoal_dist.append(nb_fieldgoal_dist[i]/nb_fourth_dist[i])
    ratio_fieldgoal_or_punt_dist.append((nb_fieldgoal_dist[i]+nb_punt_dist[i])/nb_fourth_dist[i])
    dist.append(5*i)
    
#plt.plot(dist,ratio_punt_dist)
#plt.plot(dist,ratio_fieldgoal_dist)
#plt.plot(dist,ratio_fieldgoal_or_punt_dist)
#plt.show()
f=plt.figure()
plt.plot(dist,ratio_punt_per_pass_run_kneel_play_dist)
plt.show()
f.savefig('ratio_punt_per_pass_run_kneel_play_dist.pdf',bbox_inches='tight')

groupByYardsToGo=df_play_by_play_fourth.groupby('ydstogo')
dist_to_go=[]
ratio_punt_dist_to_go=[]
ratio_punt_or_fieldgoal_dist_to_go=[]
ratio_punt_per_pass_run_kneel_dist_to_go=[]
nb_fourth_yard_to_go_punt_scrimmage=[]
for tupleDF in groupByYardsToGo:
    if tupleDF[0]<=20:
        df_tmp_perYardToGo=tupleDF[1]
        dist_to_go.append(tupleDF[0])
        ratio_punt_per_pass_run_kneel_dist_to_go.append(len(df_tmp_perYardToGo[df_tmp_perYardToGo['play_type']=='punt'])/len(df_tmp_perYardToGo[(df_tmp_perYardToGo['play_type']=='punt')|(df_tmp_perYardToGo['play_type']=='pass')|(df_tmp_perYardToGo['play_type']=='qb_kneel')|(df_tmp_perYardToGo['play_type']=='run')]))
        ratio_punt_or_fieldgoal_dist_to_go.append(len(df_tmp_perYardToGo[(df_tmp_perYardToGo['play_type']=='punt')|(df_tmp_perYardToGo['play_type']=='field_goal')])/len(df_tmp_perYardToGo))
        ratio_punt_dist_to_go.append(len(df_tmp_perYardToGo[df_tmp_perYardToGo['play_type']=='punt'])/len(df_tmp_perYardToGo))
        nb_fourth_yard_to_go_punt_scrimmage.append(len(df_tmp_perYardToGo[(df_tmp_perYardToGo['play_type']=='punt')|(df_tmp_perYardToGo['play_type']=='pass')|(df_tmp_perYardToGo['play_type']=='qb_kneel')|(df_tmp_perYardToGo['play_type']=='run')]))
#plt.plot(dist_to_go,ratio_punt_dist_to_go)
#plt.show()
#plt.plot(dist_to_go,ratio_punt_or_fieldgoal_dist_to_go)
#plt.show()
f=plt.figure()
plt.plot(dist_to_go,ratio_punt_per_pass_run_kneel_dist_to_go)
plt.show()
f.savefig('ratio_punt_per_pass_run_kneel_dist_to_go.pdf',bbox_inches='tight')

print('we can see that the ratio of punting  by fourth is depending a lot on the yardline because beneath 40 yards, fieldgoald is prefered to punting')
print('we only want to study punt, so we will only watch the proportion of punt among punt and scrimmages as in graph 2')
print('we can see that punting is majoritary choosed over scrimmage in fourth downs')
df_play_pass_run_kneel=df_play_by_play[(df_play_by_play['down']>0)&(df_play_by_play['yardline_100']>0)&(df_play_by_play['yards_gained']>-100)&((df_play_by_play['play_type']=='run')|(df_play_by_play['play_type']=='pass')|(df_play_by_play['play_type']=='qb_kneel')|(df_play_by_play['play_type']=='qb_spike'))]
df_play_pass_run_kneel=df_play_pass_run_kneel.reset_index()
plays_yards_gained=df_play_pass_run_kneel['yards_gained']
dic_proba_yard_gained={}
for i in range(len(plays_yards_gained)):
    try:
        dic_proba_yard_gained[plays_yards_gained[i]]+=1
    except:
        dic_proba_yard_gained[plays_yards_gained[i]]=1
for val in dic_proba_yard_gained:
    dic_proba_yard_gained[val]=dic_proba_yard_gained[val]/len(plays_yards_gained)
    
dic_proba_drive_yard_gained={}
for i in range(len(drivesDistance)):
    try:
        dic_proba_drive_yard_gained[drivesDistance[i]]+=1
    except:
        dic_proba_drive_yard_gained[drivesDistance[i]]=1
for val in dic_proba_drive_yard_gained:
    dic_proba_drive_yard_gained[val]=dic_proba_drive_yard_gained[val]/len(drivesDistance)

dic_proba_drive_more_than_yard_gained={}
for val1 in dic_proba_drive_yard_gained:
    dic_proba_drive_more_than_yard_gained[val1]=dic_proba_drive_yard_gained[val1]
    for val2 in dic_proba_drive_yard_gained:
        if val2>val1:
            dic_proba_drive_more_than_yard_gained[val1]+=dic_proba_drive_yard_gained[val2]

esperance_yards_to_go=[]
for i in range(1,21):
    res=0
    for val in dic_proba_yard_gained:
        if val>=i:
            res+=dic_proba_yard_gained[val]*val
        else:
            res+=dic_proba_yard_gained[val]*(val-27)
    esperance_yards_to_go.append(res)
    
f=plt.figure()
plt.plot(dist_to_go,esperance_yards_to_go)
plt.show()
f.savefig('esperance_yards_to_go.pdf',bbox_inches='tight')

df_play_pass_run_kneel['yards_gained'].hist(bins=100)
print('we can see that the expected return on going for a fourth is negative, it is indeed very risky compared to punting')

print('to reduce the proportion of punting over scrimmage during fourth, we want to power up the scrimmage, we propose that if a turnover of downs occur, we give the ball to the defensive team some yards away')
print('punt being extremely efficient, we want to make scrimmage less risky but we do not want neither that a failed fourth become more advantaging than a punt')
print('we want to find the best value to give back the ball, easy to do practicaly (a multiple of ten yards), that does not make punt irrelevant and still punishes a failed fourth')

for distanceBallGiven in [0,10,20,30]:
    esperance_yards_to_go=[]
    for i in range(1,21):
        res=0
        for val in dic_proba_yard_gained:
            if val>=i:
                res+=dic_proba_yard_gained[val]*val
            else:
                for val2 in dic_proba_drive_yard_gained:
                    res+=dic_proba_yard_gained[val]*dic_proba_drive_yard_gained[val2]*(val-val2+distanceBallGiven)
        esperance_yards_to_go.append(res)
    f=plt.figure()
    plt.plot(dist_to_go,esperance_yards_to_go)
    plt.show()
    f.savefig('esperance_yards_to_go'+str(distanceBallGiven)+'.pdf',bbox_inches='tight')
print('we can see here that we can improve the expected value of going for the fourth to positive values until 20 yards without making it more advantagous to fail the fourth')
print('no we want to measure the impact on the proportion of scrimmage of this rule')
mean_dist_punt=np.mean(puntDIsts)
proba_better_scrimmage_than_punt=[]
for i in range(1,21):
    res=0
    for val in dic_proba_yard_gained:
        if val>=i:
            for val2 in dic_proba_drive_yard_gained:
                if mean_dist_punt-val2<=val:
                    res+=dic_proba_yard_gained[val]*dic_proba_drive_yard_gained[val2]

    proba_better_scrimmage_than_punt.append(1-res)
#proba_better_scrimmage_than_punt[0]=ratio_punt_per_pass_run_kneel_dist_to_go[0]
print('probability of having better result with punt than scrimmage (we ignore 4th and one which is already favorised a lot by coaches)')
f=plt.figure()
plt.plot(dist_to_go[1:],ratio_punt_per_pass_run_kneel_dist_to_go[1:])
plt.plot(dist_to_go[1:],proba_better_scrimmage_than_punt[1:])
plt.show()
f.savefig('proba_better_scrimmage_than_punt_and_ratio_punt_per_pass_run_kneel_dist_to_go.pdf',bbox_inches='tight')

print(np.mean(proba_better_scrimmage_than_punt))
print(np.mean(ratio_punt_per_pass_run_kneel_dist_to_go))
print('we can see that the two courbs are very similar, so the coachs are naturally following probability laws when they choose wether to go for the fourth or to punt')

proba_better_scrimmage_than_punt_with_20_yard=[]
for i in tqdm(range(1,21)):
    res=0
    for val in dic_proba_yard_gained:
        if val>=i:
            for val2 in dic_proba_drive_yard_gained:
                if mean_dist_punt-val2<=val:
                    res+=dic_proba_yard_gained[val]*dic_proba_drive_yard_gained[val2]
        else:
            for val2 in dic_proba_drive_yard_gained:
                for val3 in dic_proba_drive_yard_gained:
                    if mean_dist_punt-val2<=val+20-val3:
                        res+=dic_proba_yard_gained[val]*dic_proba_drive_yard_gained[val2]*dic_proba_drive_yard_gained[val3]
    proba_better_scrimmage_than_punt_with_20_yard.append(1-res)

np.mean(proba_better_scrimmage_than_punt_with_20_yard)
f=plt.figure()
plt.plot(dist_to_go[1:],proba_better_scrimmage_than_punt_with_20_yard[1:])
plt.show()
f.savefig('proba_better_scrimmage_than_punt_with_20_yard.pdf',bbox_inches='tight')

print('this would be the new probability of punt being better than scrimmage (and so what the coach would do)')
actual_rate=0
new_rate=0
all_fourth_punt_scrimmage=np.sum(nb_fourth_yard_to_go_punt_scrimmage)

for i in range(20):
    actual_rate+=ratio_punt_per_pass_run_kneel_dist_to_go[i]*nb_fourth_yard_to_go_punt_scrimmage[i]
    new_rate+=proba_better_scrimmage_than_punt_with_20_yard[i]*nb_fourth_yard_to_go_punt_scrimmage[i]
actual_rate=actual_rate/all_fourth_punt_scrimmage
new_rate=new_rate/all_fourth_punt_scrimmage
print(actual_rate,new_rate)

print('By doing so, this rule should reduce the proportion of punt in fourth down from 83% to 64% ')