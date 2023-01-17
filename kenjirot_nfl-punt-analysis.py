# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import datetime
import collections
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#read video_review.csv
injury_df = pd.read_csv('../input/video_review.csv')
#count player activity and primary impact with the occurrence of concussions
tackling_impact = pd.Series([0,0,0,0],index = ['Helmet-to-helmet', 'Helmet-to-body', 'Helmet-to-ground', 'Unclear'])
tackled_impact = pd.Series([0,0,0,0],index = ['Helmet-to-helmet', 'Helmet-to-body', 'Helmet-to-ground', 'Unclear'])
blocking_impact = pd.Series([0,0,0,0],index = ['Helmet-to-helmet', 'Helmet-to-body', 'Helmet-to-ground', 'Unclear'])
blocked_impact = pd.Series([0,0,0,0],index = ['Helmet-to-helmet', 'Helmet-to-body', 'Helmet-to-ground', 'Unclear'])

for i in range(len(injury_df.index)):
    act = injury_df.loc[i, 'Player_Activity_Derived']
    imp = injury_df.loc[i, 'Primary_Impact_Type']
    
    if act == 'Tackling':
        if imp == 'Helmet-to-body':
            tackling_impact['Helmet-to-body'] += 1
        elif imp == 'Helmet-to-helmet':
            tackling_impact['Helmet-to-helmet'] += 1
        elif imp == 'Helmet-to-ground':
            tackling_impact['Helmet-to-ground'] += 1
        else:
            tackling_impact['Unclear'] += 1
    elif act == 'Tackled':
        if imp == 'Helmet-to-body':
            tackled_impact['Helmet-to-body'] += 1
        elif imp == 'Helmet-to-helmet':
            tackled_impact['Helmet-to-helmet'] += 1
        elif imp == 'Helmet-to-ground':
            tackled_impact['Helmet-to-ground'] += 1
        else:
            tackled_impact['Unclear'] += 1
    elif act == 'Blocking':
        if imp == 'Helmet-to-body':
            blocking_impact['Helmet-to-body'] +=1
        elif imp == 'Helmet-to-helmet':
            blocking_impact['Helmet-to-helmet'] +=1
        elif imp == 'Helmet-to-ground':
            blocking_impact['Helmet-to-ground'] +=1
        else:
            blocking_impact['Unclear'] +=1
    elif act == 'Blocked':
        if imp == 'Helmet-to-body':
            blocked_impact['Helmet-to-body'] += 1
        elif imp == 'Helmet-to-helmet':
            blocked_impact['Helmet-to-helmet'] += 1
        elif imp == 'Helmet-to-ground':
            blocked_impact['Helmet-to-ground'] += 1
        else:
            blocked_impact['Unclear'] += 1

#create dataframe
act_impact = pd.DataFrame()
act_impact['Tackling'] = tackling_impact
act_impact['Tackled'] = tackled_impact
act_impact['Blocking'] = blocking_impact
act_impact['Blocked'] = blocked_impact
act_impact
#visualize the dataframe above
plt.rcParams['font.size'] = 14
fig, ax = plt.subplots()
im = ax.imshow(act_impact.values)

cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Times', rotation=-90, va="bottom")

ax.set_xticks(np.arange(4))
ax.set_yticks(np.arange(4))
ax.set_xticklabels(act_impact.columns)
ax.set_yticklabels(act_impact.index)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

for i in range(4):
    for j in range(4):
        text = ax.text(j, i, act_impact.values[i, j], ha="center", va="center", color="w", fontweight='bold')

ax.set_title("Activity & Impact")
fig.tight_layout()
plt.show()
#Use NGS data and get them together

#read NGS in 2016 and 2017
data2016_post = pd.read_csv('../input/NGS-2016-post.csv')
data2016_pre = pd.read_csv('../input/NGS-2016-pre.csv')
data2016_reg1 = pd.read_csv('../input/NGS-2016-reg-wk1-6.csv')
data2016_reg2 = pd.read_csv('../input/NGS-2016-reg-wk7-12.csv')
data2016_reg3 = pd.read_csv('../input/NGS-2016-reg-wk13-17.csv')

data2017_post = pd.read_csv('../input/NGS-2017-post.csv')
data2017_pre = pd.read_csv('../input/NGS-2017-pre.csv')
data2017_reg1 = pd.read_csv('../input/NGS-2017-reg-wk1-6.csv')
data2017_reg2 = pd.read_csv('../input/NGS-2017-reg-wk7-12.csv')
data2017_reg3 = pd.read_csv('../input/NGS-2017-reg-wk13-17.csv')
#concat data for each year
data2016 = pd.concat([data2016_pre,data2016_reg1,data2016_reg2,data2016_reg3,data2016_post])
data2017 = pd.concat([data2017_pre,data2017_reg1,data2017_reg2,data2017_reg3,data2017_post])
#read player role
role = pd.read_csv('../input/play_player_role_data.csv')
role = role.drop('Season_Year', axis = 1)
#add "player role" column
data2016 = pd.merge(data2016,role,on=['GameKey','PlayID','GSISID'])
data2017 = pd.merge(data2017,role,on=['GameKey','PlayID','GSISID'])
#extract (GameKey, PlayID) list when punt receive or fair-catch was happened.
puntPlay16_1 = data2016.query('Role=="P"').query('Event=="punt"')
puntPlay16_2 = data2016.query('Role=="PLS"').query('Event=="ball_snap"')
puntPlay16_3 = data2016.query('Role=="PR"').query('Event=="punt_received" or Event == "fair_catch"')
gamePlay16_1 = [(i,j) for i,j in zip(puntPlay16_1.GameKey, puntPlay16_1.PlayID)]
gamePlay16_2 = [(i,j) for i,j in zip(puntPlay16_2.GameKey, puntPlay16_2.PlayID)]
gamePlay16_3 = [(i,j) for i,j in zip(puntPlay16_3.GameKey, puntPlay16_3.PlayID)]
gamePlay16 = list(set(gamePlay16_1) & set(gamePlay16_2) & set(gamePlay16_3))
puntPlay17_1 = data2017.query('Role=="P"').query('Event=="punt"')
puntPlay17_2 = data2017.query('Role=="PLS"').query('Event=="ball_snap"')
puntPlay17_3 = data2017.query('Role=="PR"').query('Event=="punt_received" or Event == "fair_catch"')
gamePlay17_1 = [(i,j) for i,j in zip(puntPlay17_1.GameKey, puntPlay17_1.PlayID)]
gamePlay17_2 = [(i,j) for i,j in zip(puntPlay17_2.GameKey, puntPlay17_2.PlayID)]
gamePlay17_3 = [(i,j) for i,j in zip(puntPlay17_3.GameKey, puntPlay17_3.PlayID)]
gamePlay17 = list(set(gamePlay17_1) & set(gamePlay17_2) & set(gamePlay17_3))
#create defensive role list 
defense_id = ['PDR1', 'PDR2', 'PDR3', 'PDL1', 'PDL2', 'PDL3', 'PLR', 'PLM', 'PLL', 'VRo', 'VRi', 'VLi', 'VLo', 'PR', 'PFB', 'VR', 'VL']
#A function that includes some features (number of rushing players, number of conrnerbacks, hang time, punt distance, fair-catch)
def punt_feature(game_key, play_id, df):
    
    #extract a specific play in a specific game
    df0 = df.query('GameKey == "%s"'%game_key).query('PlayID == "%s"'%play_id)
    
    #extract each event (ball_snap, punt, punt_received, fair_catch, others)
    df_snap = df0.query('Event == "ball_snap"').reset_index().drop('index', axis = 1)
    df_punt = df0.query('Event == "punt"').reset_index().drop('index', axis = 1)
    df_received = df0.query('Event == "punt_received"').reset_index().drop('index', axis = 1)
    df_faircatch = df0.query('Event == "fair_catch"').reset_index().drop('index', axis = 1)
    df_tackle = df0.query('Event == "tackle"').reset_index().drop('index', axis = 1)
    df_out = df0.query('Event == "out_of_bounds"').reset_index().drop('index', axis = 1)
    df_touchdown = df0.query('Event == "touchdown"').reset_index().drop('index', axis = 1)
    df_submit = df0.query('Event == "play_submit"').reset_index().drop('index', axis = 1)
    
    #extract x of scrimage line
    scrimage_x = df_snap.query('Role == "PLS"')['x'].values[0]
    
    #extract x of punt reterner at receiving
    if df_received.shape[0] > 0:
        pr_x = df_received.query('Role == "PR"')['x'].values[0]    
    else:
        pr_x = df_faircatch.query('Role == "PR"')['x'].values[0]
        
    #extract x of punt returner at play end
    if df_tackle.query('Role == "PR"').shape[0] > 0:
        pr_xe = df_tackle.query('Role == "PR"')['x'].values[0]
    elif df_out.query('Role == "PR"').shape[0] > 0:
        pr_xe = df_out.query('Role == "PR"')['x'].values[0]
    elif df_touchdown.query('Role == "PR"').shape[0] > 0:
        pr_xe = df_touchdown.query('Role == "PR"')['x'].values[0]
    elif df_submit.query('Role == "PR"').shape[0] > 0:
        pr_xe = df_submit.query('Role == "PR"')['x'].values[0]
    else:
        pr_xe = pr_x
        
    #calculate hang time
    punt_time = datetime.datetime.strptime(df_punt.loc[0,'Time'], '%Y-%m-%d %H:%M:%S.%f')
    if df_received.shape[0] > 0:
        receive_time = datetime.datetime.strptime(df_received.loc[0,'Time'], '%Y-%m-%d %H:%M:%S.%f')
        fair_count = 0
    else:
        receive_time = datetime.datetime.strptime(df_faircatch.loc[0,'Time'], '%Y-%m-%d %H:%M:%S.%f')
        fair_count = 1
    hang_time = (receive_time - punt_time).total_seconds()
    
    #calculate punt distance
    punt_dist = abs(pr_x - scrimage_x)
    #calculate return distance
    return_dist = abs(pr_x - pr_xe)
    
    #count cornerbacks
    CB_count = 0
    for r in df_snap.Role:
        if r in ['VRo', 'VRi', 'VLi', 'VLo', 'VR', 'VL']:
            CB_count += 1
        else:
            CB_count += 0
    
    return [game_key,play_id,punt_dist,return_dist,hang_time,CB_count,fair_count]
#create dataframe that includes some features
punt_feat16 = pd.DataFrame(index = ['GameKey', 'PlayID', 'PuntDist', 'ReturnDist','HangTime', 'CB', 'fair_catch'])
for i in range(len(gamePlay16)):
    g_id = gamePlay16[i][0]
    p_id = gamePlay16[i][1]
    punt_feat16[i] = punt_feature(g_id,p_id,data2016)
punt_feat17 = pd.DataFrame(index = ['GameKey', 'PlayID', 'PuntDist', 'ReturnDist','HangTime', 'CB', 'fair_catch'])
for i in range(len(gamePlay17)):
    g_id = gamePlay17[i][0]
    p_id = gamePlay17[i][1]
    punt_feat17[i] = punt_feature(g_id,p_id,data2017)
punt_feat16 = punt_feat16.T
punt_feat17 = punt_feat17.T
punt_feat16['Year'] = 2016
punt_feat17['Year'] = 2017
punt_feat = pd.concat([punt_feat16, punt_feat17])
punt_feat.index = range(punt_feat.shape[0])
punt_feat.head()
#probability of fair-catch by the number of cornerbacks
fairCatch_CB = pd.DataFrame(index = ['receive', 'fair_catch'])
for i in [2,3,4]:
    fairCatch_CB[i] = [punt_feat.query('CB=="%s"'%i).query('fair_catch==0').shape[0], punt_feat.query('CB=="%s"'%i).query('fair_catch==1').shape[0]]
    fairCatch_CB[i] = 100 * fairCatch_CB[i] / fairCatch_CB[i].sum()
fairCatch_CB
#Number of cornerbacks VS probability of fair-catch
fairCatch_CB.T.plot.bar(stacked = True)
plt.xlabel('Number of Cornerbacks')
plt.ylabel('Probability of fair-catch [%]')
plt.legend(['Receiving', 'Fair-catch'],loc = (1.01, 0.8))
fair_ratio = np.array([punt_feat.query('fair_catch==0').shape[0], punt_feat.query('fair_catch==1').shape[0]])
fair_ratio = fair_ratio / fair_ratio.sum()
print(100*fair_ratio[1])#probability of fair-catch in all punt play
concussion_id = [(y,i,j) for y,i,j in zip(injury_df.Season_Year, injury_df.GameKey, injury_df.PlayID)]
punt_injury = pd.DataFrame()
for y, g, p in concussion_id:
    df = punt_feat.query('Year=="%s"'%y).query('GameKey=="%s"'%g).query('PlayID=="%s"'%p)
    punt_injury = pd.concat([punt_injury, df])
punt_injury.query('fair_catch==1')
injury_df.query('GameKey=="%s"'%280).query('PlayID=="%s"'%2918)
# When a wing blocked a rusing player
injury_df.query('GameKey=="%s"'%506).query('PlayID=="%s"'%1988)
# A returner showed fair-catch signal but he caught a one bounced ball and returned. Therefore this play should not be counted as fair-catch.
injury_df.query('GameKey=="%s"'%607).query('PlayID=="%s"'%978)
# When a wing blocked a rusing player
#calculate probability of the concussion by number of cornerbacks
cb_count_all = collections.Counter(punt_feat.CB)
cb_count_injury = collections.Counter(punt_injury.CB)
injuryRate_cb = pd.Series(index = [2, 3, 4])
for i in [2,3,4]:
    injuryRate_cb[i] = 100 *cb_count_injury[i] / cb_count_all[i]
injuryRate_cb
injuryRate_cb.plot.bar(color = 'C2')
plt.ylabel('Probability of concussion injury [%]')
plt.xlabel('Number of cornerbacks')
100 * (injury_df.shape[0] - round(punt_feat.shape[0] * injuryRate_cb[2] / 100)) / injury_df.shape[0]
# average punt distance (about 47 yards)
punt_feat.PuntDist.mean()
# How punt distance effects fair-catch
fair_puntDist = pd.Series(index = np.arange(20,65,5))
puntDist_cb2 = pd.DataFrame(index = ['fair-catch','receive'])
for i in np.arange(20,65,5):
    df = punt_feat[(punt_feat['PuntDist'] >= i) & (punt_feat['PuntDist'] < i + 5)]
    fair0_count = df.query('CB==2').query('fair_catch==0').shape[0]
    fair1_count = df.query('CB==2').query('fair_catch==1').shape[0]
    
    puntDist_cb2[i] = [fair1_count,fair0_count]
    
    if fair0_count == 0 and fair1_count == 0:
        fair_puntDist[i] = 0
    else:
        fair_puntDist[i] = 100 * fair1_count / (fair0_count + fair1_count)
        
fair_puntDist.index = ['20-25','25-30','30-35','35-40','40-45','45-50','50-55','55-60','60-65']
puntDist_cb2.columns = ['20-25','25-30','30-35','35-40','40-45','45-50','50-55','55-60','60-65']
puntDist_cb2.T.plot.bar(stacked = True)
plt.xlabel('Punt distance [yards]')
plt.ylabel('Number of punt play [-]')
plt.ylim(0,500)
fair_puntDist.plot.bar(color = 'C0')
plt.ylim(0,100)
plt.xlabel('Punt distance [yards]')
plt.ylabel('Probability of fair-catch [%]')
# Return yards when punt distance is less than 50 yards under the conditions, two cornerbacks and no fair-catch
punt_feat.query('CB==2').query('fair_catch==0').query('PuntDist<50').ReturnDist.mean()
# Return yards when punt distance is more than 50 yards under the conditions, two cornerbacks and no fair-catch
punt_feat.query('CB==2').query('fair_catch==0').query('PuntDist>=50').ReturnDist.mean()