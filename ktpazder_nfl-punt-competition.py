%matplotlib inline
## Load necessary packages ##

import os
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix
from scipy.spatial.distance import pdist,squareform
from scipy.stats import chisquare
from tqdm.autonotebook import tqdm
from matplotlib import pyplot as plt
from matplotlib import cm
from glob import glob
from tqdm.autonotebook import tqdm
from datetime import timedelta
from shapely.geometry import LineString, Point
from toolz.itertoolz import sliding_window
from itertools import combinations

sns.set_style('darkgrid')
# distance to mph
dis2mph = 10 * 3600 / 1760 
## Read in auxiliary data ##

review = pd.read_csv('../input/video_review.csv')
players = pd.read_csv('../input/player_punt_data.csv').drop_duplicates('GSISID').set_index('GSISID')['Position']
players_all = pd.read_csv('../input/player_punt_data.csv')
roles = pd.read_csv('../input/play_player_role_data.csv')
## Group the roles ##

roles['RoleGroup'] = ''
roles.loc[roles.Role == 'P','RoleGroup'] = 'P'
roles.loc[roles.Role == 'PR','RoleGroup'] = 'PR'
roles.loc[roles.Role == 'PFB','RoleGroup'] = 'PFB'
roles.loc[roles.Role.str[0] == 'G','RoleGroup'] = 'G'
roles.loc[roles.Role.str[0] == 'V','RoleGroup'] = 'V'
roles.loc[roles.RoleGroup == '','RoleGroup'] = 'Line'

roles.head()
review.Player_Activity_Derived.value_counts().plot.bar()
plt.show()
review.Friendly_Fire.value_counts().plot.bar()
plt.show()
review.Turnover_Related.value_counts().plot.bar()
plt.show()
review.Primary_Impact_Type.value_counts().plot.bar()
plt.show()
review.Primary_Partner_Activity_Derived.value_counts().plot.bar()
plt.show()
## Add player information to the review data ##

review = pd.read_csv('../input/video_review.csv')
review = review.merge(players.reset_index(), on='GSISID', how='inner')
review = review.merge(roles, on=['Season_Year','GameKey','PlayID','GSISID'], how='inner')
review.head()
review.Role.value_counts().plot.bar()
plt.figure()
review.RoleGroup.value_counts().plot.bar()
plt.show()
rcounts = pd.DataFrame(roles.groupby(['Season_Year','GameKey','PlayID','RoleGroup'])['Role'].count()).reset_index()
rave = pd.DataFrame(rcounts.groupby(['RoleGroup'])['Role'].mean()).reset_index()
rave['Expected'] = rave.Role/22*review.shape[0]

EC = pd.DataFrame(review.RoleGroup.value_counts()).reset_index()
EC.columns = ['RoleGroup','Observed']

rave = rave.merge(EC, on='RoleGroup', how='inner')

Xstat, Xp = chisquare(rave.Observed, f_exp=rave.Expected)
print('Goodnees of fit p-value = %3f' %Xp)

rave
review['Possible_Cause'] = ['pushed from behind','engaged','poor tackling','poor tackling','pushed from side','poor blocking',\
                           'poor blocking','engaged friendly fire','pile up','','','collision',\
                           'poor tackling','illegal hit','turnover','poor blocking','poor tackling','engaged',\
                           'friendly fire','poor tackling','poor blocking','poor blocking','friendly cross fire','collision',\
                           'poor tackling','collision','poor tackling','poor blocking','friendly fire','collision',\
                            'poor blocking','poor blocking','poor tackling','poor blocking','collision','poor blocking',\
                           'poor tackling']
review['Blindsided'] = ['No','Yes','No','No','Yes','No',\
                       'No','Yes','No','','','Yes',\
                       'No','Yes','No','No','No','Yes',\
                       'No','No','No','Yes','No','Yes',\
                       'No','No','No','Yes','No','No',\
                        'Yes','No','No','Yes','No','No',\
                       'No']

## Note: Mislabeling exists on the player jersey number and friendly fire was missed when a player was pushed into another

review[['GameKey', 'PlayID', 'Possible_Cause', 'Blindsided']]
#review
## Summary table of hand curated labels

pd.crosstab(review.Possible_Cause, review.Blindsided, margins=True)
pos = ['DE', 'DT', 'NT', 'LB', 'ILB', 'OLB', 'MLB', 'CB', 'S', 'FS', 'SS', \
       'QB', 'RB', 'FB', 'WR', 'TE', 'OL', \
       'K', 'P', 'LS']
wgt = [283.1, 312.8, 312.8, 246.0, 246.0, (283.1+246.0)/2, 246.0, 200.2, 200.2, 200.2, 200.2, \
       224.1, 220.2, 220.2, 222.4, 222.4, 314.0, \
       202.3, 213.2, 245.3]

suppl = pd.DataFrame(pos)
suppl.columns = ['position']
suppl['weight'] = wgt

suppl
def viz_play(iplay, gamekey, playid):
    prole = roles.query('GameKey == @gamekey and PlayID == @playid')
    long_snapper = prole.query('Role == "PLS"').GSISID.iloc[0]
    punter = prole.query('Role == "P"').GSISID.iloc[0]
    iplay = iplay.query('GSISID in @prole.GSISID')
    assert len(iplay.GSISID.unique()) <= 22
    iplay.Time = pd.to_datetime(iplay.Time)
    iplay = iplay.sort_values('Time')

    scrimmage = iplay.query('GSISID == @long_snapper and Event == "ball_snap"').x.iloc[0]

    flip = 1
    punter = iplay.query('GSISID == @punter and Event == "ball_snap"').x.iloc[0]
    if punter > scrimmage:
        flip = -1

    rplay = review.query('GameKey == @gamekey and PlayID == @playid')

    snap = iplay.query('Event == "ball_snap"').Time.iloc[0]
    end = iplay.query('Event in ["tackle", "punt_downed", "out_of_bounds", "fair_catch", "touchdown"]').Time.iloc[-1] + timedelta(seconds=1.5)

    iplay = iplay[iplay.Time.between(snap, end)]

    for x, player in iplay.groupby('GSISID'):
        viridis = cm.viridis(np.linspace(0, 1, player.shape[0]))
        plasma = cm.plasma(np.linspace(0, 1, player.shape[0]))
        cividis = cm.cividis(np.linspace(0, 1, player.shape[0]))
        inferno = cm.inferno(np.linspace(0, 1, player.shape[0]))
        colors = cividis
        alpha = 0.1
        zorder = 0
        if int(rplay.iloc[0].GSISID) == x:
            colors = viridis
            alpha = .6
            zorder = 2
        if (rplay.Primary_Partner_GSISID.notnull().iloc[0]
            and rplay.Primary_Partner_GSISID.iloc[0] != 'Unclear' 
            and int(rplay.Primary_Partner_GSISID.iloc[0]) == x):
            colors = plasma
            alpha = .6
            zorder = 1
        plt.scatter(-flip * player.y + (flip > 0) * 53.3, flip * (player.x - scrimmage), c=colors, alpha=alpha, zorder=zorder)
    plt.xlim(-3, 55)
    plt.title(f'gamekey {gamekey} playid {playid}')
    plt.tight_layout()
def get_collisions(play, gamekey, playid, start, end, prole):
    total = (end - start).total_seconds() / 0.1 - 1
    data = []
    blocks = []
    timepoints = 2
    prev = set()
    for ts in tqdm(sliding_window(timepoints, play[play.Time.between(start, end)].Time.unique()), total=total, leave=False):
        new_prev = set()
        rev = review.query('GameKey == @gamekey and PlayID == @playid')
        try:
            injpair = set([int(rev.GSISID), int(rev.Primary_Partner_GSISID)])
        except ValueError:
            injpair = -1
        iframe = play.query('Time == @ts[1]').sort_values('GSISID')
        gsis_s = iframe.GSISID.values.astype(int)
        pairs = squareform(pdist(iframe[['x', 'y']])) < 5
        for i, j in zip(*pairs.nonzero()):
            if i >= j:
                continue
            gsis1 = gsis_s[i]
            gsis2 = gsis_s[j]
            assert gsis1 < gsis2

            locs1 = play.query('Time in @ts and GSISID == @gsis1')
            if locs1.shape[0] != timepoints:
                continue
            locs2 = play.query('Time in @ts and GSISID == @gsis2')
            if locs2.shape[0] != timepoints:
                continue
            x1 = locs1[['x', 'y']].values
            nx1 = locs1[['nx', 'ny']].values
            # sometimes players slow down so need 2x diff e.g game 448 play 2792
            #ls1 = LineString(np.append(x1[-1], np.diff(x1, axis=0) + x1[-1, :], 0))
            ls1 = LineString(np.c_[x1[-1], 2*np.diff(x1, axis=0).ravel() + x1[-1, :]].T)
            x2 = locs2[['x', 'y']].values
            nx2 = locs2[['nx', 'ny']].values
            #ls2 = LineString(np.append(x2[-1], np.diff(x2, axis=0) + x2[-1, :], 0))
            ls2 = LineString(np.c_[x2[-1], 2*np.diff(x2, axis=0).ravel() + x2[-1, :]].T)
            if ((Point(x1[0]).distance(Point(x2[0])) < 1.5)
                and ((np.abs(np.diff(x1, axis=0) - np.diff(x2, axis=0))).sum() < 0.25)):
                spd1 = locs1.iloc[0].dis * dis2mph
                spd2 = locs2.iloc[0].dis * dis2mph
                ang1 = np.rad2deg(np.arctan2(*locs1.iloc[:2][['x', 'y']].diff().iloc[1].values))
                ang2 = np.rad2deg(np.arctan2(*locs2.iloc[:2][['x', 'y']].diff().iloc[1].values))
                x = ang1 - ang2
                diffa = min(x % 360, abs((x % 360) - 360) % 360)
                pos1 = players.loc[gsis1]
                pos2 = players.loc[gsis2]
                    
                blocks.append({
                    'gamekey': gamekey,
                    'playid': playid,
                    'x': np.mean([x2[-1, 0], x1[-1, 0]]),
                    'y': np.mean([x2[-1, 1], x1[-1, 1]]),
                    'nx': np.mean([nx2[-1, 0], nx1[-1, 0]]),
                    'ny': np.mean([nx2[-1, 1], nx1[-1, 1]]),
                    'gsis1': gsis1,
                    'role1': prole.loc[gsis1, 'Role'],
                    'position1': pos1,
                    'gsis2': gsis2,
                    'role2': prole.loc[gsis2, 'Role'],
                    'position2': pos2,
                    'spd1': spd1,
                    'spd2': spd2,
                    'time': pd.to_datetime(ts[1]),
                    'angle1': ang1,
                    'angle2': ang2,
                    'angle_diff': diffa,
                })
            if ls1.distance(ls2) < 0.5 and Point(x1[0]).distance(Point(x2[0])) > 1.0:
                new_prev.add((gsis1, gsis2))
                if (gsis1, gsis2) not in prev:
                
                    spd1 = locs1.iloc[0].dis * dis2mph
                    spd2 = locs2.iloc[0].dis * dis2mph
                    ang1 = np.rad2deg(np.arctan2(*locs1.iloc[:2][['x', 'y']].diff().iloc[1].values))
                    ang2 = np.rad2deg(np.arctan2(*locs2.iloc[:2][['x', 'y']].diff().iloc[1].values))
                    x = ang1 - ang2
                    diffa = min(x % 360, abs((x % 360) - 360) % 360)
                    #ix = ls1.intersection(ls2)

                    pos1 = players.loc[gsis1]
                    pos2 = players.loc[gsis2]

    #                 if set([gsis1, gsis2]) == injpair:
    #                     print(list(ls1.coords()))

                    data.append({
                        'gamekey': gamekey,
                        'playid': playid,
                        'x': np.mean([x2[-1, 0], x1[-1, 0]]),
                        'y': np.mean([x2[-1, 1], x1[-1, 1]]),
                        'nx': np.mean([nx2[-1, 0], nx1[-1, 0]]),
                        'ny': np.mean([nx2[-1, 1], nx1[-1, 1]]),
                        'gsis1': gsis1,
                        'role1': prole.loc[gsis1, 'Role'],
                        'position1': pos1,
                        'gsis2': gsis2,
                        'role2': prole.loc[gsis2, 'Role'],
                        'position2': pos2,
                        'spd1': spd1,
                        'spd2': spd2,
                        'time': pd.to_datetime(ts[1]),
                        'angle1': ang1,
                        'angle2': ang2,
                        'angle_diff': diffa,
                        'injury': set([gsis1, gsis2]) == injpair
                    })
        prev = new_prev
    data = pd.DataFrame(data)
    blocks = pd.DataFrame(blocks)
    return data, blocks
## Vizualize motion ##

plays = pd.read_csv('../input/NGS-2016-reg-wk13-17.csv')
gamekey, playid = 281, 1526
viz_play(plays.query('GameKey == @gamekey and PlayID == @playid'), gamekey, playid)
all_blocks = []
all_collisions = []
for ngs in tqdm(glob('../input/NGS*.csv')):
    plays = pd.read_csv(ngs)
    for (gamekey, playid), play in tqdm(plays.merge(review[['GameKey', 'PlayID']]).groupby(['GameKey', 'PlayID']), leave=False):
        if play.empty:
            continue
        play.Time = pd.to_datetime(play.Time)
        
        
        prole = roles.query('GameKey == @gamekey and PlayID == @playid')
        long_snapper = prole.query('Role == "PLS"').GSISID.iloc[0]
        punter = prole.query('Role == "P"').GSISID.iloc[0]
        scrimmage = play.query('GSISID == @long_snapper and Event == "ball_snap"').x.iloc[0]

        flip = 1
        punter = play.query('GSISID == @punter and Event == "ball_snap"').x.iloc[0]
        if punter > scrimmage:
            flip = -1
            
        play['nx'] = -flip * play.y + (flip > 0) * 53.3
        play['ny'] = flip * (play.x - scrimmage)
            
        start = play.query('Event == "ball_snap"').Time.iloc[0]
        end = play.query('Event in ["tackle", "punt_downed", "out_of_bounds", "fair_catch", "touchdown"]').Time.iloc[-1] + timedelta(seconds=1.5)

        prole = roles.query('GameKey == @gamekey and PlayID == @playid')[['GSISID', 'Role']]
        play = play.merge(prole, on=('GSISID'))
        prole = prole.set_index('GSISID')
        #play = play.sort_values(by=('Time', 'GSISID'))
        play = play.sort_values(['Time', 'GSISID'])

        a, b = get_collisions(play, gamekey, playid, start, end, prole)
        all_collisions.append(a)
        all_blocks.append(b)
collisions = pd.concat(all_collisions, ignore_index=True)
#blocks = pd.concat(all_blocks, ignore_index=True)
## Data Engineering ##
collisions['sumspd'] = collisions.spd1 + collisions.spd2
collisions['maxspd'] = collisions[['spd1','spd2']].max(axis=1)
collisions['position'] = collisions.position1 + '-' + collisions.position2
collisions['role'] = collisions.role1 + '-' + collisions.role2

## Add auxiliary information ##
punt = collisions.merge(review, left_on=['gamekey','playid'], right_on=['GameKey','PlayID'], how='inner')
print(punt.shape)
print('%i concussion collisions detected' %sum(punt.injury))
punt.head()
punt_wgt = punt.merge(suppl, left_on='position1', right_on='position', how='outer')
punt_wgt = punt_wgt.merge(suppl, left_on='position2', right_on='position', how='outer')
punt_wgt['weight_diff'] = abs(punt_wgt.weight_y - punt_wgt.weight_x)

punt_wgt = punt_wgt.dropna(subset=['angle1', 'angle2'])
punt_wgt = punt_wgt.reset_index(drop=True)

print(punt_wgt.shape)
punt_wgt.tail()
sns.scatterplot(x='angle_diff', y='maxspd', hue='injury', style='injury', data=punt, s=100)
plt.ylabel('Maximum Speed')
plt.xlabel('Angle Difference')
plt.show()
sns.scatterplot(x='angle_diff', y='sumspd', hue='injury', style='injury', data=punt, s=100)
plt.ylabel('Combined Speed')
plt.xlabel('Angle Difference')
plt.show()
sns.scatterplot(data=punt_wgt, x='angle_diff', y='sumspd', size='weight_diff', style='injury', hue='injury')
plt.xlabel('Angle Difference')
plt.ylabel('Combined Speed')
plt.show()
X = np.vstack((punt.angle_diff, punt.sumspd, punt.angle_diff*punt.sumspd)).T
y = punt.injury
alpha = 0.1

print(X.shape, y.shape)
clf_cv = LogisticRegressionCV(cv=5, random_state=0).fit(X, y)


#inj_clf_cv = clf_cv.predict(X)
inj_clf_cv = clf_cv.predict_proba(X)[:,1] > alpha


print(classification_report(y,inj_clf_cv))
print(confusion_matrix(y,inj_clf_cv))
# Parameters
n_classes = 2
plot_colors = "ryb"
plot_step = 0.02

plt.figure(figsize=(6,5))

x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))


Z = clf_cv.predict(np.c_[xx.ravel(), yy.ravel(), xx.ravel()*yy.ravel()])
Z = clf_cv.predict_proba(np.c_[xx.ravel(), yy.ravel(), xx.ravel()*yy.ravel()])[:,1] > alpha
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

plt.ylabel('Speed')
plt.xlabel('Angle')



# Plot the training points
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.grid(True)
plt.ylabel("Combined Speed")
plt.xlabel("Angle Difference")
plt.title('Concussion Probability Regions')
plt.show()
punt_sub = punt.query('Possible_Cause != "poor tackling" & Possible_Cause != "poor blocking" & Possible_Cause != "illegal hit" & Possible_Cause != "pushed from behind" & Possible_Cause != "pushed from side"')
print('%i concussion collisions detected' %sum(punt_sub.injury))
punt_sub.shape
X = np.vstack((punt_sub.angle_diff, punt_sub.sumspd, punt_sub.angle_diff*punt_sub.sumspd)).T
y = punt_sub.injury
alpha = 0.1

print(X.shape, y.shape)

clf_cv = LogisticRegressionCV(cv=5, random_state=0).fit(X, y)


#inj_clf_cv = clf_cv.predict(X)
inj_clf_cv = clf_cv.predict_proba(X)[:,1] > alpha


print(classification_report(y,inj_clf_cv))
print(confusion_matrix(y,inj_clf_cv))
# Parameters
n_classes = 2
plot_colors = "ryb"
plot_step = 0.02

plt.figure(figsize=(6,5))

x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))


Z = clf_cv.predict(np.c_[xx.ravel(), yy.ravel(), xx.ravel()*yy.ravel()])
Z = clf_cv.predict_proba(np.c_[xx.ravel(), yy.ravel(), xx.ravel()*yy.ravel()])[:,1] > alpha
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

plt.ylabel('Speed')
plt.xlabel('Angle')



# Plot the training points
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.grid(True)
plt.ylabel("Combined Speed")
plt.xlabel("Angle Difference")
plt.title('Concussion Probability Regions')
plt.show()
sns.scatterplot(x='nx', y='ny', data=collisions.query('injury == False'), color='blue', s=70, alpha=0.5, label='normal')
sns.scatterplot(x='nx', y='ny', data=collisions.query('injury == True'), color='red', s=70, alpha=1.0, label='injury')
plt.title('Relative location of injuries')
plt.ylabel('distance down field')
plt.xlabel('width of field')
plt.show()
