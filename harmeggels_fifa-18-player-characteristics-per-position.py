import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



df = pd.read_csv('../input/complete.csv', encoding='latin1')



df.head()

list(df.columns.values)

attributes = ['crossing', 'finishing', 'heading_accuracy', 'short_passing', 'volleys', 'dribbling',

              'curve', 'free_kick_accuracy', 'long_passing', 'ball_control', 'acceleration', 

              'sprint_speed', 'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina', 

              'strength', 'long_shots', 'aggression', 'interceptions', 'positioning', 'vision',

              'penalties', 'composure', 'marking', 'standing_tackle', 'sliding_tackle',

              'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning', 'gk_reflexes']



positions = ['rs', 'rw', 'rf', 'ram', 'rcm', 'rm', 'rdm', 'rcb', 'rb', 'rwb', 'st', 'lw',

             'cf', 'cam', 'cm', 'lm', 'cdm', 'cb', 'lb', 'lwb', 'ls', 'lf', 'lam', 'lcm',

             'ldm', 'lcb', 'gk']



cols_pref = [x for x in df.columns if 'prefers' in x and 'gk' not in x]

min_pref_pos = min(df.loc[:, cols_pref].sum(axis=1))

max_pref_pos = max(df.loc[:, cols_pref].sum(axis=1))



print('The minimum of preferred positions of a player is {0}.'.format(min_pref_pos))

print('The maximum of preferred positions of a player is {0}.'.format(max_pref_pos))
import matplotlib.pyplot as plt

df_att = df.loc[:, attributes].applymap(lambda x: x/100)

cols_inc = [x for x in df_att.columns if not x.startswith('gk')]

df_att = df_att.loc[df_att.sum(axis=1) > 0, cols_inc]

df_att = df_att.apply(lambda x: (x - x.mean()))



df_prefs = df.loc[:, cols_pref]



y = df_prefs.sum().sort_values(ascending=False)

x =np.arange(1, len(y) + 1)



fig, ax = plt.subplots(1,1, figsize=(15, 6))

ax.bar(np.arange(1, len(y) + 1), y)

ax.set_xticks(x)

ax.set_xticklabels(y.index,rotation = 45, ha="right")



#Filter out the positions which are never preferred

cols_incl = [x for x in y.index if y[x] > 0]

df_prefs = df_prefs.loc[:, cols_incl]
from sklearn.multiclass import OneVsRestClassifier

from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier()

mclf = OneVsRestClassifier(clf)



mclf.fit(df_att, df_prefs)
from matplotlib import colors

#norm = colors.BoundaryNorm(np.linspace(0,cmap.N,len(feat_imp.index)) , cmap.N)



feat_imp = pd.DataFrame()

feat_imp_sum = np.zeros(len(df_prefs.columns))



fig, ax = plt.subplots(1,1, figsize=(15, 10))



x = 1

for l, e in zip(df_prefs.columns, mclf.estimators_):

    feat_imp.loc[:, l] = e.feature_importances_



colors = np.concatenate((plt.cm.tab20(np.linspace(0, 1, 20)),

                        plt.cm.Set3(np.linspace(0, 1, 10))))

#colors = plt.cm.tab20c(np.linspace(0, 1, len(df_att.columns)))

x = np.arange(1, len(feat_imp.columns) + 1)

y = np.zeros(len(feat_imp.columns))

for n in range(0, len(feat_imp.index)):

    ax.bar(x, feat_imp.loc[n, :], bottom=y, label=df_att.columns[n], color=colors[n])

    y += feat_imp.loc[n, :]



ax.set_xticks(x)

ax.set_xticklabels(feat_imp.columns,rotation = 45, ha="right")

ax.legend(bbox_to_anchor=(1.02,1), loc="upper left")