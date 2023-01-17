import numpy as np
import pandas as pd
pd.options.display.max_columns = 100

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('Set1')

import warnings
warnings.simplefilter('ignore')
df = pd.read_csv('/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')
df.head(3)
null = df.isnull().sum().to_frame(name='null').T
dtypes = df.dtypes.to_frame(name='types').T
pd.concat([null, dtypes], axis=0)
i = df['gameId'].value_counts().max()
print('max gameId unique == ',i)
df['blueWins'].value_counts().plot.bar(title='target values')
sub_df = df[['blueWins','blueWardsPlaced','redWardsPlaced']]
sub_df['wards_diff'] = sub_df['blueWardsPlaced'] - sub_df['redWardsPlaced']
sub_df['wards_diff_bin'] = pd.cut(sub_df['wards_diff'], [-250,-100,-50,-25,-20,-10,-5,0,5,10,20,25,50,100,250])
plt.figure(figsize=(15,4))
sns.countplot(sub_df['wards_diff_bin'],hue=sub_df['blueWins'])
sub_df = df[['blueWins','blueWardsPlaced','redWardsPlaced','blueWardsDestroyed','redWardsDestroyed']]
sub_df['blueRemainedWards'] = sub_df['blueWardsPlaced'] - sub_df['redWardsDestroyed']
sub_df['redRemainedWards'] = sub_df['redWardsPlaced'] - sub_df['blueWardsDestroyed']
sub_df['wards_remained_diff'] = sub_df['blueRemainedWards'] - sub_df['redRemainedWards']
sub_df['wards_remained_diff_bin'] = pd.cut(sub_df['wards_remained_diff'], [-250,-100,-50,-25,-20,-10,-5,0,5,10,20,25,50,100,250])
plt.figure(figsize=(15,4))
sns.countplot(sub_df['wards_remained_diff_bin'],hue=sub_df['blueWins'])
sub_df = df[['blueWins','blueFirstBlood','blueKills','blueDeaths']]
sub_df['kill_death_diff'] = sub_df['blueKills'] - sub_df['blueDeaths']
sns.catplot(x='blueWins', y='kill_death_diff',col='blueFirstBlood',
            data=sub_df, kind="violin")
sub_df = df[['blueWins','blueAssists','redAssists']]
sub_df['assists_diff'] = sub_df['blueAssists'] - sub_df['redAssists']
sns.boxplot(x=sub_df['blueWins'], y=sub_df['assists_diff'])
sub_df = df[['blueWins','blueEliteMonsters','blueDragons','blueHeralds','redEliteMonsters','redDragons','redHeralds']]
sub_df = pd.pivot_table(sub_df, index='blueWins', values=['blueEliteMonsters','blueDragons','blueHeralds',
                                                 'redEliteMonsters','redDragons','redHeralds'], aggfunc=np.mean)
cm = sns.light_palette("blue", as_cmap=True)
sub_df.style.background_gradient(cmap=cm)
sub_df = df[['blueWins','blueTowersDestroyed','redTowersDestroyed']]
pd.crosstab(sub_df['blueWins'],sub_df['blueTowersDestroyed'])
pd.crosstab(sub_df['blueWins'],sub_df['redTowersDestroyed'])
sub_df = df[['blueWins','blueTotalGold','blueGoldDiff','blueGoldPerMin',
        'redTotalGold','redGoldDiff','redGoldPerMin']]
sns.boxplot(x=sub_df['blueWins'], y=sub_df['blueGoldDiff'])
sub_df = df[['blueWins','blueAvgLevel','redAvgLevel']]
sub_df['Avglevel_diff'] = sub_df['blueAvgLevel'] - sub_df['redAvgLevel']
sns.boxplot(x=sub_df['blueWins'], y=sub_df['Avglevel_diff'])
sub_df = df[['blueWins','blueTotalExperience','blueExperienceDiff','redTotalExperience','redExperienceDiff']]
sns.boxplot(x=sub_df['blueWins'], y=sub_df['blueExperienceDiff'])
sub_df = df[['blueWins','blueTotalMinionsKilled','blueTotalJungleMinionsKilled','blueCSPerMin',
            'redTotalMinionsKilled','redTotalJungleMinionsKilled','redCSPerMin']]
sub_df['TotalMinionsKilled_diff'] = sub_df['blueTotalMinionsKilled'] - sub_df['redTotalMinionsKilled'] 
sub_df['TotalJungleMinionsKilled_diff']  =sub_df['blueTotalJungleMinionsKilled'] - sub_df['redTotalJungleMinionsKilled']
plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
sns.boxplot(x=sub_df['blueWins'], y=sub_df['TotalMinionsKilled_diff'])
plt.subplot(1,2,2)
sns.boxplot(x=sub_df['blueWins'], y=sub_df['TotalJungleMinionsKilled_diff'])
df_corr = round(df.corr(), 2)
plt.figure(figsize=(18,15))
sns.heatmap(df_corr,cmap='jet')
df['wards_diff'] = df['blueWardsPlaced'] - df['redWardsPlaced']

df['blueRemainedWards'] = df['blueWardsPlaced'] - df['redWardsDestroyed']
df['redRemainedWards'] = df['redWardsPlaced'] - df['blueWardsDestroyed']
df['wards_remained_diff'] = df['blueRemainedWards'] - df['redRemainedWards']

df['blue_kill_death_diff'] = df['blueKills'] - df['blueDeaths']

df['assists_diff'] = df['blueAssists'] - df['redAssists']

df['Avglevel_diff'] = df['blueAvgLevel'] - df['redAvgLevel']

df['TotalMinionsKilled_diff'] = df['blueTotalMinionsKilled'] - df['redTotalMinionsKilled'] 
df['TotalJungleMinionsKilled_diff']  =df['blueTotalJungleMinionsKilled'] - df['redTotalJungleMinionsKilled']
drop_col = ['blueWardsPlaced','redWardsPlaced','blueWardsDestroyed','redWardsDestroyed',
            'blueRemainedWards','redRemainedWards','blueKills','redKills',
           'blueDeaths','redDeaths','blueAssists','redAssists',
           'blueAvgLevel','redAvgLevel','blueTotalMinionsKilled','redTotalMinionsKilled',
           'blueTotalJungleMinionsKilled','redTotalJungleMinionsKilled',
           'blueDragons','blueHeralds','redDragons','redHeralds',
           'blueTotalGold','redTotalGold','blueGoldPerMin','redGoldPerMin',
           'blueTotalExperience','redTotalExperience','blueCSPerMin','redCSPerMin',
           'redGoldDiff','redExperienceDiff','redFirstBlood']

df = df.drop(drop_col, 1)
df_corr = round(df.corr(), 2)
plt.figure(figsize=(15,10))
sns.heatmap(df_corr, annot=True, cmap='jet')
X = df.drop(['gameId','blueWins'], 1)
y = df['blueWins']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=5, random_state=0)
clf.fit(x_train, y_train)
clf.score(x_test, y_test)