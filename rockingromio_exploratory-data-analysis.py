import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("../input/college-basketball-dataset/cbb.csv")
df.describe()
print(df.shape)
print(df.info())
df.columns
df['W_ratio'] = df['W'] / df['G']
df.head()
del df['G']
del df['W']
df.columns
df.shape
df.head(20)
df['POSTSEASON'].unique()
df['POSTSEASON'].value_counts()
d = {'Champions' : 1, '2ND' : 2, 'F4' : 3, 'E8' : 8, 'R68' : 5, 'S16' : 5, 'R32' : 6, 'R64' : 7}

df['POSTSEASON'] = df['POSTSEASON'].map(d)
df.head(10)
df['CONF'].value_counts()
df['Win_prob_.5'] = df['W_ratio'] >= 0.5
df.head(10)
df['Win_prob_.5'].value_counts()
df.corr()
pd.crosstab(df['Win_prob_.5'], df['POSTSEASON'])
corr_mat = df.corr()
corr_mat['Win_prob_.5']
plt.figure(figsize=(20,10))

sns.heatmap(corr_mat)
corr_mat['W_ratio']
sns.distplot(df['ADJOE'])
sns.boxplot(x = "Win_prob_.5", y = "ADJOE", data = df)
sns.boxplot(x = "Win_prob_.5", y = "ADJDE", data = df)
df_melt = pd.melt(frame = df , id_vars = ['Win_prob_.5'], value_vars = ['ADJOE'])
df_melt.head()
ax = sns.violinplot(x = "variable", y = "value", hue = "Win_prob_.5", data = df_melt , palette="muted", split=True)
sns.distplot(df['BARTHAG'])
sns.boxplot(x = "Win_prob_.5", y = "BARTHAG", data = df)
sns.boxplot(x = "Win_prob_.5", y = "EFG_O", data = df)
sns.boxplot(x = "Win_prob_.5", y = "EFG_D", data = df)
df['shot_diff'] = df['EFG_O'] - df['EFG_D']
sns.boxplot(x = "Win_prob_.5", y = "shot_diff", data = df)
sns.boxplot(x = "Win_prob_.5", y = "TOR", data = df)
sns.boxplot(x = "Win_prob_.5", y = "TORD", data = df)
sns.boxplot(x = "Win_prob_.5", y = "ORB", data = df)
sns.boxplot(x = "Win_prob_.5", y = "DRB", data = df)
df['rebound_diff'] = df['ORB'] - df['DRB']
sns.boxplot(x = "Win_prob_.5", y = "rebound_diff", data = df)
sns.boxplot(x = 'Win_prob_.5' , y = 'FTR', data = df)
sns.boxplot(x = 'Win_prob_.5', y = 'FTRD', data = df)
df['throw_diff'] = df['FTR'] - df['FTRD']
sns.boxplot(x = 'Win_prob_.5', y = 'throw_diff', data = df)
sns.boxplot(x = 'Win_prob_.5', y = '2P_O', data = df)
sns.boxplot(x = 'Win_prob_.5', y = '2P_D', data = df)
df['2p_diff'] = df['2P_O'] - df['2P_D']
sns.boxplot(x = 'Win_prob_.5', y = '2p_diff', data = df)
sns.boxplot(x = 'Win_prob_.5', y = '3P_O', data = df)
sns.boxplot(x = 'Win_prob_.5', y = '3P_D', data = df)
df.columns
df['3p_diff'] = df['3P_O'] - df['3P_D']
sns.boxplot(x = 'Win_prob_.5', y = '3p_diff', data = df)
df['Win_prob_.5'] = df['Win_prob_.5'].astype(int)
sns.distplot(df[df['Win_prob_.5'] ==1]['3p_diff'])
df_win = df[df['Win_prob_.5'] == 1]
df_win['3p_diff'].hist()
df_loss = df[df['Win_prob_.5'] == 0]
df_loss['3p_diff'].hist()
df_win['2p_diff'].hist()
df_loss['2p_diff'].hist()
df['2p_diff'].describe()
sns.boxplot(x = 'Win_prob_.5' , y = 'WAB' , data = df)