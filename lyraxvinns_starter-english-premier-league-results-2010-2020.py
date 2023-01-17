# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import necessary libraries
##
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('bright')
sns.set_context('notebook')

from tqdm import tqdm
# load csv file and preview
##
df = pd.read_csv("/kaggle/input/epl results 2010-2020.csv")

display(df.head(3))
# overall stats
##
print(f"DataFrame size : {df.size}")
print(f"DataFrame shape : {df.shape}")
# percentage of missing data
##
print(f"Overall missing data : {df.isnull().values.sum()/ df.size * 100 :.2f}%")
# Checking if there are columns with over 60% missing data
##
to_drop = []
for col in tqdm(df.columns):

    # If column has missing values and they are more than 60% of the total
    if df[str(col)].isnull().any() == True and df[str(col)].isnull().values.sum() > len(df[str(col)]) * 0.6:
        print(col)
        to_drop.append(col)
    else:
        pass

# Dropping these columns
##
df.drop(to_drop, axis=1, inplace=True)
# checking percentage of missing data again
##
print(f"Overall missing data : {df.isnull().values.sum()/ df.size * 100 :.2f}%")
# df stats
##
print(f"DataFrame size : {df.size}")
print(f"DataFrame shape : {df.shape}")
# now previewing number of missing data points in every column with missing data
##
df.isnull().sum()
# Note there is a row with missing date, this row might be null all through the columns
##
df[df.Date.isnull()]
df.dropna(subset=['Date'], axis=0, inplace=True)
df
df.isnull().sum().unique()
# dropping nans
##
df.dropna(inplace=True)
# reset index
##
df.reset_index(drop=True, inplace=True)
# A plot of how much the teams played from home
##
plt.figure(figsize=(12,6))
sns.countplot(y='HomeTeam', data=df)
plt.tight_layout()
# A plot of how much the teams played from away
##
plt.figure(figsize=(12,6))
sns.countplot(y='AwayTeam', data=df)
plt.tight_layout()
# The above two plots as subplots
##
fig, ax = plt.subplots(1,2, figsize=(14,8))
sns.countplot(y='HomeTeam', data=df, ax=ax[0])
ax[0].set_title("HomeTeam")
sns.countplot(y='AwayTeam', data=df, ax=ax[1])
ax[1].set_title("AwayTeam")
plt.tight_layout()
# new column 'result' to show home-win, draw or away-win
##
df['result'] = np.nan

for i, team in enumerate(df.HomeTeam):
    if df.loc[i, 'FTHG'] > df.loc[i, 'FTAG']:
        df.loc[i, 'result'] = 'Home Win'
    elif df.loc[i, 'FTHG'] == df.loc[i, 'FTAG']:
        df.loc[i, 'result'] = 'Draw'
    elif df.loc[i, 'FTHG'] < df.loc[i, 'FTAG']:
        df.loc[i, 'result'] = 'Away Win'
    else:
        pass
df.head(3)
# distribution of results
##
sns.countplot('result', data=df, order=['Home Win', 'Away Win', 'Draw'])
# plot to show home wins for every team
##
plt.figure(figsize=(12,7))
sns.countplot(y='HomeTeam',data=df[df.result == 'Home Win'])
# plot to show Away wins for every team
##
plt.figure(figsize=(12,7))
sns.countplot(y='AwayTeam',data=df[df.result == 'Away Win'])
# plot to show draws
##
fig, ax = plt.subplots(1, 2, figsize=(12,8))
sns.countplot(y='HomeTeam', data=df[df.result == 'Draw'], ax=ax[0])
ax[0].set_title("HomeTeam")
sns.countplot(y='AwayTeam', data=df[df.result == 'Draw'], ax=ax[1])
ax[1].set_title("AwayTeam")

plt.tight_layout()
# preview which teams the above teams with most draws have Drawn with
##
print("Teams drawing with Southampton\n===============================================")
display(df[(df.HomeTeam == 'Southampton') & (df.result == 'Draw')])

print("Teams drawing with West Brom\n===============================================")
display(df[(df.HomeTeam == 'West Brom') & (df.result == 'Draw')])

print("Teams drawing with Everton\n===============================================")
display(df[(df.HomeTeam == 'Everton') & (df.result == 'Draw')])
fig, ax = plt.subplots(3,2, figsize=(15,10))
sns.countplot(y='AwayTeam', data=df[(df.HomeTeam == 'Southampton') & (df.result == 'Draw')], ax=ax[0,0])
ax[0,0].set_title("Teams that did Draw with Southampton(Home)")
sns.countplot(y='HomeTeam', data=df[(df.AwayTeam == 'Southampton') & (df.result == 'Draw')], ax=ax[0,1])
ax[0,1].set_title("(Away)")

sns.countplot(y='AwayTeam', data=df[(df.HomeTeam == 'West Brom') & (df.result == 'Draw')], ax=ax[1,0])
ax[1,0].set_title("Teams that did Draw with West Brom(Home)")
sns.countplot(y='HomeTeam', data=df[(df.AwayTeam == 'West Brom') & (df.result == 'Draw')], ax=ax[1,1])
ax[1,1].set_title("(Away)")

sns.countplot(y='AwayTeam', data=df[(df.HomeTeam == 'Everton') & (df.result == 'Draw')], ax=ax[2,0])
ax[2,0].set_title("Teams that did Draw with Everton(Home)")
sns.countplot(y='HomeTeam', data=df[(df.AwayTeam == 'Everton') & (df.result == 'Draw')], ax=ax[2,1])
ax[2,1].set_title("(Away)")

plt.tight_layout()
# heatmap
##
plt.figure(figsize=(12,7))
sns.heatmap(df.corr())