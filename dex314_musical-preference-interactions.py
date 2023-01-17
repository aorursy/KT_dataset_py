# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
cols = pd.read_csv('../input/columns.csv')

cols.head()
data = pd.read_csv('../input/responses.csv')

data.head()
cnt = 0

for c in data.columns:

    print(c, cnt, end=',')

    cnt += 1
music = data.iloc[:,0:18]

quesOrPoll = data.iloc[:,131:]
df = pd.concat([music, quesOrPoll], axis=1, join='outer')

df.columns

df_obj = df.select_dtypes(include=['object'])

for c in df_obj:

    df[c] = pd.factorize(df_obj[c])[0]
from collections import Counter

print(Counter(df_obj['Gender']))

print(Counter(df['Gender']))
fig, ax=plt.subplots(1,1,figsize=(12,10))

sns.heatmap( df.corr(), ax=ax)
df['binnedHeight'] = pd.cut(df['Height'], bins=8)

df['binnedWeight'] = pd.cut(df['Weight'], bins=8)
fig, ax=plt.subplots(1,1,figsize=(12,8))

sns.violinplot(x='binnedWeight',y='Musical',hue='Gender',data=df,inner='quart')
fig, ax=plt.subplots(1,1,figsize=(12,8))

sns.violinplot(x='binnedWeight',y='Latino',hue='Gender',data=df,inner='quart')
grp = df.groupby('Gender')

gender = grp.agg('count')

gender.reset_index(inplace=True)

gender.drop(['Music','binnedHeight','binnedWeight','Gender'], axis=1, inplace=True)

gender.drop([0], axis=0, inplace=True)

gender.head()
gT = gender.T

color2 = iter(plt.cm.rainbow(np.linspace(0,1,len(gT))))

fig, ax=plt.subplots(1,2,figsize=(12,8),sharey=True)

ax=ax.ravel()

for j in range(len(gT)):

    col2 = next(color2);

    _ = ax[0].barh(j, gT.iloc[j,0], color=col2, align='center');

    ax[0].set_title('Female'); ax[0].set_xlabel('Counts');

    _ = ax[1].barh(j, gT.iloc[j,1], color=col2, align='center');

    ax[1].set_title('Male'); ax[0].set_xlabel('Counts');

plt.yticks(np.arange(len(gT.index)), gT.index, fontsize=9);