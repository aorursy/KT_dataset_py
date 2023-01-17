# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/PUBG_Player_Statistics.csv')
df.head()

df.info()
df.shape
df.describe()
df['solo_KillDeathRatio'].value_counts()

# From this we can see solo kd ratio on avg is high for 1.0
df['solo_RoundsPlayed'].value_counts()
solo=df.filter(regex='solo')
duo=df.filter(regex='duo')
squad=df.filter(regex='squad')

print(len(df)
      ,len(solo)
      ,len(duo)
      ,len(squad))
print(len(df.columns)
      ,len(solo.columns)
      ,len(duo.columns)
      ,len(squad.columns))
corr = df.corr()
corr.info(verbose=True)
cols_to_drop = ['Id', 'groupId']
cols_to_fit = [col for col in train_df.columns if col not in cols_to_drop]
corr = df[cols_to_fit].corr()

plt.figure(figsize=(9,7))
sns.heatmap(
    corr,
    xticklabels=corr.columns.values,
    yticklabels=corr.columns.values,
    linecolor='white',
    linewidths=0.1,
    cmap="RdBu"
)
plt.show()