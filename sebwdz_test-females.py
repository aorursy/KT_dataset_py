# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt
import scipy.stats

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/CHATSWORTH MASTER_DATABASE.csv')
df = df.drop(columns=['V60']).apply(pd.to_numeric, errors='coerce')
df.head(10)
df.describe()
df = df.where(df['V55'] == 2).where(df['V54'] <= 2).where(df['V57'] <= 2).where(df['V11'] >= 2).dropna()
df.count()
df.describe()
columns = ['V' + str(x) for x in list(range(10, 21)) + [45]]
df_mode = df[columns].mode()
df_mean = pd.DataFrame(df[columns].mean()).transpose()
pd.concat([df_mean, df_mode])
continous = ['V' + str(x) for x in list(range(10, 21))]
corr = df[continous].corrwith(df['V56'])
corr.sort_values(ascending=False)
continous = ['V' + str(x) for x in list(range(22, 44)) + [56]]
corr = df[continous].corrwith(df['V45'])
corr.sort_values(ascending=False)
columns = ['V' + str(x) for x in list(range(22, 44))]
df_mean_option = df[columns].mean()
df_mean_option.sort_values(ascending=False)
columns = ['V' + str(x) for x in list(range(10, 22))]
df_mean_option = df[columns].mean()
df_mean_option.sort_values(ascending=False)
plt.scatter(df['V42'], df['V45'])
plt.show()
scipy.stats.pearsonr(df['V42'], df['V45'])
df[['V45']].mean()
df[['V56']].mean()