# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')
df.tail()
df.dtypes
df.describe().T
df[df['Year_of_Release']>2016.0]
df['Name'].isnull().values.sum()
df.dropna(subset=['Name'],inplace=True)
df['Name'].isnull().values.sum()
# df['Publisher'].unique()
df['Publisher'].replace({'Sony Computer Entertainment':'Sony',
                         'Sony Computer Entertainment America':'Sony',
                         'Sony Computer Entertainment Europe':'Sony',
                         'Sony Music Entertainment':'Sony',
                         'Sony Online Entertainment':'Sony',
                        },inplace=True)
# df['Publisher']
sns.catplot(y="Genre", x="Global_Sales", data=df) #, kind="swarm")
sns.catplot(y="Genre", x="Global_Sales", data=df,
            kind="box", showfliers=False) #outliner 제거
sns.catplot(y="Genre", x="Critic_Score", data=df,
            kind="box", showfliers=False) #outliner 제거
# f, ax = plt.subplots(figsize=(200,10))
ax = sns.catplot(x="Year_of_Release", hue="Platform", kind="count", data=df,height=5, aspect=10)