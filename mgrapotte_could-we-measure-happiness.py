# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # beauty plts
import matplotlib.pyplot as plt #plts


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/2017.csv')
df.head()
df.columns = ['Country','HRank','HScore','Whisker_High','Whisker_low','GDP_Capita','Family','LifeExp','Freedom','Generosity','GovTrust','Dystopia']
df.head()
sns.relplot(x="GDP_Capita", y="HScore", data=df);
df_filtered = df[(df['GDP_Capita'] > 0.60)]
sns.relplot(x="GDP_Capita", y="HScore", data=df_filtered);
df_filtered2 = df[(df['GDP_Capita'] > 1.0 )]
df_filtered2 = df_filtered2[(df_filtered['GDP_Capita'] < 1.2 )]

sns.relplot(x="GDP_Capita", y="HScore", data=df_filtered2);
df_filtered2['Country']
df_filtered2.sort_values(by=['HScore'], ascending=False)

sns.relplot(x="Family", y="HScore", data=df_filtered2)
plt.scatter(df_filtered2.GDP_Capita.iloc[0], df_filtered2.HScore.iloc[0], color='r')
sns.relplot(x="LifeExp", y="HScore", data=df_filtered2)
plt.scatter(df_filtered2.LifeExp.iloc[0], df_filtered2.HScore.iloc[0], color='r')
sns.relplot(x="Freedom", y="HScore", data=df_filtered2)
plt.scatter(df_filtered2.Freedom.iloc[0], df_filtered2.HScore.iloc[0], color='r')
sns.relplot(x="Generosity", y="HScore", data=df_filtered2)
plt.scatter(df_filtered2.Generosity.iloc[0], df_filtered2.HScore.iloc[0], color='r')
sns.relplot(x="GovTrust", y="HScore", data=df_filtered2)
plt.scatter(df_filtered2.GovTrust.iloc[0], df_filtered2.HScore.iloc[0], color='r')
sns.relplot(x="LifeExp", y="HScore", data=df);
plt.scatter(df_filtered2.LifeExp.iloc[0], df_filtered2.HScore.iloc[0], color='r')
sns.relplot(x="Freedom", y="HScore", data=df);
plt.scatter(df_filtered2.Freedom.iloc[0], df_filtered2.HScore.iloc[0], color='r')
df_filtered3 = df[(df['GDP_Capita'] > 1.4 )]
sns.relplot(x="GDP_Capita", y="HScore", data=df_filtered3);
df_filtered3['Country']
df_filtered3.sort_values(by=['HScore'], ascending=False)
sns.relplot(x="Family", y="HScore", data=df_filtered3)
plt.scatter(df_filtered3.GDP_Capita.iloc[0], df_filtered3.HScore.iloc[0], color='r')
sns.relplot(x="LifeExp", y="HScore", data=df_filtered3)
plt.scatter(df_filtered3.LifeExp.iloc[0], df_filtered3.HScore.iloc[0], color='r')
sns.relplot(x="Freedom", y="HScore", data=df_filtered3)
plt.scatter(df_filtered3.Freedom.iloc[0], df_filtered3.HScore.iloc[0], color='r')
sns.relplot(x="Generosity", y="HScore", data=df_filtered3)
plt.scatter(df_filtered3.Generosity.iloc[0], df_filtered3.HScore.iloc[0], color='r')
sns.relplot(x="GovTrust", y="HScore", data=df_filtered3)
plt.scatter(df_filtered3.GovTrust.iloc[0], df_filtered3.HScore.iloc[0], color='r')
sns.relplot(x="Family", y="HScore", data=df);
plt.scatter(df_filtered3.Family.iloc[0], df_filtered3.HScore.iloc[0], color='r')
df_filtered4 = df[(df['GDP_Capita'] < 0.50)]
sns.relplot(x="GDP_Capita", y="HScore", data=df_filtered4);
df_filtered4['Country']
df_filtered4.sort_values(by=['HScore'], ascending=False)
sns.relplot(x="Family", y="HScore", data=df_filtered4)
plt.scatter(df_filtered4.GDP_Capita.iloc[0], df_filtered4.HScore.iloc[0], color='r')
plt.scatter(df_filtered4.GDP_Capita.iloc[1], df_filtered4.HScore.iloc[1], color='g')
sns.relplot(x="LifeExp", y="HScore", data=df_filtered4)
plt.scatter(df_filtered4.LifeExp.iloc[0], df_filtered4.HScore.iloc[0], color='r')
plt.scatter(df_filtered4.LifeExp.iloc[1], df_filtered4.HScore.iloc[1], color='g')
sns.relplot(x="Freedom", y="HScore", data=df_filtered4)
plt.scatter(df_filtered4.Freedom.iloc[0], df_filtered4.HScore.iloc[0], color='r')
plt.scatter(df_filtered4.Freedom.iloc[1], df_filtered4.HScore.iloc[1], color='g')
sns.relplot(x="Generosity", y="HScore", data=df_filtered4)
plt.scatter(df_filtered4.Generosity.iloc[0], df_filtered4.HScore.iloc[0], color='r')
plt.scatter(df_filtered4.Generosity.iloc[1], df_filtered4.HScore.iloc[1], color='g')
sns.relplot(x="GovTrust", y="HScore", data=df_filtered4)
plt.scatter(df_filtered4.GovTrust.iloc[0], df_filtered4.HScore.iloc[0], color='r')
plt.scatter(df_filtered4.GovTrust.iloc[1], df_filtered4.HScore.iloc[1], color='g')
sns.relplot(x="Freedom", y="HScore", data=df);
plt.scatter(df_filtered2.Freedom.iloc[0], df_filtered2.HScore.iloc[0], color='r')
plt.scatter(df_filtered3.Freedom.iloc[0], df_filtered3.HScore.iloc[0], color='y')
plt.scatter(df_filtered4.Freedom.iloc[0], df_filtered4.HScore.iloc[0], color='g')
sns.relplot(x="GDP_Capita", y="HScore", data=df);
plt.scatter(df_filtered2.GDP_Capita.iloc[0], df_filtered2.HScore.iloc[0], color='r')
plt.scatter(df_filtered3.GDP_Capita.iloc[0], df_filtered3.HScore.iloc[0], color='y')
plt.scatter(df_filtered4.GDP_Capita.iloc[0], df_filtered4.HScore.iloc[0], color='g')
sns.relplot(x="GDP_Capita", y="Generosity", data=df);
sns.relplot(x="Generosity", y="HScore", data=df);
