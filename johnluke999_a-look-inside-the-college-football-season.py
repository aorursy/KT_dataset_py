# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import DataFrame, Series



# These are the plotting modules adn libraries we'll use:

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats



# Command so that plots appear in the iPython Notebook

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#read in dataset

df = pd.read_csv('/kaggle/input/college-football-team-stats-2019/CFB2019.csv',sep=',')

#separate Team, Conference, Wins, Losses and calculate Win Percentage, Average Rank(Off Rank & Def Rank), and yard diff per play



df[['wins','losses']]=df['Win-Loss'].str.split("-",expand=True)

df['wins']=pd.to_numeric(df['wins'])

df1=pd.DataFrame(df.Team.str.split("(",1).tolist(), columns = ['Tm','Conf'])

df.insert(1,'Tm',df1['Tm'])

df2=pd.DataFrame(df1.Conf.str.split(")",1).tolist(), columns = ['Conference','x'])

df.insert(2,'Conference',df2['Conference'])

df.drop('Team',axis=1,inplace=True)

df['winpct']=df['wins'] / df['Games']

df['Conference1']=df["Conference"].replace({"FL":"ACC","OH":"MAC"})

df.drop('Conference',axis=1,inplace=True)

df.insert(2,'Conference',df['Conference1'])

df['Avg Rank']=(df['Off Rank']+df['Def Rank'])/2

df['Yard Diff Per Play']=df['Off Yards/Play'] - df['Yards/Play Allowed']
#Regression graph for Offensive Rank and Defensive Rank Vs Win Pct with correlation



#Results show a stronger correlation between Defensive Rank and Win Pct



g=sns.jointplot('winpct','Off Rank',df,kind='reg')

g.annotate(stats.pearsonr)

plt.show()

g=sns.jointplot('winpct','Def Rank',df,kind='reg')

g.annotate(stats.pearsonr)

plt.show()
filter=df['Conference']== 'ACC'

filter2=df['Conference']== 'SEC'

filter3=df['Conference']== 'Big 12'

filter4=df['Conference']== 'Big Ten' 

filter5=df['Conference']== 'Pac-12'

dfn=df.where(filter | filter2 | filter3 | filter4 | filter5)

sns.boxplot(dfn['Avg Rank'], dfn['Conference'], whis=np.inf)

plt.title('Avg Rank(Off & Def) for Power 5 Conferences')

plt.show()

df[['Tm','Off Rank','Def Rank','Avg Rank']].sort_values('Avg Rank')[0:10]
# Bottom 10 in Avg Rank

df[['Tm','Off Rank','Def Rank','Avg Rank']].sort_values('Avg Rank')[-10:]
#Rushing yards vs Passing yards per game

sns.jointplot(df['Pass Yards Per Game'],df["Rushing Yards per Game"],kind='hex')

# How did Rushing and Passing yards affect Win Pct

forpair=df[['Pass Yards Per Game','Rushing Yards per Game','winpct']]

sns.pairplot(forpair)
# Noticable correlation between turnovers and win percentage

g=sns.jointplot('Turnover Rank','winpct',df,kind='reg')

g.annotate(stats.pearsonr)

plt.show()

# Ohio State and Clemson Conference Disparity 



acc1=df['Conference']== 'ACC'

acc=df.where(acc1).dropna()



acc[['Tm','Win-Loss','Yard Diff Per Play', 'Turnover Margin','Off Rank','Def Rank', 'Avg Rank']].sort_values('Yard Diff Per Play', ascending=False)
big1=df['Conference']== 'Big Ten'

bigten=df.where(big1).dropna()



bigten[['Tm','Win-Loss','Yard Diff Per Play','Turnover Margin','Off Rank','Def Rank','Avg Rank']].sort_values('Yard Diff Per Play', ascending=False)
sec1=df['Conference']== 'SEC'

sec=df.where(sec1).dropna()



sec[['Tm','Win-Loss','Yard Diff Per Play','Turnover Margin','Off Rank','Def Rank','Avg Rank']].sort_values('Yard Diff Per Play', ascending=False)