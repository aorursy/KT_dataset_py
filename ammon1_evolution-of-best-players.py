# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/NBA_player_of_the_week.csv')
df.head()
print('Missing values:\n',df.isna().sum())
print('All values\n',df.shape)
teams=df[(df.Conference=='East')|(df.Conference=='West')]
teams_=teams.Team.unique()
conf={}
for team in teams_:
    conferences=teams.Conference[teams.Team==team].unique()[0]
    conf[team]=conferences
df['Conference']=df['Team'].map(conf)
print('Missing values:\n',df.isna().sum())
df.head()
df.groupby(['Player']).sum().sort_values(['Real_value'],ascending=False,)
df_LBJ=df[df.Player=='LeBron James']
df_MJ=df[df.Player=='Michael Jordan']
df_KM=df[df.Player=='Karl Malone']
df_KB=df[df.Player=='Kobe Bryant']
df_SO=df[df.Player=="Shaquille O'Neal"]
LBJ=df_LBJ.groupby(['Season short']).sum()
LBJ['Season']=LBJ.index
LBJ['Name']='LeBron James'

MJ=df_MJ.groupby(['Season short']).sum()
MJ['Season']=MJ.index
MJ['Name']='Michael Jordan'

KM=df_KM.groupby(['Season short']).sum()
KM['Season']=KM.index
KM['Name']='Karl Malone'

KB=df_KB.groupby(['Season short']).sum()
KB['Season']=KB.index
KB['Name']='Kobe Bryant'

SO=df_SO.groupby(['Season short']).sum()
SO['Season']=SO.index
SO['Name']="Shaquille O'Neal"

KB.head()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.figure(figsize=(15,15))
sns.lineplot(x='Season',y='Real_value',data=LBJ,hue='Name',palette='Accent')
sns.lineplot(x='Season',y='Real_value',data=MJ,hue='Name',palette='Blues')
sns.lineplot(x='Season',y='Real_value',data=KM,hue='Name',palette='OrRd')
sns.lineplot(x='Season',y='Real_value',data=KB,hue='Name',palette='colorblind')
sns.lineplot(x='Season',y='Real_value',data=SO,hue='Name',palette='BuPu')