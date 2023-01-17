import pandas as pd

import numpy as numpy

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

plt.rcParams['figure.figsize']=30,15
df = pd.read_excel("../input/laliga-player-stats/La liga Player stats.xlsx")
df.head()
df['Nation'].value_counts()[:5]
df[df['Nation'] == 'fr FRA'].reset_index()
df[df['Squad'] == 'Barcelona']
df.describe()
df.isnull().sum()
a = df['Age'].value_counts()
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

sns.countplot(x = "Age",

              data = df,

              palette= "Spectral",)

plt.show

plt.xticks(fontsize = 30,rotation = 0)

plt.yticks(fontsize = 30)

plt.xlabel("Age",fontsize = 35)

plt.ylabel("Number of players",fontsize = 35)

plt.title("Number of players based on age",fontsize = 50)
b = df['Pos'].value_counts()
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

sns.countplot(x = "Pos",

              data = df,

              palette= "summer")

plt.show()

plt.xticks(fontsize = 30,rotation = 0)

plt.yticks(fontsize = 30)

plt.xlabel("Position",fontsize = 30)

plt.ylabel("Number of players",fontsize = 35)

plt.title("Number of players based on Position",fontsize = 50)
ycdf = df[df['CrdY']>=12].reset_index()

c = dict(zip(ycdf['Player'],ycdf['CrdY']))

c
ycdf.plot(x = 'Player', y = 'CrdY', kind = 'bar',color = 'y')

plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.xticks(fontsize = 30,rotation = 60)

plt.yticks(fontsize = 30)

plt.xlabel("Player",fontsize = 30)

plt.ylabel("Number of yellow cards",fontsize = 35)

plt.title("Number of yellow cards per player",fontsize = 50)

plt.show()
rcdf = df[df['CrdR']>=2].reset_index()

d = dict(zip(rcdf['Player'],rcdf['CrdR']))

d
rcdf.plot(x = 'Player', y = 'CrdR', kind = 'bar',color = 'r')

plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.xticks(fontsize = 30,rotation = 60)

plt.yticks(fontsize = 30)

plt.xlabel("Player",fontsize = 30)

plt.ylabel("Number of red cards",fontsize = 35)

plt.title("Number of red cards per player",fontsize = 50)

plt.show()
gsdf = df[df['Gls']>=10].reset_index()

e = dict(zip(gsdf['Player'],gsdf['Gls']))

e
gsdf.plot(x = 'Player', y = 'Gls', kind = 'bar',color = 'b')

plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.xticks(fontsize = 30,rotation = 60)

plt.yticks(fontsize = 30)

plt.xlabel("Player",fontsize = 30)

plt.ylabel("Number of goals",fontsize = 35)

plt.title("Number of goals per player",fontsize = 50)

plt.show()
asdf = df[df['Ast']>=7].reset_index()

f = dict(zip(asdf['Player'],asdf['Ast']))

f
asdf.plot(x = 'Player', y = 'Ast', kind = 'bar',color = 'g')

plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.xticks(fontsize = 30,rotation = 60)

plt.yticks(fontsize = 30)

plt.xlabel("Position",fontsize = 30)

plt.ylabel("Number of assists",fontsize = 35)

plt.title("Number of assists per player",fontsize = 50)

plt.show()
df[df['Starts'] >= 37]
xgdf = df[df['xG']>= 10]
xgdf.plot(x = 'Player', y = 'xG',kind = 'bar')

plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.xticks(fontsize = 30,rotation = 60)

plt.yticks(fontsize = 30)

plt.xlabel("Player",fontsize = 30)

plt.ylabel("Number of predicted goals",fontsize = 35)

plt.title("Number of predicted goals per player",fontsize = 50)

plt.show()