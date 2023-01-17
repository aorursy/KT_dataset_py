import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv('../input/liverpool-premier-league-champions-1920-complete/scores_and_fixtures.csv')

df.head()
#Teams which defeated Liverpool FC this season

df.Opponent[df.Result=='L'].unique()
#Teams which drew with Liverpool FC this season

df.Opponent[df.Result=='D'].value_counts()
#Teams which drew with Liverpool FC at home

df.Opponent[(df.Result=='D') & (df.Venue=='Home')]
x=[df.Result.value_counts()['W'],df.Result.value_counts()['D'],df.Result.value_counts()['L']]

labels=['Win', 'Draw', 'Loss']

explode=[0.05,0.05,0.05]

plt.pie(x=x, labels=labels, autopct='%1.2f%%', explode=explode)

plt.show()
plt.figure(figsize=(6,4))

sns.countplot('Result', hue='Venue', data=df)

plt.show()
plt.figure(figsize=(6,4))

sns.barplot('Result','Poss', hue='Venue', data=df)

plt.axhline(y=df.Poss.mean(),color='k', xmin=0,xmax=1)

plt.ylim(50,80)

plt.show()



print("Average Possession: {}%".format(round(df.Poss.mean(),2)))

print("Win : {}%".format(round(df.Poss[df.Result=='W'].mean(),2)))

print("Draw : {}%".format(round(df.Poss[df.Result=='D'].mean(),2)))

print("Lose : {}%".format(round(df.Poss[df.Result=='L'].mean(),2)))



print("Home : {}%".format(round(df.Poss[df.Venue=='Home'].mean(),2)))

print("Away : {}%".format(round(df.Poss[df.Venue=='Away'].mean(),2)))
plt.figure(figsize=(6,4))

plot=sns.barplot('Opponent','Poss', data=df)

plt.axhline(y=df.Poss.mean(),color='k', xmin=0,xmax=1, alpha=0.8, linestyle='--')

plt.axhline(y=50,xmin=0, xmax=1, color='red', alpha=0.8, linestyle='--')

plt.xticks(rotation=90)

plt.ylim(30,80)

plot.text(0,df.Poss.mean()+1,"Average Posssession")

plot.text(0,51,"50% Posssession")

plt.show()