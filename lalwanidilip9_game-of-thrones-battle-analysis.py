import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib as plt

%matplotlib inline
battles = pd.read_csv("../input/battles.csv")

battles.head()
battles = battles[['name','year','attacker_king','defender_king','attacker_outcome']]

battles.head()
battles['outcome'] = (battles['attacker_outcome'] == 'win')*1

battles.head()
attack = pd.DataFrame(battles.groupby("attacker_king").size().sort_values())

attack = attack.rename(columns = {0:'Battle'})

attack.plot(kind='bar')
defend = pd.DataFrame(battles.groupby("defender_king").size().sort_values())

defend = defend.rename(columns = {0:'Battle'})

defend.plot(kind='bar')
pvt = battles.pivot_table(index='attacker_king',columns='defender_king',aggfunc='sum',values='outcome')

pvt
sns.heatmap(pvt,annot=True)

sns.plt.suptitle('attacker win')