# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/fifa19/data.csv', encoding='utf-8')

df.head()
df = df.drop(columns=['Unnamed: 0'])

df.keys()
df.shape
for_age = df.copy()



plt.figure(figsize=(13,7))

plt.title("HOW OLD?")

sns.distplot(for_age['Age'])
for_overall = df.copy()



plt.figure(figsize=(13,7))

plt.title("TALENTS DIST")

sns.distplot(for_overall['Overall'])
for_money = df.copy()



for_money['Value'].head()
ending = [x['Value'][-1] for i, x in for_money.iterrows()]

set(ending)
def value_cal(num):

    if num[-1] == '0':

        return 0

    elif num[-1] == 'K':

        return int(num[1:-1])

    else:

        return int(float(num[1:-1])*1000)
for_money['Money_Value'] = for_money['Value'].apply(value_cal)



plt.figure(figsize=(15,8))

plt.title("CUANTO CUESTA?")

sns.distplot(for_money['Money_Value'])
bigones = for_money.sort_values(by='Money_Value', ascending=False)[['Name', 'Money_Value']][:15]

bigones
df.columns
stats = df[['Crossing',

       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

       'Marking', 'StandingTackle', 'SlidingTackle']].copy()
f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(stats.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
talent_hub = dict()

stats_ = stats.corr()

for ind, val in stats_.iterrows():

    talent_hub[ind] = val.sum()

    

talent_hub
[k for k,v in talent_hub.items() if v == max(talent_hub.values())][0]
[k for k,v in talent_hub.items() if v == min(talent_hub.values())][0]
for_money.columns
stats.columns
money = for_money.iloc[:500].copy()
print("For World Top 500 Players: ")

print()



for col in stats.columns:

    print(" {0}'s correlation with value: ".format(col), "{0:.2f}".format(np.corrcoef(np.array(money[col]), np.array(money['Money_Value']))[0][1]))