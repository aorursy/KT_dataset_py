import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('seaborn-whitegrid')



import warnings

warnings.filterwarnings('ignore')
dir = '/kaggle/input/league-of-legends-challenger-ranked-games2020/'

ranks = ['Challenger', 'GrandMaster', 'Master']



data = pd.DataFrame()



for rank in ranks:

    data_in = pd.read_csv(dir+rank+'_Ranked_Games.csv')

    data_in['Rank'] = rank

    print("Size of {}: {}".format(rank,data_in.shape))

    data = data.append(data_in, ignore_index=True)

    

print("Total size: {}".format(data.shape))
data.columns
data.head(3)
labels = ['Blue', 'Red']

sizes = [data['blueWins'].mean(),  data['redWins'].mean()]

colors = ['#63D1F4', '#EE6363']



fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%',

        colors=colors, shadow=True, startangle=140)

plt.title("Win Percentage by Side")

plt.show()
print("Average game length: {:.2f} minutes".format(data['gameDuraton'].mean()/60))

plt.figure(figsize=(20,10))

sns.distplot(data['gameDuraton']/60, hist=True, kde=False)

sns.set(font_scale = 2)

plt.xlabel('Duration (min)')

plt.ylabel('Number of Games')



plt.show()
categories = ['Wins', 'FirstBlood', 'FirstTower',

       'FirstBaron', 'FirstDragon', 'FirstInhibitor',

       'DragonKills', 'BaronKills', 'TowerKills',

       'InhibitorKills', 'WardPlaced', 'Wardkills', 'Kills',

       'Death', 'Assist', 'ChampionDamageDealt', 'TotalGold',

       'TotalMinionKills', 'TotalLevel', 'AvgLevel',

       'JungleMinionKills', 'KillingSpree', 'TotalHeal',

       'ObjectDamageDealt']



blue_percentage_data = {}



for category in categories:

    blue_total = data['blue'+category].sum()

    red_total = data['red'+category].sum()

    total = (blue_total + red_total).sum()

    

    blue_percent = blue_total/total

    red_percent = red_total/total

    

    blue_percentage_data[category] = blue_percent



blue_over = {k:v for k,v in blue_percentage_data.items() if (abs(v-0.5)>0.01) or k=="Wins"}

red_over = {k:1-v for k,v in blue_over.items()}
y = range(len(blue_over))

plt.figure(figsize=(20,10))

barWidth = 0.9

# blue

plt.bar(y, list(blue_over.values()), color='#0EBFE9', edgecolor='black', width=barWidth)

# red

plt.bar(y, list(red_over.values()), bottom=list(blue_over.values()), color='#cc0000', edgecolor='black', width=barWidth)



# Custom x axis

plt.xticks(y, blue_over.keys())

plt.ylim((0.45,0.55))

plt.ylabel("Percentage Taken")

plt.title("Differences in sides for values over 1%")

# Show graphic

plt.show()
blue_corr = data.drop('Rank', axis=1).corr()['blueWins'][:].sort_values(axis=0, ascending=False) 

red_corr = data.drop('Rank', axis=1).corr()['redWins'][:].sort_values(axis=0, ascending=False) 

# pearson method normalizes values for me

blue_corr.head()
# get correlations of 0.5 or more

corr_cols = [prop for prop,corr in blue_corr.iteritems() if abs(corr)>0.5]

plt.figure(figsize=(12,12))

sns.set(font_scale = 1)

sns.heatmap(data[corr_cols].corr(), annot=True)
# Blue wins to any correlation above 0.5



plt.figure(figsize=(2,7))

sns.heatmap(blue_corr[corr_cols].to_frame(), annot=True, cbar=False)
f, axes = plt.subplots(1, 2, figsize=(10, 7))



plt.figure(figsize=(2,7))

red_corra = [prop for prop,corr in blue_corr.iteritems() if 'red' in prop]

sns.heatmap(red_corr[red_corra].sort_values(axis=0, ascending=False).to_frame(), annot=True, cbar=False, ax=axes[0], cmap='Reds')



plt.figure(figsize=(2,7))

blue_corra = [prop for prop,corr in blue_corr.iteritems() if 'blue' in prop]

sns.heatmap(blue_corr[blue_corra].to_frame(), annot=True, cbar=False, ax=axes[1], cmap='Blues')



f.tight_layout(w_pad=8)



plt.show()