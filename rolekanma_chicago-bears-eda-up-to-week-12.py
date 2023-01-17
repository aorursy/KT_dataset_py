# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

plt.style.use('fivethirtyeight')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

pd.options.mode.chained_assignment = None



# Any results you write to the current directory are saved as output.
import pandas as pd

data = pd.read_csv("../input/nfl-data-up-to-week-12/Week12NFLData.csv")
data = data.loc[(data.play_type.isin(['pass','run', 'no_play'])) &

               data.epa.isna()==False]
data[data['play_type'] =='no_play']
data.loc[data.desc.str.contains('left end|left tackle|left guard|up the middle|right guard|right tackle|right end|rushes'),

'play_type'] = 'run'



data.loc[data.desc.str.contains('scrambles|sacked|pass'), 'play_type'] = 'pass'

data.reset_index(drop = True, inplace = True)
#Create a smaller dataframe with plays where rusher_player_name is null

rusher_nan = data.loc[(data['play_type'] == 'run') &

         (data['rusher_player_name'].isnull())]

         

#Create a list of the indexes/indices for the plays where rusher_player_name is null

rusher_nan_indices = list(rusher_nan.index)



for i in rusher_nan_indices:

    #Split the description on the blank spaces, isolating each word

    desc = data['desc'].iloc[i].split()

    #For each word in the play description

    for j in range(0,len(desc)):

        #If a word is right, up, or left

        if desc[j] == 'right' or desc[j] == 'up' or desc[j] == 'left':

            #Set rusher_player_name for that play to the word just before the direction

            data['rusher_player_name'].iloc[i] = desc[j-1]     

        else:

            pass

#Create a smaller dataframe with plays where passer_player_name is null

passer_nan = data.loc[(data['play_type'] == 'pass') &

         (data['passer_player_name'].isnull())]

#Create a list of the indexes/indices for the plays where passer_player_name is null

passer_nan_indices = list(passer_nan.index)



for i in passer_nan_indices:

    #Split the description on the blank spaces, isolating each word

    desc = data['desc'].iloc[i].split()

    #For each word in the play description

    for j in range(0,len(desc)):

        #If a word is pass

        if desc[j] == 'pass':

            data['passer_player_name'].iloc[i] = desc[j-1]            

        else:

            pass

#Change any backwards passes that incorrectly labeled passer_player_name as Backward

data.loc[data['passer_player_name'] == 'Backward', 'passer_player_name'] == float('NaN')



receiver_nan = data.loc[(data['play_type'] == 'pass') & 

                        (data['receiver_player_name'].isnull()) &

                        (data['desc'].str.contains('scrambles|sacked|incomplete')==False)]



receiver_nan_indices = list(receiver_nan.index)



for i in receiver_nan_indices:

    desc = data['desc'].iloc[i].split()



    for j in range(0,len(desc)):

        if (desc[j]=='left' or desc[j]=='right' or desc[j]=='middle') and (desc[j] != desc[-1]):

            if desc[j+1]=='to':

                data['receiver_player_name'].iloc[i] = desc[j+2]

        else:

            pass
data.loc[data['epa'] > 0, 'success'] = 1
bears_stats = data.loc[data['posteam'] == 'CHI']

bears_stats
plt.figure(1, figsize = (10,6))

plt.hist(bears_stats['epa'].loc[bears_stats['play_type'] == 'pass'], bins = 50, label = 'Pass', color = 'orange')

plt.hist(bears_stats['epa'].loc[bears_stats['play_type'] == 'run'], bins = 50, label = 'Run', alpha = .7, color = 'darkblue')

plt.xlabel('Expected Points Added')

plt.ylabel('Number of plays')

plt.title('EPA Distribution Based on Play Type - Bears 2019')

plt.text(6,50,'Data from nflscrapR', fontsize=10, alpha=.7)

#Will show the colors and labels of each histogram

plt.legend()

plt.show()
sns.lmplot(data = bears_stats, x = 'yardline_100', y = 'epa', 

                  fit_reg = False, hue = 'play_type',

                    height = 8, 

                  scatter_kws = {'s':200})

plt.show()
selected_column = ['pass_length', 'pass_location', 'run_location', 'run_gap', 'play_type']

for c in selected_column:

    print(bears_stats[c].value_counts(normalize=True).to_frame(), '\n')
plt.figure(figsize=(16,9))

sns.swarmplot(x= bears_stats['qtr'],

              y = bears_stats['yards_gained'],

              hue = bears_stats['play_type'])

plt.legend(loc='upper right')

plt.show()
nfcnorth = data.loc[(data['posteam']== 'CHI') | (data['posteam']== 'DET') | 

                   (data['posteam']== 'GB') | (data['posteam']== 'MIN')]
sns.barplot( x = 'posteam', y = 'epa', data = nfcnorth)
nfl_wrs = data.loc[(data['play_type']=='pass') & (data['down']<=4)].groupby(by='receiver_player_name')[['epa','success','yards_gained', 'td_prob', 'air_yards', 'yardline_100','qtr', 'yac_epa']].mean()
bears_wrs = data.loc[(data['posteam']=='CHI') & (data['play_type']=='pass') & (data['down']<=4)].groupby(by='receiver_player_name')[['epa','success','yards_gained', 'td_prob', 'air_yards', 'yardline_100','qtr', 'yac_epa']].mean()
#Add new column

nfl_wrs['attempts'] = data.loc[(data['play_type']=='pass') & 

                        (data['down']<=4)].groupby(by='receiver_player_name')['epa'].count()



#Sort by mean epa

nfl_wrs.sort_values('epa', ascending=False, inplace=True)



#Filter by attempts

nfl_wrs = nfl_wrs.loc[nfl_wrs['attempts'] > 10] 



#

bears_wrs['attempts'] = data.loc[(data['posteam']=='CHI') & (data['play_type']=='pass') & 

                        (data['down']<=4)].groupby(by='receiver_player_name')['epa'].count()



#Sort by mean epa

bears_wrs.sort_values('epa', ascending=False, inplace=True)



#Filter by attempts

bears_wrs = bears_wrs.loc[bears_wrs['attempts'] > 10] 

bears_wrs
plt.figure(figsize=(16,9))

plt.title('Receiver Usage for the Bears')

sns.countplot(bears_stats['receiver_player_name'])
plt.figure(figsize=(16,9))

plt.title('RunningBack Usage for the Bears')

sns.countplot(bears_stats['rusher_player_name'])
wr_table = pd.pivot_table(bears_stats, index = ['receiver_player_name'],

               columns = ['defteam'],

               values = ['epa'], aggfunc = [np.mean], fill_value = 0)        

wr_table
plt.figure(figsize=(16,9))

plt.rcParams['font.size'] = 10

bg_color = (0.88,0.85,0.95)

plt.rcParams['figure.facecolor'] = bg_color

plt.rcParams['axes.facecolor'] = bg_color

fig, ax = plt.subplots(1)

p = sns.heatmap(wr_table,

                cmap='coolwarm',

                annot=True,

                fmt=".1f",

                annot_kws={'size':16},

                ax=ax)

plt.xlabel('Vs Defense')

plt.ylabel('Receiver')

ax.set_ylim((0,15))

plt.text(3,13.3, "Bears WRs Heat Map", fontsize = 20, color='Black', fontstyle='italic')

plt.show()
bears_rbs_pivot = pd.pivot_table(bears_stats,

            index = ['rusher_player_name'],

            columns = ['run_location'],

            values = ['epa'], 

            aggfunc = [np.mean],

            fill_value = 0)
plt.figure(figsize=(16,9))

plt.rcParams['font.size'] = 10

bg_color = (0.88,0.85,0.95)

plt.rcParams['figure.facecolor'] = bg_color

plt.rcParams['axes.facecolor'] = bg_color

fig, ax = plt.subplots(1)

p = sns.heatmap(bears_rbs_pivot,

                cmap='Blues',

                annot=True,

                fmt=".1f",

                annot_kws={'size':18},

                ax=ax)

plt.xlabel('Run Direction')

plt.ylabel('Receiver')

ax.set_ylim((0,15))

plt.text(1,10, "Bears RBs Heat Map", fontsize = 20, color='Black', fontstyle='italic')

plt.show()
sns.set(style="white", context="talk")

plt.figure(figsize=(20,9))

sns.violinplot(data = data[data.posteam =='CHI'],

                   x ='receiver_player_name', y = 'epa')

plt.show()
bears_wrs.epa.mean()
nfl_wrs.epa.mean()
plt.figure(figsize=(16,9))

sns.set(style="white", context="talk")

sns.catplot(x="qtr", y='epa', hue="receiver_player_name", kind="swarm", data=bears_stats);

plt.show()