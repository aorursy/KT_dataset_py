import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.manifold import TSNE

import operator

from collections import Counter

import itertools as it



# visualize the important characteristics of the dataset

import matplotlib.pyplot as plt

from matplotlib_venn import venn2

import pylab

%matplotlib inline



# Some options

pd.set_option("display.max_columns", 100)

pylab.rcParams['figure.figsize'] = (10, 5)

markers=('_', '+','v', 's', 'd', '*', '^')

color_map = {0:'blue', 1:'red', 2:'white', 3: 'cyan', 4:'green', 5:'yellow', 6:'magenta'}



# define some functions



def mapToTSNE(x_in):

    # scale extended features

    standard_scaler = StandardScaler()

    x_std = standard_scaler.fit_transform(x_in)



    # t-distributed Stochastic Neighbor Embedding (t-SNE) visualization

#     tsne = TSNE(n_components=2, random_state=0)

    tsne = TSNE(method="exact", random_state=0, n_iter=1000,learning_rate=400, 

                verbose=2, n_iter_without_progress=300)

    

    return tsne.fit_transform(x_std)



def plotTSNE(label, x_in, y_in, legendLoc="lower left"):

    

    y = y_in[label]

    y_frequencies = Counter(y.values)

    y_ratios = Counter({key: value / y.size for key, value in y_frequencies.items()})

    print(y_frequencies)

    print(y_ratios)



    # encode the class label

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(y)



    # do some python magic so the least frequent labels will be drawn in end

    classesSortedByFrequency = [sortedLabel[0:2] for sortedLabel in sorted([label + (y_ratios[label[1]],) 

                      for label in list(enumerate(label_encoder.classes_))], 

                 key=operator.itemgetter(2), reverse=True)]



    # scatter plot the sample points among the classes

    plt.figure()

    for idx, cl in classesSortedByFrequency:

        plt.scatter(x=x_in[y==idx,0], y=x_in[y==idx,1], c=color_map[idx], marker=markers[idx], label=cl)

    plt.xlabel('X in t-SNE')

    plt.ylabel('Y in t-SNE')

    plt.legend(loc=legendLoc, title=label)

    plt.title('t-SNE visualization for GoT characters')

    plt.show()
battles_df = pd.read_csv("../input/battles.csv")

characterDeaths_df = pd.read_csv("../input/character-deaths.csv")

characterPredictions_df = pd.read_csv("../input/character-predictions.csv")
battles_df.head()
characterDeaths_df.head()
characterPredictions_df.head()
intersection = list(set(characterPredictions_df['name']) & set(characterDeaths_df['Name']))

characterPredictions_noIntersect_df = characterPredictions_df[~characterPredictions_df['name'].isin(intersection)]

characterDeaths_noIntersect_df = characterDeaths_df[~characterDeaths_df['Name'].isin(intersection)]



venn2([set(characterDeaths_df['Name']), set(characterPredictions_df['name'])], set_labels=('characterDeaths', 'characterPredictions'))

plt.show()
bookCount = characterPredictions_df[['book1','book2','book3','book4','book5']].sum(1)

x = characterPredictions_df[['name', 'numDeadRelations']].copy()

x['bookCount'] = bookCount

x = x.set_index('name')

x.head()
y_all = characterPredictions_df[['name','male','book1','book2','book3','book4','book5',

                                 'isMarried','boolDeadRelations','popularity','isPopular','alive','isAlive']]

y_all = y_all.set_index('name')

y_all.head()
x_std_2d = mapToTSNE(x)
y_all.columns
labelsToPlot = ['isPopular', 'boolDeadRelations', 'isAlive', 'male', 'isMarried']

for label in labelsToPlot:

    plotTSNE(label, x_std_2d, y_all, legendLoc="upper right")
# to battle - get all them features!



battles_features = battles_df[

    ['attacker_outcome', 'major_death','major_capture', 

     'attacker_size', 'defender_size', 'attacker_commander', 'defender_commander',

     'battle_type']].copy()

battles_features['attackerCount'] = battles_df[['attacker_1','attacker_2','attacker_3','attacker_4']].notnull().sum(1)

battles_features['defenderCount'] = battles_df[['defender_1','defender_2','defender_3','defender_4']].notnull().sum(1)



battles_features['attackingCommanders'] = battles_df['attacker_commander'].map(lambda x: 0 if str(x) == "nan" else len(str(x).split(",")))

battles_features['defenderCommanders'] = battles_df['defender_commander'].map(lambda x: 0 if str(x) == "nan" else len(str(x).split(",")))

battles_features['attacker_outcome'] = battles_features['attacker_outcome'].map(lambda x: 1 if x == "win" else 0)

battles_features.head()
# generate new features and labels for characters

x_extended = x.copy()

y_all_extended = y_all.copy()



x_extended['witnessed_wins'] = 0

x_extended['witnessed_losses'] = 0

x_extended['witnessed_own_attacker_size_mean'] = 0

x_extended['witnessed_opponent_attacker_size_mean'] = 0

x_extended['witnessed_own_defender_size_mean'] = 0

x_extended['witnessed_opponent_defender_size_mean'] = 0

x_extended['witnessed_major_deaths'] = 0

x_extended['witnessed_major_capture'] = 0

x_extended['battleCountAsAttackerCommander'] = 0

x_extended['battleCountAsDefenderCommander'] = 0

x_extended['battleCountAsCommander'] = 0



y_all_extended['hadMoreWinsThanLosses'] = 0

y_all_extended['wasAttackerCommander'] = 0

y_all_extended['wasDefenderCommander'] = 0

y_all_extended['wasCommander'] = 0

y_all_extended['preferredBattleType'] = "None"





for characterName in x_extended.index:

    

    character = x_extended.loc[characterName].copy()

    character_labels = y_all_extended.loc[characterName].copy()

    

    witnessed_own_attacker_size = 0

    witnessed_opponent_attacker_size = 0

    witnessed_own_defender_size = 0

    witnessed_opponent_defender_size = 0

    battleTypes = []

    

    for row in battles_features.iterrows():

        battle = row[1] 

        

        wasAttackerCommander = pd.notnull(battle['attacker_commander']) and characterName in battle['attacker_commander']

        wasDefenderCommander = pd.notnull(battle['defender_commander']) and characterName in battle['defender_commander']



        if (wasAttackerCommander or wasDefenderCommander):

            character['battleCountAsCommander'] += 1

            character['witnessed_wins'] += battle['attacker_outcome'] if wasAttackerCommander else (1-battle['attacker_outcome'])

            character['witnessed_losses'] += (1-battle['attacker_outcome'] if wasAttackerCommander else battle['attacker_outcome'])

    

            if (pd.notnull(battle['attacker_size'])):

                if (wasAttackerCommander):

                    witnessed_own_attacker_size += battle['attacker_size']

                else:

                    witnessed_opponent_attacker_size += battle['attacker_size']

            if (pd.notnull(battle['defender_size'])):

                if (wasDefenderCommander):

                    witnessed_own_defender_size += battle['defender_size']

                else:

                    witnessed_opponent_defender_size += battle['defender_size']

                    

            if(pd.notnull(battle['major_death'])):

                character['witnessed_major_deaths'] += battle['major_death']

            if(pd.notnull(battle['major_capture'])):

                character['witnessed_major_capture'] += battle['major_capture']

               

            character['battleCountAsAttackerCommander'] += wasAttackerCommander

            character['battleCountAsDefenderCommander'] += wasDefenderCommander

            if (pd.notnull(battle['battle_type'])):

                battleTypes.append(battle['battle_type'])

            

    # aggregate data for character

    if (character['battleCountAsAttackerCommander'] > 0):

        character_labels['wasAttackerCommander'] = 1

        character['witnessed_own_attacker_size_mean'] = witnessed_own_attacker_size / character['battleCountAsAttackerCommander']

        character['witnessed_opponent_defender_size_mean'] = witnessed_opponent_defender_size / character['battleCountAsAttackerCommander']

    

    if (character['battleCountAsDefenderCommander'] > 0):

        character_labels['wasDefenderCommander'] = 1

        character['witnessed_own_defender_size_mean'] = witnessed_own_defender_size / character['battleCountAsDefenderCommander']

        character['witnessed_opponent_attacker_size_mean'] = witnessed_opponent_attacker_size / character['battleCountAsDefenderCommander']

        

    if (character['battleCountAsCommander'] > 0):

        character_labels['hadMoreWinsThanLosses'] = int(character['witnessed_wins'] > character['witnessed_losses'])

        character_labels['wasCommander'] = 1

        battleType_frequencies = Counter(battleTypes)

        character_labels['preferredBattleType'] = max(battleType_frequencies,key=operator.itemgetter(1))

    

    x_extended.loc[characterName] = character

    y_all_extended.loc[characterName] = character_labels
x_extended.sort_values('battleCountAsCommander',ascending=False).head()
y_all_extended.sort_values('wasCommander',ascending=False).head()
x_extended_std_2d = mapToTSNE(x_extended)
y_all_extended.columns
labelsToPlot = ['isPopular', 'boolDeadRelations', 'isAlive', 'male', 'isMarried', 'wasCommander']

for label in labelsToPlot:

    plotTSNE(label, x_extended_std_2d, y_all_extended)
## From above, taking only commanders
y_all_extended.columns
filterLabel = "wasCommander"

filterValue = 1

x_filtered_std_2d = np.array(list(it.compress(x_extended_std_2d, y_all_extended[filterLabel] == filterValue)))

y_filtered = y_all_extended.loc[y_all_extended[filterLabel].values == filterValue]
labelsToPlot = ['isPopular', 'boolDeadRelations', 'isAlive', 'wasAttackerCommander', 'wasDefenderCommander', 'hadMoreWinsThanLosses',

                'male', 'isMarried', 'preferredBattleType']

for label in labelsToPlot:

    plotTSNE(label, x_filtered_std_2d, y_filtered)
x_commanders = x_extended.loc[y_all_extended['wasCommander'].values == 1]

y_commanders = y_all_extended.loc[y_all_extended['wasCommander'].values == 1]

x_commanders.head()
x_commanders_std_2d = mapToTSNE(x_commanders)
y_commanders.columns
labelsToPlot = ['isPopular', 'isAlive', 'wasAttackerCommander', 'wasDefenderCommander', 'hadMoreWinsThanLosses',

                'male', 'isMarried', 'preferredBattleType']

for label in labelsToPlot:

    plotTSNE(label, x_commanders_std_2d, y_commanders)