import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelBinarizer

import graphviz 
from subprocess import call

sns.set()
%matplotlib inline
data = pd.read_csv("../input/smogon-6v6-pokemon-tiers/smogon.csv")
test = pd.read_csv("../input/collecting-test-set-pokemon/pokemontestset.csv")

data.head()
mapping = {'PU': 10, 'BL4': 15, 'NU': 20, 'BL3': 25, 'RU': 30, 'BL2': 35, 'UU': 40,'BL': 45, 'OU': 50, 'Uber': 60, 'AG': 70}

data.Tier.replace(to_replace=mapping, inplace=True)
#Grab the BST of the pokemon in each tier
UBER_BST = data[data['Tier'] == 60]['Total']
OU_BST = data[data['Tier'] == 50]['Total']
UU_BST = data[data['Tier'] == 40]['Total']
RU_BST = data[data['Tier'] == 30]['Total']
NU_BST = data[data['Tier'] == 20]['Total']
PU_BST = data[data['Tier'] == 10]['Total']

#Initialize matplotlib plot
fig = plt.figure(figsize=(24,10))
plt.title("BST Impact on Tier Usage")
sns.set_style("whitegrid")

#Graph all the BST
UBER_BST.hist(alpha = 0.7, bins = 30, label='UBER', density=True)
OU_BST.hist(alpha = 0.7, bins = 30, label='OU', density=True)
UU_BST.hist(alpha = 0.7, bins = 30, label='UU', density=True)
RU_BST.hist(alpha = 0.7, bins = 30, label='RU', density=True)
NU_BST.hist(alpha = 0.7, bins = 30, label='NU', density=True)
PU_BST.hist(alpha = 0.7, bins = 30, label='PU', density=True)

plt.legend(loc="upper right")
#Grabbing Tier Data for each pokemon type
NORMAL_TIER_DATA = data[data['Type.1'] == 'Normal']['Tier']
POISON_TIER_DATA = data[data['Type.1'] == 'Poison']['Tier']
PSYCHIC_TIER_DATA = data[data['Type.1'] == 'Psychic']['Tier']
GRASS_TIER_DATA = data[data['Type.1'] == 'Grass']['Tier']
GROUND_TIER_DATA = data[data['Type.1'] == 'Ground']['Tier']
ICE_TIER_DATA = data[data['Type.1'] == 'Ice']['Tier']
FIRE_TIER_DATA = data[data['Type.1'] == 'Fire']['Tier']
ROCK_TIER_DATA = data[data['Type.1'] == 'Rock']['Tier']
DRAGON_TIER_DATA = data[data['Type.1'] == 'Dragon']['Tier']
WATER_TIER_DATA = data[data['Type.1'] == 'Water']['Tier']
BUG_TIER_DATA = data[data['Type.1'] == 'Bug']['Tier']
DARK_TIER_DATA = data[data['Type.1'] == 'Dark']['Tier']
FIGHT_TIER_DATA = data[data['Type.1'] == 'Fighting']['Tier']
GHOST_TIER_DATA = data[data['Type.1'] == 'Ghost']['Tier']
STEEL_TIER_DATA = data[data['Type.1'] == 'Steel']['Tier']
FLYING_TIER_DATA = data[data['Type.1'] == 'Flying']['Tier']
ELECTRIC_TIER_DATA = data[data['Type.1'] == 'Electric']['Tier']
FAIRY_TIER_DATA = data[data['Type.1'] == 'Fairy']['Tier']

#Aggregating the data for boxplot use
AGGREGATE_DATA = [NORMAL_TIER_DATA, POISON_TIER_DATA, PSYCHIC_TIER_DATA, GRASS_TIER_DATA, GROUND_TIER_DATA, ICE_TIER_DATA, FIRE_TIER_DATA, ROCK_TIER_DATA,
                  DRAGON_TIER_DATA, WATER_TIER_DATA, BUG_TIER_DATA, DARK_TIER_DATA, FIGHT_TIER_DATA, GHOST_TIER_DATA, STEEL_TIER_DATA, FLYING_TIER_DATA, 
                  ELECTRIC_TIER_DATA, FAIRY_TIER_DATA]
xticklabels = ['Normal', 'Poison', 'Psychic', 'Grass', 'Ground', 'Ice', 'Fire', 'Rock', 'Dragon', 'Water', 'Bug', 'Dark', 'Fighting', 'Ghost', 'Steel', 'Flying', 'Electric', 'Fairy']

#Initializing matplotlib plot
fig2 = plt.figure(figsize=(24,10))
ax = fig2.add_subplot(111)
plt.title("Average Tier by Type")

#Set the labels for the plot
ax.set_yticklabels([' ', 'PU', 'NU', 'RU', 'UU', 'OU', 'Uber', 'AG'])
ax.set_xlabel('Pokemon Typing')

bp = ax.boxplot(AGGREGATE_DATA, patch_artist=True)

#Customization of the matplotlib plot using Seaborn API
for box in bp['boxes']:

    box.set( color='#4DE29F', linewidth=2)

    box.set( facecolor = '#4DE29F' )
    

for whisker in bp['whiskers']:
    whisker.set(color='#5F3FA8', linewidth=2)
    
for cap in bp['caps']:
    cap.set(color='#5F3FA8', linewidth=2)
    
for median in bp['medians']:
    median.set(color='#FF8F69', linewidth=2)

ax.set_xticklabels(xticklabels)
#Convert Primary and Secondary Typing into one-hot encoded arrays
#Issue: Duplicates types because there are two columns
dummies = data[['Type.1', 'Type.2']]

#Solution: Convert the columns into stacks. Crosstab is used to find the frequency of the type occuring in both stacks. Effectively returning one single merged one-hot-encoded dataset
stacked = dummies.stack()
index = stacked.index.get_level_values(0)
result = pd.crosstab(index=index, columns=stacked)
result.index.name = None
result.columns.name = None

#Concatinate the one-hot-encoded dataset onto the original one
data = pd.concat([data, result], axis=1 )

#Dropping the useless columns
data = data.drop(['Type.1', 'Type.2', 'Mega', 'Legendary', 'Generation'], axis=1)
#Cleaning dataset
data = data.rename(index=str, columns={"Sp..Atk": "SpAtk", "Sp..Def": "SpDef"})

data.head()
#Set minimum bucket size to 15.
c = DecisionTreeClassifier(min_samples_split=15)

features=["HP", 'Attack', 'Defense', 'SpAtk', 'SpDef', 'Speed', 'Bug', 'Dark', 'Electric', 'Fairy', 'Fighting', 'Fire', 'Flying', 'Ghost', 'Grass', 'Ground', 'Ice',
         'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water']
#Extract the training and testing set from the training and testing data
x_training = data[features]
y_training = data['Tier']

x_test = test[features]
y_test = test['Tier']
#Build decision tree model with training data
dt = c.fit(x_training, y_training)
#Draw out the decision tree
export_graphviz(dt, out_file='tree.dot')
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

plt.figure(figsize = (40, 40))
plt.imshow(plt.imread('tree.png'))
plt.axis('off');
plt.show();
y_pred = c.predict(x_test)
#Calculate the delta between the prediction and actual tier
delta = y_pred - y_test

delta = np.absolute(delta)
delta = delta.mean()
print(delta)