# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from subprocess import check_output
print(check_output(["ls", "../input/cleanedpokemon"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
my_data = pd.read_csv('../input/cleanedpokemon/pokemon_clean.csv')

all_data = pd.read_csv('../input/pokemon/pokemon.csv')

update_data = pd.read_csv('../input/final-pokemon/pokemon_clean _update.csv')

pd.set_option('display.max_columns', None)
#my_data.shape #Dimensions of dataset

my_data.dtypes

#print(my_data)
# pd.set_option('display.max_columns', None)

print(my_data.head(9)) #Look at first 9 pokemon, but is missing some stats?
missing_val = all_data.columns[all_data.isnull().any()].tolist() #defining missing_val
#checking for meaningful missing data entries

for col in missing_val:
    print("%s : %d" % (col,all_data[col].nunique()))
print( my_data.describe() ) #Flying, ice, rock are weakest out of all 802 pokemon. 

#filling in blanks for height and weight so no NaN variables
all_data['height_m'].fillna(np.int(0), inplace=True)
all_data['weight_kg'].fillna(np.int(0), inplace=True)

#Strongest so far are normal, dragon, and psychic in that order would be poison or bug but they do not have good pokemon
#baseStats = ('attack' + 'defense' + 'hp' + 'sp_attack' + 'sp_defense' + 'speed')
mod_attack = my_data['attack'] * 1.5
mod_sp_attack = my_data['sp_attack'] * 1.5
mod_speed = my_data['speed'] * 2

my_data['base_stats'] = my_data['attack'] + my_data['defense'] + my_data['hp'] + my_data['sp_attack'] + my_data['sp_defense'] + my_data['speed']
my_data['off_stats'] = my_data['attack'] + my_data['sp_attack'] + my_data['speed']  #Offensive stats, first hit and killing with that hit
my_data['def_stats'] = my_data['defense'] + my_data['hp'] + my_data['sp_defense'] #Defensive stats, surviving the hit
my_data['mod_stats'] = mod_attack + my_data['defense'] + my_data['hp'] + mod_sp_attack + my_data['sp_defense'] + mod_speed
#df['variance'] = df['budget'] + df['actual']  # assigned to a column
#print(my_data.loc[[2,5,8]])
#print(my_data.loc[[0,1,2,3,4,5,6,7,8]])

#df = my_data.loc[[0,1,2,3,4,5,6,7,8]]

print(my_data.loc[[2,5,8, 153, 156, 159, 253, 256, 259, 388, 391, 394, 496, 499, 502, 651, 654, 657, 723, 726, 729]])

df = my_data.loc[[2,5,8, 153, 156, 159, 253, 256, 259, 388, 391, 394, 496, 499, 502, 651, 654, 657, 723, 726, 729]] #pikachu is 24

generation_1 = my_data.iloc [0:151]
generation_2 = my_data.iloc [152:251]
generation_3 = my_data.iloc [252:386]
generation_4 = my_data.iloc [387:493]
generation_5 = my_data.iloc [494:649]
generation_6 = my_data.iloc [650:721]
generation_7 = my_data.iloc [722:800]

legendaries = my_data.loc[[143,144,145,149,150,242,243,244,248,249,250,376,377,378,379,380,381,382,383,384,385,479,480,481,482,483,484,485,486,487,489,490,491,492,493,637,638,639,640,641,642,643,644,645,646,647,648,715,716,717,718,719,720,784,785,786,787,788,789,790,791,792,793,794,795,796,797,798,799,800]] 
#manually input pokedex numbers of all legendaries since couldn't work indexer.

legendaries2 = all_data.loc[[143,144,145,149,150,242,243,244,248,249,250,376,377,378,379,380,381,382,383,384,385,479,480,481,482,483,484,485,486,487,489,490,491,492,493,637,638,639,640,641,642,643,644,645,646,647,648,715,716,717,718,719,720,784,785,786,787,788,789,790,791,792,793,794,795,796,797,798,799,800]]

bsgeneration_1 =my_data['base_stats'].iloc [0:151]
bsgeneration_2 =my_data['base_stats'].iloc [152:252]
bsgeneration_3 =my_data['base_stats'].iloc [252:386]
bsgeneration_4 =my_data['base_stats'].iloc [387:493]
bsgeneration_5 =my_data['base_stats'].iloc [494:649]
bsgeneration_6 =my_data['base_stats'].iloc [650:721]
bsgeneration_7 =my_data['base_stats'].iloc [722:800]

absgeneration_1 =statistics.mean(bsgeneration_1)
absgeneration_2 =statistics.mean(bsgeneration_2)
absgeneration_3 =statistics.mean(bsgeneration_3)
absgeneration_4 =statistics.mean(bsgeneration_4)
absgeneration_5 =statistics.mean(bsgeneration_5)
absgeneration_6 =statistics.mean(bsgeneration_6)
absgeneration_7 =statistics.mean(bsgeneration_7)


#Photoshop top pokemon with their icon.
#Generation I and III have the strongest Starter Pokemon on average because of their mega evolutions. Greninja wins out still!

pokedex_num = df.pokedex_number #define for bargraph
pokedex_name = df.name #define for bargraph
base = df.base_stats #define for bargraph
pokedex_number = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21] #1-3 = Generation I, 4-6 = Generation II, 7-9 = Generation III, 10-12 = Generation IV, 13-15 = Generation V, 16-18 = Generation VI, 19-21 = Generation VII
generation_tick = [2, 5, 8, 11, 14, 17, 20]


generations = ('I', 'II', 'III', 'IV', 'V', 'VI', 'VII')
# y_pos = np.arange(len(generations))

#plt.axis([0,25,500,650]) #Zoom in on data
plt.ylim(500,650) #zoom in on data
barlist = plt.bar(pokedex_number, base, width=0.7) #(x, y, width of columns, color)
plt.ylabel('Base Stats')
plt.xlabel('Generation Number')
plt.title('All Starter Pokemon and their base stats (Gen I-VII)')
plt.xticks(generation_tick, generations)
#barlist[0].set_color('g')
#barlist[1].set_color('r')

x = 0
list_of_lists = [ [0], [3], [6], [9], [12], [15], [18]]
for list in list_of_lists:
    for x in list:
        barlist[x].set_color('g')

list_of_lists = [ [1], [4], [7], [10], [13], [16], [19]]
for list in list_of_lists:
    for x in list:
        barlist[x].set_color('r')


#barlist = plt.bar(pokedex_name, base, width=0.5) #(x, y, width of columns, color)
#barlist = plt.bar(pokedex_num, base, width=100) #(x, y, width of columns, color)
# barlist[10].set_color('r')
# barlist[20].set_color('y')
# my_data['off_stats'] = my_data['attack'] + my_data['sp_attack'] + my_data['speed']  #Offensive stats, first hit and killing with that hit
# my_data['def_stats'] = my_data['defense'] + my_data['hp'] + my_data['sp_defense'] #Defensive stats, surviving the hit
#base = df.base_stats #define for bargraph


off = df.off_stats
defense = df.def_stats

barlist1 = plt.bar(pokedex_number, off, width=0.7) #(x, y, width of columns, color)
plt.ylim(200, 450)
plt.ylabel('Offensive Stats')
plt.title('All Starter Pokemon and their offensive stats (Gen I-VII)')
plt.xlabel('Generation Number')
plt.xticks(generation_tick, generations)



x = 0
list_of_lists = [ [0], [3], [6], [9], [12], [15], [18]]
for list in list_of_lists:
    for x in list:
        barlist1[x].set_color('g')

list_of_lists = [ [1], [4], [7], [10], [13], [16], [19]]
for list in list_of_lists:
    for x in list:
        barlist1[x].set_color('r')

#Generation I, II, IV, and VI have the best starters and best defensive stats. Only Swampert is able to compete because of his extreme defense and movesets.

barlist2 = plt.bar(pokedex_number, defense, width=0.7) #(x, y, width of columns, color)
plt.ylim(150, 350)
plt.ylabel('Defensive Stats')
plt.title('All Starter Pokemon and their defensive stats (Gen I-VII)')
plt.xlabel('Generation Number')
plt.xticks(generation_tick, generations)

x = 0
list_of_lists = [ [0], [3], [6], [9], [12], [15], [18]]
for list in list_of_lists:
    for x in list:
        barlist2[x].set_color('g')

list_of_lists = [ [1], [4], [7], [10], [13], [16], [19]]
for list in list_of_lists:
    for x in list:
        barlist2[x].set_color('r')
mod = df.mod_stats
plt.ylim(650,950) #zoom in on data
modlist = plt.bar(pokedex_number, mod, width=0.7) #(x, y, width of columns, color)
plt.ylabel('Modded Stats')
plt.xlabel('Generation Number')
plt.title('All Starter Pokemon and their modded stats (Gen I-VII)')
plt.xticks(generation_tick, generations)


x = 0
list_of_lists = [ [0], [3], [6], [9], [12], [15], [18]]
for list in list_of_lists:
    for x in list:
        modlist[x].set_color('g')

list_of_lists = [ [1], [4], [7], [10], [13], [16], [19]]
for list in list_of_lists:
    for x in list:
        modlist[x].set_color('r')
#HP, attack, sp_attack, defense, sp_defense, and speed

print('Highest Base Stats Starter Pokemon: \t \t'+ "\033[1m" + '{}'.format(df.name[df['base_stats'].idxmax()] ) + "\033[0;0m")
#print(legendaries['base_stats'].max())
print('Lowest Base Stats Starter Pokemon: \t \t'+ "\033[1m" + '{}'.format(df.name[df['base_stats'].idxmin()] ) + "\033[0;0m")
print('\n')

#Printing stat with tabs to format uniformly and added bold to better see pokemon name.
print('Highest HP Starter Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(df.name[df['hp'].idxmax()] ) + "\033[0;0m")
print('Lowest HP Starter Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(df.name[df['hp'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Attack Starter Pokemon: \t \t'+ "\033[1m" + '{}'.format(df.name[df['attack'].idxmax()] ) + "\033[0;0m")
print('Lowest Attack Starter Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(df.name[df['attack'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Special Attack Starter Pokemon: \t'+ "\033[1m" + '{}'.format(df.name[df['sp_attack'].idxmax()] ) + "\033[0;0m")
print('Lowest Special Attack Starter Pokemon: \t \t' + "\033[1m" + '{}'.format(df.name[df['sp_attack'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Defense Starter Pokemon: \t \t'+ "\033[1m" + '{}'.format(df.name[df['defense'].idxmax()] ) + "\033[0;0m")
print('Lowest Defense Starter Pokemon: \t \t'+ "\033[1m" + '{}'.format(df.name[df['defense'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Special Defense Starter Pokemon: \t'+ "\033[1m" + '{}'.format(df.name[df['sp_defense'].idxmax()] ) + "\033[0;0m")
print('Lowest Special Defense Starter Pokemon: \t'+ "\033[1m" + '{}'.format(df.name[df['sp_defense'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Fastest Starter Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(df.name[df['speed'].idxmax()] ) + "\033[0;0m")
print('Slowest Starter Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(df.name[df['speed'].idxmin()] ) + "\033[0;0m")
print('\n')
ax_hp = sns.distplot(df['hp'], color="pink")
ax_attack = sns.distplot(df['attack'], color="red")
ax_spattack = sns.distplot(df['sp_attack'], color="purple")
ax_attack = sns.distplot(df['attack'], color="red")
ax_spattack = sns.distplot(df['sp_attack'], color="purple")
ax_defense = sns.distplot(df['defense'], color="#4b7d4b")
ax_spdefense = sns.distplot(df['sp_defense'], color="#6cb46c")
ax_defense = sns.distplot(df['defense'], color="#4b7d4b")
ax_spdefense = sns.distplot(df['sp_defense'], color="#6cb46c")
ax_speed = sns.distplot(df['speed'], color="orange")

ax_base = sns.distplot(df['base_stats'], color="blue")
cols = df.columns
against_ = []
for col in cols:
    if ('against_' in str(col)):
        against_.append(col)
        
print(len(against_)) 
print(against_)
del list #list is called earlier and was messing with apending a new list, probably for loop used for coloring

unique_elem = []
for col in against_:
    unique_elem.append(df[col].unique().tolist())
    
result = set(x for l in unique_elem for x in l)

result = list(result)
print(result)
for col in against_:
    if (np.mean(df[col]) > 1.2):
        print(col)

for col in against_:
    if (np.sum(df[col]) > 1000):
        print(col)
import random

for col in range(0, len(against_)):
    print (against_[col])
    print (df[against_[col]].unique())
    pp = pd.value_counts(df[against_[col]])
    
    color = ['g', 'b', 'r', 'y', 'pink', 'orange', 'brown']
            
    pp.plot.bar(color = random.choice(color))
    plt.show()
t2list = df['type2'].value_counts().plot.bar()

plt.ylabel('Count') #total of 18, 3 starters with no secondary type
plt.xlabel('Pokemon type')
plt.title('All Starter Pokemon and their secondary type (Gen I-VII)')


# my_colors = 'rgbkymc'

# newlist = plt.bar('type2', count, width=0.7, color = my_colors) #(x, y, width of columns, color)
#t2list[1].set_color('m')
#plt.show
#Same graph above but using seaborne package
t2 = pd.value_counts(df['type2'])

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("whitegrid")

ax = sns.barplot(x=t2.index, y=t2, data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 12)
ax.set(xlabel='Secondary types', ylabel='Count')
ax.set_title('Distribution of Secondary Pokemon type')
print(df.describe()) #Summary stats for all starters
single_types = []
dual_types = []

count = 0
for i in my_data.index:
    if(pd.isnull(my_data.type2[i]) == True):
        count += 1
        single_types.append(my_data.name[i])
    else:
        dual_types.append(my_data.name[i])
print('Single Types')
print(len(single_types))
print('\nDual Types')
print(len(dual_types))

data = [417, 384]
colors = ['lightblue', '#ff4c4c']

# Create a pie chart
plt.pie(data, 
        labels= ['Dual type', 'Single type'], 
        shadow=True, 
        colors=colors, 
        explode=(0, 0.15), 
        startangle=90, 
        autopct='%1.1f%%')

# View the plot drop above
plt.axis('equal')
plt.title('Dual vs Single type Pokemon')
# View the plot
plt.tight_layout()
plt.show()
ax_height = sns.distplot(all_data['height_m'], color="y")
ax = sns.pointplot(all_data['height_m'], color = 'g')
#Playing around trying to figure out a way to correlate base stats with height and/or weight

all_data.base_total.corr(all_data.weight_kg)
all_data.base_total.corr(all_data.height_m)
ax_weight = sns.distplot(all_data['weight_kg'], color="r")
ax = sns.pointplot(all_data['weight_kg'])
for col in against_:
    if (np.mean(my_data[col]) > 1.2):
        print(col)

for col in against_:
    if (np.sum(my_data[col]) > 1000):
        print(col)  
import random

for col in range(0, len(against_)):
    print (against_[col])
    print (my_data[against_[col]].unique())
    pp = pd.value_counts(my_data[against_[col]])
    
    color = ['g', 'b', 'r', 'y', 'pink', 'orange', 'brown']
            
    pp.plot.bar(color = random.choice(color))
    plt.show()
yy = pd.value_counts(all_data['capture_rate'])

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("whitegrid")

ax = sns.barplot(x=yy.index, y=yy, data=all_data)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 12)
ax.set(xlabel='Capture_rate', ylabel='Number of Pokemon')
ax.set_title('Distribution of capture_rate against number of Pokemon')
yy = pd.value_counts(all_data['base_happiness'])

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("whitegrid")

ax = sns.barplot(x=yy.index, y=yy, data=all_data)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 12)
ax.set(xlabel='Base Happiness', ylabel='Number of Pokemon')
ax.set_title('Distribution of Base Happiness against number of Pokemon')
yy = pd.value_counts(my_data['type1'])

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("whitegrid")

ax = sns.barplot(x=yy.index, y=yy, data=my_data)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 12)
ax.set(xlabel='Primary types', ylabel='Count')
ax.set_title('Distribution of Primary Pokemon type')
yy = pd.value_counts(my_data['type2'])

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("whitegrid")

ax = sns.barplot(x=yy.index, y=yy, data=my_data)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 12)
ax.set(xlabel='Secondary types', ylabel='Count')
ax.set_title('Distribution of Secondary Pokemon type')
yy = pd.value_counts(my_data['base_stats'])

fig, ax = plt.subplots()
fig.set_size_inches(42, 8.27)
sns.set_style("whitegrid")

ax = sns.barplot(x=yy.index, y=yy, data=my_data)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 12)
ax.set(xlabel='Base Stats', ylabel='Number of Pokemon')
ax.set_title('Distribution of Base Stats against number of Pokemon')
#HP, attack, sp_attack, defense, sp_defense, and speed

print('Highest Base Stats Pokemon: \t \t'+ "\033[1m" + '{}'.format(my_data.name[my_data['base_stats'].idxmax()] ) + "\033[0;0m")
#print(legendaries['base_stats'].max())
print('Lowest Base Stats Pokemon: \t \t'+ "\033[1m" + '{}'.format(my_data.name[my_data['base_stats'].idxmin()] ) + "\033[0;0m")
print('\n')

#Printing stat with tabs to format uniformly and added bold to better see pokemon name.
print('Highest HP Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(my_data.name[my_data['hp'].idxmax()] ) + "\033[0;0m")
print('Lowest HP Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(my_data.name[my_data['hp'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Attack Pokemon: \t \t'+ "\033[1m" + '{}'.format(my_data.name[my_data['attack'].idxmax()] ) + "\033[0;0m")
print('Lowest Attack Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(my_data.name[my_data['attack'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Special Attack Pokemon: \t'+ "\033[1m" + '{}'.format(my_data.name[my_data['sp_attack'].idxmax()] ) + "\033[0;0m")
print('Lowest Special Attack Pokemon: \t \t' + "\033[1m" + '{}'.format(my_data.name[my_data['sp_attack'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Defense Pokemon: \t \t'+ "\033[1m" + '{}'.format(my_data.name[my_data['defense'].idxmax()] ) + "\033[0;0m")
print('Lowest Defense Pokemon: \t \t'+ "\033[1m" + '{}'.format(my_data.name[my_data['defense'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Special Defense Pokemon: \t'+ "\033[1m" + '{}'.format(my_data.name[my_data['sp_defense'].idxmax()] ) + "\033[0;0m")
print('Lowest Special Defense Pokemon: \t'+ "\033[1m" + '{}'.format(my_data.name[my_data['sp_defense'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Fastest Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(my_data.name[my_data['speed'].idxmax()] ) + "\033[0;0m")
print('Slowest Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(my_data.name[my_data['speed'].idxmin()] ) + "\033[0;0m")
print('\n')
ax_hp = sns.distplot(my_data['hp'], color="pink")
ax_attack = sns.distplot(my_data['attack'], color="red")
ax_spattack = sns.distplot(my_data['sp_attack'], color="purple")
ax_attack = sns.distplot(my_data['attack'], color="red")
ax_spattack = sns.distplot(my_data['sp_attack'], color="purple")
#Pokemon tend to have higher attack on average.
ax_defense = sns.distplot(my_data['defense'], color="#4b7d4b")
ax_spdefense = sns.distplot(my_data['sp_defense'], color="#6cb46c")
ax_defense = sns.distplot(my_data['defense'], color="#4b7d4b")
ax_spdefense = sns.distplot(my_data['sp_defense'], color="#6cb46c")
ax_speed = sns.distplot(my_data['speed'], color="orange")
ax_base = sns.distplot(my_data['base_stats'], color="blue")
sns.lmplot( x="weight_kg", y="height_m", data=all_data, fit_reg=False)
g = sns.jointplot("attack", "hp", data=my_data, kind="kde")
update_data['weakness_count'] =  update_data['against_flying'] + update_data['against_ghost'] + update_data['against_ground'] + update_data['against_grass'] + update_data['against_ice'] + update_data['against_normal'] + update_data['against_poison'] + update_data['against_psychic'] +  update_data['against_rock'] + update_data['against_steel'] + update_data['against_water'] + update_data['against_bug'] + update_data['against_fire'] + update_data['against_dragon'] +  update_data['against_dark'] + update_data['against_electric'] + update_data['against_fairy'] + update_data['against_fight']

#redefine base and mod stats for Marco's sections
update_data['base_stats'] = update_data['attack'] + update_data['defense'] + update_data['hp'] + update_data['sp_attack'] + update_data['sp_defense'] + update_data['speed']
update_data['mod_stats'] = (update_data['attack'] * 1.5) + update_data['defense'] + update_data['hp'] + (update_data['sp_attack'] * 1.5) + update_data['sp_defense'] + (update_data['speed'] * 2)

#weakness 0.0, .25, 1, 0.5, 2, 4.0
update_data['mod_stats_weakness'] = (update_data['attack'] * 1.5) + update_data['defense'] + update_data['hp'] + (update_data['sp_attack'] * 1.5) + update_data['sp_defense'] + (update_data['speed'] * 2)

update_data = update_data.replace(np.nan,' ',regex=True)

# We organize the data by generation
pokemon_list = []
generation_list = []
i = 0
gen_count = 1

# Lets get the max weakness and least weakness.
max_weakness = update_data['weakness_count'].max();
min_weakness = update_data['weakness_count'].min();

# Process for organizing pokemon by generation and filtering out the legendary ones
while i < len(update_data):
    if  update_data['name'][i] is not None:
        if update_data['generation'][i] == gen_count:
# Add pokemon if not legendary
            if update_data['is_legendary'][i] == 0:
                # We now use affine transformation in order to create a weakness rating system from 1-10
                weakness_count = (update_data['weakness_count'][i] - min_weakness) * (10 - 1)/ (max_weakness - min_weakness) + 1
                
                # Dealing with pokemon weaknesses
                base_weakness = update_data['mod_stats_weakness'][i]
                big_impact = 10
                small_impact = 5
                
                # Check for weaknesses and reduce base stats if weakness is > 2
                if update_data['against_bug'][i] == 4: 
                    base_weakness -= big_impact
                elif update_data['against_bug'][i] == 2:
                    base_weakness -= small_impact
                if update_data['against_dark'][i] == 4: 
                    base_weakness -= big_impact
                elif update_data['against_dark'][i] == 2:
                    base_weakness -= small_impact
                if update_data['against_dragon'][i] == 4: 
                    base_weakness -= big_impact
                elif update_data['against_dragon'][i] == 2:
                    base_weakness -= small_impact
                if update_data['against_electric'][i] == 4: 
                    base_weakness -= big_impact
                elif update_data['against_electric'][i] == 2:
                    base_weakness -= small_impact
                if update_data['against_fairy'][i] == 4: 
                    base_weakness -= big_impact
                elif update_data['against_fairy'][i] == 2:
                    base_weakness -= small_impact
                if update_data['against_fight'][i] == 4: 
                    base_weakness -= big_impact
                elif update_data['against_fight'][i] == 2:
                    base_weakness -= small_impact
                if update_data['against_flying'][i] == 4: 
                    base_weakness -= big_impact
                elif update_data['against_flying'][i] == 2:
                    base_weakness -= small_impact
                if update_data['against_ghost'][i] == 4: 
                    base_weakness -= big_impact
                elif update_data['against_ghost'][i] == 2:
                    base_weakness -= small_impact
                if update_data['against_grass'][i] == 4: 
                    base_weakness -= big_impact
                elif update_data['against_grass'][i] == 2:
                    base_weakness -= small_impact
                if update_data['against_ground'][i] == 4: 
                    base_weakness -= big_impact
                elif update_data['against_ground'][i] == 2:
                    base_weakness -= small_impact
                if update_data['against_ice'][i] == 4: 
                    base_weakness -= big_impact
                elif update_data['against_ice'][i] == 2:
                    base_weakness -= small_impact
                if update_data['against_rock'][i] == 4: 
                    base_weakness -= big_impact
                elif update_data['against_rock'][i] == 2:
                    base_weakness -= small_impact
                if update_data['against_normal'][i] == 4: 
                    base_weakness -= big_impact
                elif update_data['against_normal'][i] == 2:
                    base_weakness -= small_impact
                if update_data['against_psychic'][i] == 4: 
                    base_weakness -= big_impact
                elif update_data['against_psychic'][i] == 2:
                    base_weakness -= small_impact
                if update_data['against_poison'][i] == 4: 
                    base_weakness -= big_impact
                elif update_data['against_poison'][i] == 2:
                    base_weakness -= small_impact
                if update_data['against_steel'][i] == 4: 
                    base_weakness -= big_impact
                elif update_data['against_steel'][i] == 2:
                    base_weakness -= small_impact
                if update_data['against_water'][i] == 4: 
                    base_weakness -= big_impact
                elif update_data['against_water'][i] == 2:
                    base_weakness -= small_impact

                # Store single pokemon record in dictionary and append to out generation_list array
                pokemon = {'name': update_data['name'][i], 'base stats': update_data['base_stats'][i], 'weakness rating': round(weakness_count, 1), 'type': str(update_data['type1'][i]) +" "+ str(update_data['type2'][i]), 'mod stats': update_data['mod_stats'][i],'true base stats': base_weakness, 'special attack': update_data['sp_attack'][i], 'attack': update_data['attack'][i]}
                generation_list.append(pokemon)
            
            if i == 800:
                tmp_pokemon_list = pd.DataFrame(generation_list)
                pokemon_list.append(tmp_pokemon_list)
                generation_list = []
        else:
            # Save the frame to pokemon_list array
            tmp_pokemon_list = pd.DataFrame(generation_list)
            pokemon_list.append(tmp_pokemon_list)
            
            # Reset generation list!
            generation_list = []
            
            # Go to the next generation
            gen_count +=1  
            i -=1
    i += 1



             
#lets look at the total base stats
#DataFrame.plot.bar(x=Pokemon, y=Base_Stats, pandas.DataFrame.plot().)
# Any results you write to the current directory are saved as output.
generation_1 = pokemon_list[0].sort_values(by=['true base stats'], ascending = False)
generation_1 = generation_1.drop([100]) #Drop Electrode 
generation_1
generation_1.loc[:129].plot.bar(y='true base stats', x='name')
finalists = pd.DataFrame(columns=generation_1.columns)
finalists = finalists.append(generation_1.loc[:129])
generation_2 = pokemon_list[1].sort_values(by=['true base stats'], ascending = False)
generation_2
generation_2.loc[:8].plot.bar(y='true base stats', x='name')
finalists = finalists.append(generation_2.loc[:8])
generation_3 = pokemon_list[2].sort_values(by=['true base stats'], ascending = False)
generation_3
generation_3.loc[:98].plot.bar(y='true base stats', x='name')
finalists = finalists.append(generation_3.loc[:98])
generation_4 = pokemon_list[3].sort_values(by=['true base stats'], ascending = False)
# Drop pokemon who do not make the cut
generation_4 = generation_4.drop([87,73,5]) 
generation_4
generation_4.loc[:32].plot.bar(y='true base stats', x='name')
finalists = finalists.append(generation_4.loc[:32])
generation_5 = pokemon_list[4].sort_values(by=['true base stats'], ascending = False)
# Drop pokemon who do not make the cut
generation_5 = generation_5.drop([122]) 
generation_5
generation_5.loc[:125].plot.bar(y='true base stats', x='name')
finalists = finalists.append(generation_5.loc[:125])
generation_6 = pokemon_list[5].sort_values(by=['true base stats'], ascending = False)
# Drop pokemon who do not make the cut
generation_6 = generation_6.drop([65]) 
generation_6
generation_6.loc[:5].plot.bar(y='true base stats', x='name')
finalists = finalists.append(generation_6.loc[:5])
generation_7 = pokemon_list[6].sort_values(by=['true base stats'], ascending = False)
# Drop pokemon who do not make the cut
generation_7 = generation_7.drop([50]) 
generation_7
generation_7.loc[:8].plot.bar(y='true base stats', x='name')
finalists = finalists.append(generation_7.loc[:8])
finalists = finalists.sort_values(by=['true base stats'], ascending = False)
finalists

# Append the final pokemon for the BEST team!
top_6 = pd.DataFrame(columns=finalists.columns)
top_6 = top_6.append(finalists.loc[finalists['name'] == 'Greninja'])
top_6 = top_6.append(finalists.loc[finalists['name'] == 'Slaking'])
top_6 = top_6.append(finalists.loc[finalists['name'] == 'Garchomp'])
top_6 = top_6.append(finalists.loc[finalists['name'] == 'Wishiwashi'])
top_6 = top_6.append(finalists.loc[finalists['name'] == 'Metagross'])
top_6 = top_6.append(finalists.loc[finalists['name'] == 'Arcanine'])
top_6

top_6.plot.bar(y='true base stats', x='name')

#Redefining generation variables since was redfined in Marco's section
generation_1 = my_data.iloc [0:151]
generation_2 = my_data.iloc [152:251]
generation_3 = my_data.iloc [252:386]
generation_4 = my_data.iloc [387:493]
generation_5 = my_data.iloc [494:649]
generation_6 = my_data.iloc [650:721]
generation_7 = my_data.iloc [722:800]
ax_generation = sns.countplot(x="generation", data=my_data)
pp = pd.value_counts(my_data.generation)
pp.plot.pie(startangle=90, autopct='%1.1f%%', shadow=False, explode=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05))
plt.axis('equal')
plt.show()
bsgeneration_1 =my_data['base_stats'].iloc [0:151]
bsgeneration_2 =my_data['base_stats'].iloc [152:252]
bsgeneration_3 =my_data['base_stats'].iloc [252:386]
bsgeneration_4 =my_data['base_stats'].iloc [387:493]
bsgeneration_5 =my_data['base_stats'].iloc [494:649]
bsgeneration_6 =my_data['base_stats'].iloc [650:721]
bsgeneration_7 =my_data['base_stats'].iloc [722:800]

absgeneration_1 =statistics.mean(bsgeneration_1)
absgeneration_2 =statistics.mean(bsgeneration_2)
absgeneration_3 =statistics.mean(bsgeneration_3)
absgeneration_4 =statistics.mean(bsgeneration_4)
absgeneration_5 =statistics.mean(bsgeneration_5)
absgeneration_6 =statistics.mean(bsgeneration_6)
absgeneration_7 =statistics.mean(bsgeneration_7)

print("Base stat mean for generation 1:", absgeneration_1)
print("Base stat mean for generation 2:", absgeneration_2)
print("Base stat mean for generation 3:", absgeneration_3)
print("Base stat mean for generation 4:", absgeneration_4)
print("Base stat mean for generation 5:", absgeneration_5)
print("Base stat mean for generation 6:", absgeneration_6)
print("Base stat mean for generation 7:", absgeneration_7)


#######################################################################################
#MODIFIED BASE STAT AVERAGES

mbsgeneration_1 =my_data['mod_stats'].iloc [0:151]
mbsgeneration_2 =my_data['mod_stats'].iloc [152:252]
mbsgeneration_3 =my_data['mod_stats'].iloc [252:386]
mbsgeneration_4 =my_data['mod_stats'].iloc [387:493]
mbsgeneration_5 =my_data['mod_stats'].iloc [494:649]
mbsgeneration_6 =my_data['mod_stats'].iloc [650:721]
mbsgeneration_7 =my_data['mod_stats'].iloc [722:800]

mabsgeneration_1 =statistics.mean(mbsgeneration_1)
mabsgeneration_2 =statistics.mean(mbsgeneration_2)
mabsgeneration_3 =statistics.mean(mbsgeneration_3)
mabsgeneration_4 =statistics.mean(mbsgeneration_4)
mabsgeneration_5 =statistics.mean(mbsgeneration_5)
mabsgeneration_6 =statistics.mean(mbsgeneration_6)
mabsgeneration_7 =statistics.mean(mbsgeneration_7)

print("Modified Base Stats mean for generation 1", mbsgeneration_1)
print("Modified Base Stats mean for generation 2", mbsgeneration_2)
print("Modified Base Stats mean for generation 3", mbsgeneration_3)
print("Modified Base Stats mean for generation 4", mbsgeneration_4)
print("Modified Base Stats mean for generation 5", mbsgeneration_5)
print("Modified Base Stats mean for generation 6", mbsgeneration_6)
print("Modified Base Stats mean for generation 7", mbsgeneration_7)

# Base stat mean for generation 1: 416.25165562913907
# Base stat mean for generation 2: 413.1
# Base stat mean for generation 3: 420.84328358208955
# Base stat mean for generation 4: 452.4339622641509
# Base stat mean for generation 5: 425.9225806451613
# Base stat mean for generation 6: 439.36619718309856
# Base stat mean for generation 7: 447.85897435897436

#Mod Stats
#gen1 = 420 ,gen2 = 527 , gen3 = 575 , gen4 = 513, gen5 = 416, gen6= 529 , gen7= 544
norm_mean = (416,413,420,452,425,439,448)

generation = ('I', 'II', 'III', 'IV', 'V', 'VI', 'VII')

colors = ('r','g','steelblue','goldenrod', 'teal', 'purple', 'darkseagreen')

barlist = plt.bar(generation, norm_mean, width=0.7, color = colors) #(x, y, width of columns, color)
plt.ylim(200,500)
plt.ylabel('Base Stats Mean')
plt.title('Base Stats Mean by Generation')
plt.xlabel('Generation Number')
mod_mean = (420, 527, 575, 513, 416, 529, 544)

# generation = ('I', 'II', 'III', 'IV', 'V', 'VI', 'VII')

barlist = plt.bar(generation, mod_mean, width=0.7, color = colors) #(x, y, width of columns, color)
plt.ylim(250,600)
plt.ylabel('Mod Stats Mean')
plt.title('Mod Stats Mean by Generation')
plt.xlabel('Generation Number')

#HP, attack, sp_attack, defense, sp_defense, and speed

### Generation 1 ###
print('\t\t\t' + "\033[1m" + 'Generation I' + "\033[0;0m")

#Printing stat with tabs to format uniformly and added bold to better see pokemon name.
print('Highest HP Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_1.name[generation_1['hp'].idxmax()] ) + "\033[0;0m")
print('Lowest HP Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_1.name[generation_1['hp'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Attack Pokemon: \t \t'+ "\033[1m" + '{}'.format(generation_1.name[generation_1['attack'].idxmax()] ) + "\033[0;0m")
print('Lowest Attack Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_1.name[generation_1['attack'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Special Attack Pokemon: \t'+ "\033[1m" + '{}'.format(generation_1.name[generation_1['sp_attack'].idxmax()] ) + "\033[0;0m")
print('Lowest Special Attack Pokemon: \t \t' + "\033[1m" + '{}'.format(generation_1.name[generation_1['sp_attack'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Defense Pokemon: \t \t'+ "\033[1m" + '{}'.format(generation_1.name[generation_1['defense'].idxmax()] ) + "\033[0;0m")
print('Lowest Defense Pokemon: \t \t'+ "\033[1m" + '{}'.format(generation_1.name[generation_1['defense'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Special Defense Pokemon: \t'+ "\033[1m" + '{}'.format(generation_1.name[generation_1['sp_defense'].idxmax()] ) + "\033[0;0m")
print('Lowest Special Defense Pokemon: \t'+ "\033[1m" + '{}'.format(generation_1.name[generation_1['sp_defense'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Fastest Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_1.name[generation_1['speed'].idxmax()] ) + "\033[0;0m")
print('Slowest Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_1.name[generation_1['speed'].idxmin()] ) + "\033[0;0m")
print('\n')

### Generation 2 ###
print('\t\t\t' + "\033[1m" + 'Generation II' + "\033[0;0m")

print('Highest HP Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_2.name[generation_2['hp'].idxmax()] ) + "\033[0;0m")
print('Lowest HP Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_2.name[generation_2['hp'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Attack Pokemon: \t \t'+ "\033[1m" + '{}'.format(generation_2.name[generation_2['attack'].idxmax()] ) + "\033[0;0m")
print('Lowest Attack Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_2.name[generation_2['attack'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Special Attack Pokemon: \t'+ "\033[1m" + '{}'.format(generation_2.name[generation_2['sp_attack'].idxmax()] ) + "\033[0;0m")
print('Lowest Special Attack Pokemon: \t \t' + "\033[1m" + '{}'.format(generation_2.name[generation_2['sp_attack'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Defense Pokemon: \t \t'+ "\033[1m" + '{}'.format(generation_2.name[generation_2['defense'].idxmax()] ) + "\033[0;0m")
print('Lowest Defense Pokemon: \t \t'+ "\033[1m" + '{}'.format(generation_2.name[generation_2['defense'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Special Defense Pokemon: \t'+ "\033[1m" + '{}'.format(generation_2.name[generation_2['sp_defense'].idxmax()] ) + "\033[0;0m")
print('Lowest Special Defense Pokemon: \t'+ "\033[1m" + '{}'.format(generation_2.name[generation_2['sp_defense'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Fastest Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_2.name[generation_2['speed'].idxmax()] ) + "\033[0;0m")
print('Slowest Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_2.name[generation_2['speed'].idxmin()] ) + "\033[0;0m")
print('\n')

### Generation 3 ###
print('\t\t\t' + "\033[1m" + 'Generation III' + "\033[0;0m")

print('Highest HP Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_3.name[generation_3['hp'].idxmax()] ) + "\033[0;0m")
print('Lowest HP Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_3.name[generation_3['hp'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Attack Pokemon: \t \t'+ "\033[1m" + '{}'.format(generation_3.name[generation_3['attack'].idxmax()] ) + "\033[0;0m")
print('Lowest Attack Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_3.name[generation_3['attack'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Special Attack Pokemon: \t'+ "\033[1m" + '{}'.format(generation_3.name[generation_3['sp_attack'].idxmax()] ) + "\033[0;0m")
print('Lowest Special Attack Pokemon: \t \t' + "\033[1m" + '{}'.format(generation_3.name[generation_3['sp_attack'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Defense Pokemon: \t \t'+ "\033[1m" + '{}'.format(generation_3.name[generation_3['defense'].idxmax()] ) + "\033[0;0m")
print('Lowest Defense Pokemon: \t \t'+ "\033[1m" + '{}'.format(generation_3.name[generation_3['defense'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Special Defense Pokemon: \t'+ "\033[1m" + '{}'.format(generation_3.name[generation_3['sp_defense'].idxmax()] ) + "\033[0;0m")
print('Lowest Special Defense Pokemon: \t'+ "\033[1m" + '{}'.format(generation_3.name[generation_3['sp_defense'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Fastest Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_3.name[generation_3['speed'].idxmax()] ) + "\033[0;0m")
print('Slowest Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_3.name[generation_3['speed'].idxmin()] ) + "\033[0;0m")
print('\n')

### Generation 4 ###
print('\t\t\t' + "\033[1m" + 'Generation IV' + "\033[0;0m")

print('Highest HP Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_4.name[generation_4['hp'].idxmax()] ) + "\033[0;0m")
print('Lowest HP Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_4.name[generation_4['hp'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Attack Pokemon: \t \t'+ "\033[1m" + '{}'.format(generation_4.name[generation_4['attack'].idxmax()] ) + "\033[0;0m")
print('Lowest Attack Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_4.name[generation_4['attack'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Special Attack Pokemon: \t'+ "\033[1m" + '{}'.format(generation_4.name[generation_4['sp_attack'].idxmax()] ) + "\033[0;0m")
print('Lowest Special Attack Pokemon: \t \t' + "\033[1m" + '{}'.format(generation_4.name[generation_4['sp_attack'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Defense Pokemon: \t \t'+ "\033[1m" + '{}'.format(generation_4.name[generation_4['defense'].idxmax()] ) + "\033[0;0m")
print('Lowest Defense Pokemon: \t \t'+ "\033[1m" + '{}'.format(generation_4.name[generation_4['defense'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Special Defense Pokemon: \t'+ "\033[1m" + '{}'.format(generation_4.name[generation_4['sp_defense'].idxmax()] ) + "\033[0;0m")
print('Lowest Special Defense Pokemon: \t'+ "\033[1m" + '{}'.format(generation_4.name[generation_4['sp_defense'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Fastest Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_4.name[generation_4['speed'].idxmax()] ) + "\033[0;0m")
print('Slowest Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_4.name[generation_4['speed'].idxmin()] ) + "\033[0;0m")
print('\n')

### Generation 5 ###
print('\t\t\t' + "\033[1m" + 'Generation V' + "\033[0;0m")

print('Highest HP Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_5.name[generation_5['hp'].idxmax()] ) + "\033[0;0m")
print('Lowest HP Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_5.name[generation_5['hp'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Attack Pokemon: \t \t'+ "\033[1m" + '{}'.format(generation_5.name[generation_5['attack'].idxmax()] ) + "\033[0;0m")
print('Lowest Attack Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_5.name[generation_5['attack'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Special Attack Pokemon: \t'+ "\033[1m" + '{}'.format(generation_5.name[generation_5['sp_attack'].idxmax()] ) + "\033[0;0m")
print('Lowest Special Attack Pokemon: \t \t' + "\033[1m" + '{}'.format(generation_5.name[generation_5['sp_attack'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Defense Pokemon: \t \t'+ "\033[1m" + '{}'.format(generation_5.name[generation_5['defense'].idxmax()] ) + "\033[0;0m")
print('Lowest Defense Pokemon: \t \t'+ "\033[1m" + '{}'.format(generation_5.name[generation_5['defense'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Special Defense Pokemon: \t'+ "\033[1m" + '{}'.format(generation_5.name[generation_5['sp_defense'].idxmax()] ) + "\033[0;0m")
print('Lowest Special Defense Pokemon: \t'+ "\033[1m" + '{}'.format(generation_5.name[generation_5['sp_defense'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Fastest Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_5.name[generation_5['speed'].idxmax()] ) + "\033[0;0m")
print('Slowest Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_5.name[generation_5['speed'].idxmin()] ) + "\033[0;0m")
print('\n')

### Generation 6 ###
print('\t\t\t' + "\033[1m" + 'Generation VI' + "\033[0;0m")

print('Highest HP Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_6.name[generation_6['hp'].idxmax()] ) + "\033[0;0m")
print('Lowest HP Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_6.name[generation_6['hp'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Attack Pokemon: \t \t'+ "\033[1m" + '{}'.format(generation_6.name[generation_6['attack'].idxmax()] ) + "\033[0;0m")
print('Lowest Attack Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_6.name[generation_6['attack'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Special Attack Pokemon: \t'+ "\033[1m" + '{}'.format(generation_6.name[generation_6['sp_attack'].idxmax()] ) + "\033[0;0m")
print('Lowest Special Attack Pokemon: \t \t' + "\033[1m" + '{}'.format(generation_6.name[generation_6['sp_attack'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Defense Pokemon: \t \t'+ "\033[1m" + '{}'.format(generation_6.name[generation_6['defense'].idxmax()] ) + "\033[0;0m")
print('Lowest Defense Pokemon: \t \t'+ "\033[1m" + '{}'.format(generation_6.name[generation_6['defense'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Special Defense Pokemon: \t'+ "\033[1m" + '{}'.format(generation_6.name[generation_6['sp_defense'].idxmax()] ) + "\033[0;0m")
print('Lowest Special Defense Pokemon: \t'+ "\033[1m" + '{}'.format(generation_6.name[generation_6['sp_defense'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Fastest Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_6.name[generation_6['speed'].idxmax()] ) + "\033[0;0m")
print('Slowest Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_6.name[generation_6['speed'].idxmin()] ) + "\033[0;0m")
print('\n')

### Generation 7 ###
print('\t\t\t' + "\033[1m" + 'Generation VII' + "\033[0;0m")

print('Highest HP Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_7.name[generation_7['hp'].idxmax()] ) + "\033[0;0m")
print('Lowest HP Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_7.name[generation_7['hp'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Attack Pokemon: \t \t'+ "\033[1m" + '{}'.format(generation_7.name[generation_7['attack'].idxmax()] ) + "\033[0;0m")
print('Lowest Attack Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_7.name[generation_7['attack'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Special Attack Pokemon: \t'+ "\033[1m" + '{}'.format(generation_7.name[generation_7['sp_attack'].idxmax()] ) + "\033[0;0m")
print('Lowest Special Attack Pokemon: \t \t' + "\033[1m" + '{}'.format(generation_7.name[generation_7['sp_attack'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Defense Pokemon: \t \t'+ "\033[1m" + '{}'.format(generation_7.name[generation_7['defense'].idxmax()] ) + "\033[0;0m")
print('Lowest Defense Pokemon: \t \t'+ "\033[1m" + '{}'.format(generation_7.name[generation_7['defense'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Special Defense Pokemon: \t'+ "\033[1m" + '{}'.format(generation_7.name[generation_7['sp_defense'].idxmax()] ) + "\033[0;0m")
print('Lowest Special Defense Pokemon: \t'+ "\033[1m" + '{}'.format(generation_7.name[generation_7['sp_defense'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Fastest Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_7.name[generation_7['speed'].idxmax()] ) + "\033[0;0m")
print('Slowest Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(generation_7.name[generation_7['speed'].idxmin()] ) + "\033[0;0m")
print('\n')

ax_legendary = sns.countplot(x="is_legendary", data=my_data)


legendary = pd.value_counts(my_data.is_legendary)
print(legendary)
f, ax = plt.subplots(figsize=(8, 8))
ax = sns.countplot(x="generation", hue = 'is_legendary', data=my_data)
ax.set_title('Number of Legendaries and non-Legendaries by Generation')
#Gen 7 has the highest amount of legendaries. EVEN MORE SINCE NOT ALL LEGENDARIES INCLUDED FROM THAT GEN
sns.lmplot( x="weight_kg", y="height_m", data=legendaries2, fit_reg=False)
yy = pd.value_counts(legendaries2['capture_rate'])

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("whitegrid")

ax = sns.barplot(x=yy.index, y=yy, data=legendaries2)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 12)
ax.set(xlabel='Capture_rate', ylabel='Number of Pokemon')
ax.set_title('Distribution of capture_rate against number of Pokemon')

# Each species of Pokémon has a catch rate that applies to all its members. Higher catch rates mean that the Pokémon is easier to catch, up to a maximum of 255. 
# When a Poké Ball is thrown at a wild Pokémon, the game uses that Pokémon's catch rate in a formula to determine the chances of catching that Pokémon. The formula also takes into account the following factors:
yy = pd.value_counts(legendaries2['base_happiness'])

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("whitegrid")

ax = sns.barplot(x=yy.index, y=yy, data=legendaries2)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 12)
ax.set(xlabel='Base Happiness', ylabel='Number of Pokemon')
ax.set_title('Distribution of Base Happiness against number of Pokemon')
#legendaries2.describe
t1L = pd.value_counts(legendaries['type1'])

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("whitegrid")

ax = sns.barplot(x=t1L.index, y=t1L, data=legendaries)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 12)
ax.set(xlabel='Primary types', ylabel='Count')
ax.set_title('Distribution of Legendary Pokemon Primary type')

# t2 = pd.value_counts(df['type2'])

# fig, ax = plt.subplots()
# fig.set_size_inches(11.7, 8.27)
# sns.set_style("whitegrid")

# ax = sns.barplot(x=t2.index, y=t2, data=df)
# ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 12)
# ax.set(xlabel='Secondary types', ylabel='Count')
# ax.set_title('Distribution of Secondary Pokemon type')
t2L = pd.value_counts(legendaries['type2'])

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("whitegrid")

ax = sns.barplot(x=t2L.index, y=t2L, data=legendaries)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 12)
ax.set(xlabel='Secondary types', ylabel='Count')
ax.set_title('Distribution of Legendary Pokemon Secondary type')
yy = pd.value_counts(legendaries['base_stats'])

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("whitegrid")

ax = sns.barplot(x=yy.index, y=yy, data=legendaries)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 12)
ax.set(xlabel='Base Stats', ylabel='Number of Pokemon')
ax.set_title('Distribution of Base Stats against number of Pokemon')


print('Highest Base Stats Legendary Pokemon: \t \t'+ "\033[1m" + '{}'.format(legendaries.name[legendaries['base_stats'].idxmax()] ) + "\033[0;0m")
#print(legendaries['base_stats'].max())
print('Lowest Base Stats Legendary Pokemon: \t \t'+ "\033[1m" + '{}'.format(legendaries.name[legendaries['base_stats'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest HP Legendary Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(legendaries.name[legendaries['hp'].idxmax()] ) + "\033[0;0m")
print('Lowest HP Legendary Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(legendaries.name[legendaries['hp'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Attack Legendary Pokemon: \t \t'+ "\033[1m" + '{}'.format(legendaries.name[legendaries['attack'].idxmax()] ) + "\033[0;0m")
print('Lowest Attack Legendary Pokemon: \t \t'+ "\033[1m" + '{}'.format(legendaries.name[legendaries['attack'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Special Attack Legendary Pokemon: \t'+ "\033[1m" + '{}'.format(legendaries.name[legendaries['sp_attack'].idxmax()] ) + "\033[0;0m")
print('Lowest Special Attack Legendary Pokemon: \t' + "\033[1m" + '{}'.format(legendaries.name[legendaries['sp_attack'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Defense Legendary Pokemon: \t \t'+ "\033[1m" + '{}'.format(legendaries.name[legendaries['defense'].idxmax()] ) + "\033[0;0m")
print('Lowest Defense Legendary Pokemon: \t \t'+ "\033[1m" + '{}'.format(legendaries.name[legendaries['defense'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Highest Special Defense Legendary Pokemon: \t'+ "\033[1m" + '{}'.format(legendaries.name[legendaries['sp_defense'].idxmax()] ) + "\033[0;0m")
print('Lowest Special Defense Legendary Pokemon: \t'+ "\033[1m" + '{}'.format(legendaries.name[legendaries['sp_defense'].idxmin()] ) + "\033[0;0m")
print('\n')

print('Fastest Legendary Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(legendaries.name[legendaries['speed'].idxmax()] ) + "\033[0;0m")
print('Slowest Legendary Pokemon: \t \t \t'+ "\033[1m" + '{}'.format(legendaries.name[legendaries['speed'].idxmin()] ) + "\033[0;0m")

#del list #list is called earlier and was messing with apending a new list, probably for loop used for coloring

unique_elem = []
for col in against_:
    unique_elem.append(legendaries[col].unique().tolist())
    
result = set(x for l in unique_elem for x in l)

result = list(result)
print(result)
for col in against_:
    if (np.mean(legendaries[col]) > 1.2):
        print(col)

for col in against_:
    if (np.sum(legendaries[col]) > 1000):
        print(col)
import random

for col in range(0, len(against_)):
    print (against_[col])
    print (df[against_[col]].unique())
    pp = pd.value_counts(df[against_[col]])
    
    color = ['g', 'b', 'r', 'y', 'pink', 'orange', 'brown']
            
    pp.plot.bar(color = random.choice(color))
    plt.show()
ax_hp = sns.distplot(legendaries['hp'], color="pink")
ax_attack = sns.distplot(legendaries['attack'], color="red")
ax_spattack = sns.distplot(legendaries['sp_attack'], color="purple")
ax_attack = sns.distplot(legendaries['attack'], color="red")
ax_spattack = sns.distplot(legendaries['sp_attack'], color="purple")



ax_defense = sns.distplot(legendaries['defense'], color="#4b7d4b")

ax_spdefense = sns.distplot(legendaries['sp_defense'], color="#6cb46c")

ax_defense = sns.distplot(legendaries['defense'], color="#4b7d4b")
ax_spdefense = sns.distplot(legendaries['sp_defense'], color="#6cb46c")
#exammple of taking out histogram

ax_spattack = sns.distplot(legendaries['sp_attack'], color="g", hist=False)
ax_spdefense = sns.distplot(legendaries['sp_defense'], color="y", hist=False)
ax_speed = sns.distplot(legendaries['speed'], color="orange")
#Loop to find all legendary pokemon pokedex numbers

dex_num = []
for i in range(0, len(all_data)):
    if(all_data.is_legendary[i] > 0):
        dex_num.append(all_data.pokedex_number[i])
        
print(set(dex_num))  

# all_data.base_egg_steps.corr(all_data.is_legendary)


