# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

from pandas import DataFrame

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Pokemon.csv')



## removing the # and Total columns from the dataframe

df.drop(df.columns[[0, 4,11,12]], axis=1, inplace=True)



## renaming the column names 'Type 1' and 'Type 2' to Type_1 and Type_2

df = df.rename(columns={'Type 1': 'Type_1', 'Type 2': 'Type_2'})

df.fillna(value='missing', axis=1, inplace=True)



print(df.head(5))
## Now we pull out the different types of Pokemon from the Type columns

types = df.Type_1.unique()

print(types)
## Now we create a function to return separate dataframes for each Type in the types list

def returnDataFrame(typeName):

    col1 = df[df['Type_1'].str.contains(typeName)]

    col2 = df[df['Type_2'].str.contains(typeName)]

    frames = [col1,col2]

    new_df = pd.concat(frames)

    return new_df
# We create a dictionary to hold each dataframe generated from the above function

type_dict = {}

for typeName in types:

    type_dict.update({typeName: returnDataFrame(typeName)})
# We create a dictionary which contains the Type:stats, where stats is a list containing the average stats fora given type

stats_dict = {}

for key,value in type_dict.items():

    typeName = key

    statsList = []

    # axis = 0 means along the column, axis = 1 means along the row

    # shape[0] gives the length of the dataframe (number of rows) associated with each Type

    hp = (np.sum(type_dict[key]['HP'].values, axis=0))/type_dict[key].shape[0]

    statsList.append(round(hp))

    attack = (np.sum(type_dict[key]['Attack'].values, axis=0))/type_dict[key].shape[0]

    statsList.append(round(attack))

    defense = (np.sum(type_dict[key]['Defense'].values, axis=0))/type_dict[key].shape[0]

    statsList.append(round(defense))

    sp_atk = (np.sum(type_dict[key]['Sp. Atk'].values, axis=0))/type_dict[key].shape[0]

    statsList.append(round(sp_atk))

    sp_def = (np.sum(type_dict[key]['Sp. Def'].values, axis=0))/type_dict[key].shape[0]

    statsList.append(round(sp_def))

    speed = (np.sum(type_dict[key]['Speed'].values, axis=0))/type_dict[key].shape[0]

    statsList.append(round(speed))

    stats_dict.update({typeName:statsList})
## Creating a DataFrame from the dictionary

stats_df = pd.DataFrame(stats_dict)

stats_df['Stats'] = ['HP','ATK','DEF','SpATK','SpDEF','SPD']

print(stats_df.head(5))
## Plotting comparisons via bar chart

## hp

hp = stats_df[0:1]

ax = hp.plot(kind='bar', title ="Comparison of Hitpoints among various Types", figsize=(15, 10), legend=True, fontsize=12)

ax.set_xlabel("Type", fontsize=12)

ax.set_ylabel("HP", fontsize=12)

plt.show()
# attack

hp = stats_df[1:2]

ax = hp.plot(kind='bar', title ="Comparison of Attack among various Types", figsize=(15, 10), legend=True, fontsize=12)

ax.set_xlabel("Type", fontsize=12)

ax.set_ylabel("Attack", fontsize=12)

plt.show()
#defense

hp = stats_df[2:3]

ax = hp.plot(kind='bar', title ="Comparison of Defense among various Types", figsize=(15, 10), legend=True, fontsize=12)

ax.set_xlabel("Type", fontsize=12)

ax.set_ylabel("Defense", fontsize=12)

plt.show()
#spatk

hp = stats_df[3:4]

ax = hp.plot(kind='bar', title ="Comparison of Special Attacks among various Types", figsize=(15, 10), legend=True, fontsize=12)

ax.set_xlabel("Type", fontsize=12)

ax.set_ylabel("Sp_Atk", fontsize=12)

plt.show()
#spdef

hp = stats_df[4:5]

ax = hp.plot(kind='bar', title ="Comparison of Special Defense moves among various Types", figsize=(15, 10), legend=True, fontsize=12)

ax.set_xlabel("Type", fontsize=12)

ax.set_ylabel("Sp_Def", fontsize=12)

plt.show()
#speed

hp = stats_df[5:6]

ax = hp.plot(kind='bar', title ="Comparison of Speed among various Types", figsize=(15, 10), legend=True, fontsize=12)

ax.set_xlabel("Type", fontsize=12)

ax.set_ylabel("Speed", fontsize=12)

plt.show()