filedir = '../input/Pokemon.csv'

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

dataframe = pd.read_csv(filedir)

elementalTypesList = ["Fire", "Water", "Bug", "Normal", "Fighting", "Flying", "Poison", "Electric", "Ground", "Fairy", "Dragon", "Grass", "Steel", "Ghost", "Ice", "Rock", "Dark", "Psychic"]

elementalTotalList = []

elementalStatList = []

elementalStatListStdev = []

for elementalType in elementalTypesList:    

    totalXTypes = 0

    averageStats = 0

    statCounter = 1

    statTest = []

    for index, row in dataframe.iterrows():

        if "Mega" not in row['Name']: 

            if((row['Type 1']==elementalType) or (row['Type 2']==elementalType)):

                #print(row)

                totalXTypes  += 1

                statTest += [row['Total']]

    #print("total " + elementalType + " Types = " + str(totalXTypes))

    elementalTotalList += [totalXTypes]

    #print("average " + elementalType + " Stats = " + str(sum(statTest)/len(statTest)))

    elementalStatList += [sum(statTest)/len(statTest)]

    #print(np.std(statTest))

    elementalStatListStdev += [np.std(statTest)]

    

# Ready data for plt

types = elementalTypesList

y_pos = np.arange(len(types))

performance = elementalStatList

error = elementalStatListStdev



# Sort by avg stats

idx = np.array(elementalStatList).argsort()

types, elementalStatList, elementalTotalList = [np.take(x, idx) for x in [types, elementalStatList, elementalTotalList]]



y = np.arange(elementalTotalList.size)



# Draw plot

fig, axes = plt.subplots(ncols=2, sharey=True)

axes[0].barh(y, elementalStatList, align='center', color='gray', zorder=10)

axes[0].set(title='Avg stats')

axes[1].barh(y, elementalTotalList, align='center', color='gray', zorder=10)

axes[1].set(title='Total pokemon')



axes[0].invert_xaxis()

axes[0].set(yticks=y, yticklabels=types)

axes[0].yaxis.tick_right()



for ax in axes.flat:

    ax.margins(0.03)

    ax.grid(True)



fig.tight_layout()

fig.subplots_adjust(wspace=0.29)

plt.show()



# Sort by avg stats

idx = np.array(elementalTotalList).argsort()

types, elementalTotalList, elementalStatList = [np.take(x, idx) for x in [types, elementalTotalList, elementalStatList]]



y = np.arange(elementalStatList.size)



# Draw plot2

fig2, axes = plt.subplots(ncols=2, sharey=True)

axes[0].barh(y, elementalStatList, align='center', color='gray', zorder=10)

axes[0].set(title='Avg stats')

axes[1].barh(y, elementalTotalList, align='center', color='gray', zorder=10)

axes[1].set(title='Total pokemon')



axes[0].invert_xaxis()

axes[0].set(yticks=y, yticklabels=types)

axes[0].yaxis.tick_right()



for ax in axes.flat:

    ax.margins(0.03)

    ax.grid(True)



fig2.tight_layout()

fig2.subplots_adjust(wspace=0.29)

plt.show()

filedir = '../input/Pokemon.csv'

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from collections import defaultdict

dataframe = pd.read_csv(filedir)

elementalTypesList = ["Fire", "Water", "Bug", "Normal", "Fighting", "Flying", "Poison", "Electric", "Ground", "Fairy", "Dragon", "Grass", "Steel", "Ghost", "Ice", "Rock", "Dark", "Psychic"]

elementalTotalList = []

elementalStatList = []

elementalStatListStdev = []

dualTypePokemonList = []

dualTypePokemonDict = defaultdict(list)

for elementalType in elementalTypesList:    

    totalXTypes = 0

    averageStats = 0

    statCounter = 1

    statTest = []

    for index, row in dataframe.iterrows():

        if "Mega" not in row['Name'] and row['#'] not in dualTypePokemonList: 

            for elementalType2 in elementalTypesList:

                if((row['Type 1']==elementalType) and (row['Type 2']==elementalType2)):

                    #print(row)

                    totalXTypes  += 1

                    statTest += [row['Total']]

                    dualTypePokemonList += [row['#']]

                    dualTypePokemonDict[elementalType+","+elementalType2].append(row['Total'])

    #print("total " + elementalType + " Types = " + str(totalXTypes))

    elementalTotalList += [totalXTypes]

    #print("average " + elementalType + " Stats = " + str(sum(statTest)/len(statTest)))

    elementalStatList += [sum(statTest)/len(statTest)]

    #print(np.std(statTest))

    elementalStatListStdev += [np.std(statTest)]

#print('dualtype list' + str(len(list(dualTypePokemonList))))

#print(sum(elementalTotalList))

#print(dualTypePokemonDict)



listForGraph = []

elementList = []

for element in elementalTypesList:

    fireList = []

    elementList += element

    for key, val in dualTypePokemonDict.items():

        if element in key:

            fireList += val

    #print(fireList)

    #print("Avg of all dual type pokemon with one " + str(element) + " element: " + str(sum(fireList)/len(fireList)) )

    listForGraph += [sum(fireList)/len(fireList)]



    

# Ready data for plt

types = elementalTypesList

y_pos = np.arange(len(types))

performance = listForGraph

error = np.std(listForGraph)



# Sort by avg stats

idx = np.array(listForGraph).argsort()

types, listForGraph = [np.take(x, idx) for x in [types, listForGraph]]



y = np.arange(listForGraph.size)



# Draw plot

fig, axes = plt.subplots(ncols=1, sharey=True)

axes.barh(y, listForGraph, align='center', color='gray', zorder=10)

axes.set(title='Avg stats for dual types where one type is...')



axes.invert_xaxis()

axes.set(yticks=y, yticklabels=types)

axes.yaxis.tick_right()





axes.margins(0.03)

axes.grid(True)



fig.tight_layout()

fig.subplots_adjust(wspace=0.29)

plt.show()

    