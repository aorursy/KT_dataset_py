# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import csv

reader = csv.reader(open('../input/pokemon.csv'))

counter = 0

pokemonDict = {}

for row in reader:

    counter += 1

    key = row[0]

    if key in pokemonDict:

        # implement your duplicate row handling here

        pass

    pokemonDict[key] = row[1:]

print(pokemonDict)

print(counter - 1)
##**Create a Dictionary of Each Type and Counts of Everytime They Are Present**

totalCounter = 0

noTypesCounter = 0

oneTypeCounter = 0

twoTypesCounter = 0

totalTypeCounter = 0

typeDict = {}

for key in pokemonDict.keys():

    if key == "Name":

        continue

    elif len(pokemonDict[key]) == 1:

        oneTypeCounter += 1

    elif len(pokemonDict[key]) == 2:

        twoTypesCounter += 1

    totalCounter += 1

    totalTypeCounter += len(pokemonDict[key])

    for pokemonType in pokemonDict[key]:

        if pokemonType in typeDict:

            typeDict[pokemonType] += 1

        else:

            typeDict[pokemonType] = 1

print(typeDict)
label = list(typeDict.keys())

plt.bar(['One-Type', 'Two-Types'], [oneTypeCounter, twoTypesCounter])

plt.xlabel('No. of Types', fontsize=20)

plt.ylabel('No. of Instances', fontsize=10)

plt.xticks(['One-Type', 'Two-Types'], ['One-Type', 'Two-Types'], fontsize=10, rotation=90)

plt.title('No. Of Type Instances Representative in All Pokemon')

plt.show()
totalTypes = 0

for eachType in typeDict.keys():

    totalTypes += typeDict[eachType]



label = list(typeDict.keys())

index = np.arange(len(label))

plt.bar(list(typeDict.keys()), list(typeDict.values()))

plt.xlabel('Types', fontsize=20)

plt.ylabel('No. of Instances', fontsize=10)

plt.xticks(index, label, fontsize=10, rotation=90)

plt.title('No. Of Type Instances Representative in All Pokemon')

plt.show()
dualTypeDict = {}

for key in pokemonDict.keys():

    if key == "Name":

        continue

    elif len(pokemonDict[key]) == 1:

        if pokemonDict[key][0] in dualTypeDict:

            dualTypeDict[pokemonDict[key][0]] += 1

        else:

            dualTypeDict[pokemonDict[key][0]] = 1

    elif len(pokemonDict[key]) == 2:

        typeList = list()

        typeList.append(pokemonDict[key][0])

        typeList.append(pokemonDict[key][1])

        typeList.sort()

        if typeList[0] + "-" + typeList[1] in dualTypeDict:

            dualTypeDict[typeList[0] + "-" + typeList[1]] += 1

        else:

            dualTypeDict[typeList[0] + "-" + typeList[1]] = 1

print(dualTypeDict)
for eachType in typeDict.keys():

    tmpTypes = list()

    tmpTypeCount = list()

    for eachDualType in dualTypeDict.keys():

        if eachType in eachDualType:

            tmpTypes.append(eachDualType)

            tmpTypeCount.append(dualTypeDict[eachDualType])    

    label = tmpTypes

    index = np.arange(len(label))

    plt.bar(tmpTypes, tmpTypeCount)

    plt.xlabel(eachType, fontsize=20)

    plt.ylabel('No. of Instances', fontsize=10)

    plt.xticks(index, label, fontsize=10, rotation=90)

    plt.title('No. Of Dual-Type Instances Representative in All Pokemon')

    plt.show()

    