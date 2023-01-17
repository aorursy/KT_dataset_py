# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
poke = pd.read_csv('../input/Pokemon Competitive Statistics.csv')
poke.head(10)
pokemoncount = poke['Name'].groupby(poke['Name']).count().sort_values(ascending=True)

#print(pokemoncount)
pokemoncount.plot(kind='bar', figsize=(20, 8))
pokemonteams = poke['Name'].groupby(poke['Trainer'])
for x, group in pokemonteams:
    line = str(x)+": "
    for i in group:
        line += i+" "
    print(line)
    
    
pokemonteams = poke['Name'].groupby(poke['Trainer'])
def getSecond(item):
    return item[1]
pairings = []
for x, group in pokemonteams:
    listedGroup = list(group)
    for a in range(0,6):
        for b in range(0,6):
            if(a != b):
                pair = [listedGroup[a],listedGroup[b]]
                pair.sort()
                pairings.append(pair)
uniquePairings = []
pairingNum = []
for x in pairings:
    if(uniquePairings.count(x) == 0):
        uniquePairings.append(x)
for x in uniquePairings:
    pairingNum.append([x,pairings.count(x)])
pairingNum = pd.Series(sorted(pairingNum,key=getSecond,reverse=True))
for x in pairingNum:
    print(x[0][0],"and",x[0][1],"=",x[1],"pairings")
#pairingNum.plot(kind='bar', figsize=(13, 6))

pokeskills = poke[['Move1', 'Move2', 'Move3', 'Move4', 'Ability']]

pokeskills
pokeability = poke['Ability'].groupby(poke['Ability']).count().sort_values(ascending=True)
#pokeability
pokeability.plot(kind='bar', figsize=(13, 6))
mergedmoves = list(poke['Move1'])

mergedmoves.extend(list(poke['Move2']))
mergedmoves.extend(list(poke['Move3']))
mergedmoves.extend(list(poke['Move4']))
mergeddisplaytail = pd.Series(mergedmoves).groupby(mergedmoves).count().sort_values(ascending=True).tail(10)

mergeddisplaytail.plot(kind='bar', figsize=(24, 10))

mergeddisplay = pd.Series(mergedmoves).groupby(mergedmoves).count().sort_values(ascending=True)

mergeddisplayhead = mergeddisplay.head(len(mergeddisplay) - len(mergeddisplaytail))

mergeddisplayhead.plot(kind='bar', figsize=(22, 6))