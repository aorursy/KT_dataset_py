import pandas as pd

import numpy as np

import os

print(os.listdir("../input"))



df = pd.read_csv('../input/Pokemon.csv')

plotFontSize = 14
print(df.shape)

df.head(10)
typePlot = df['Type 1'].value_counts().plot.bar()

typePlot.set_title("Type 1 frequencies", fontsize = plotFontSize)

typePlot.set_xlabel("Type", fontsize = plotFontSize)

typePlot.set_ylabel("Count", fontsize = plotFontSize)
typePlot = df['Type 2'].value_counts().plot.bar()

typePlot.set_title("Type 2 frequencies", fontsize = plotFontSize)

typePlot.set_xlabel("Type", fontsize = plotFontSize)

typePlot.set_ylabel("Count", fontsize = plotFontSize)
typeCounts = df['Type 1'].value_counts() + df['Type 2'].value_counts()



typePlot = typeCounts.plot.bar()

typePlot.set_title("Total type frequency", fontsize = plotFontSize)

typePlot.set_xlabel("Type", fontsize = plotFontSize)

typePlot.set_ylabel("Count", fontsize = plotFontSize)
# df.head(151).plot.scatter(x='#', y='Total')
# Check out properties of typeCounts

print(type(typeCounts))

print(typeCounts.shape)

print(typeCounts[0])

print(type(df))

print(typeCounts.axes)
typeDF = typeCounts.to_frame()

typeDF = typeDF.rename(columns={0: "Frequency"})



typeDF['Avg. HP'] = 0

typeDF['Avg. Attack'] = 0

typeDF['Avg. Defense'] = 0

typeDF['Avg. Sp. Atk'] = 0

typeDF['Avg. Sp. Def'] = 0

typeDF['Avg. Speed'] = 0

typeDF['Avg. Total'] = 0



statList = ['HP', 'Speed', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Total']

typeList = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire',

       'Flying', 'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison',

       'Psychic', 'Rock', 'Steel', 'Water']



for pType in typeList:

    for stat in statList:

        statSum = 0

        monsterCount = 0



        for index, row in df.iterrows():

            if row['Type 1'] == pType or row['Type 2'] == pType:

                statSum += row[stat]

                monsterCount += 1



        average = statSum / monsterCount

        typeDF.at[pType, 'Avg. ' + stat] = average



typeDF.sort_values('Avg. Total', inplace = True)

typeDF
typeTotals = typeDF['Avg. Total']



typePlot = typeTotals.plot.bar()

typePlot.set_title("Avg. total stats by type", fontsize = plotFontSize)

typePlot.set_xlabel("Type", fontsize = plotFontSize)

typePlot.set_ylabel("Avg. Total", fontsize = plotFontSize)
cleanTypeDF = typeCounts.to_frame()

cleanTypeDF = typeDF.rename(columns={0: "Frequency"})



cleanTypeDF['Avg. HP'] = 0

cleanTypeDF['Avg. Attack'] = 0

cleanTypeDF['Avg. Defense'] = 0

cleanTypeDF['Avg. Sp. Atk'] = 0

cleanTypeDF['Avg. Sp. Def'] = 0

cleanTypeDF['Avg. Speed'] = 0

cleanTypeDF['Avg. Total'] = 0



statList = ['HP', 'Speed', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Total']

typeList = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire',

       'Flying', 'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison',

       'Psychic', 'Rock', 'Steel', 'Water']



for pType in typeList:

    for stat in statList:

        statSum = 0

        monsterCount = 0



        for index, row in df.iterrows():

            if row['Legendary'] == False and 'Mega' not in row['Name'] and (row['Type 1'] == pType or row['Type 2'] == pType):

                statSum += row[stat]

                monsterCount += 1

                

        average = statSum / monsterCount

        cleanTypeDF.at[pType, 'Avg. ' + stat] = average



cleanTypeDF.sort_values('Avg. Total', inplace = True)

cleanTypeDF
cleanTypeTotals = cleanTypeDF['Avg. Total']



typePlot = cleanTypeTotals.plot.bar()

typePlot.set_title("Avg. total stats by type (excluding legendaries and megas)", fontsize = plotFontSize)

typePlot.set_xlabel("Type", fontsize = plotFontSize)

typePlot.set_ylabel("Avg. Total", fontsize = plotFontSize)
genCounts = df['Generation'].value_counts()



genPlot = genCounts.plot.bar()

genPlot.set_title("Total gen frequency (including legendaries and megas)", fontsize = plotFontSize)

genPlot.set_xlabel("Gen", fontsize = plotFontSize)

genPlot.set_ylabel("Count", fontsize = plotFontSize)
genDF = genCounts.to_frame()

genDF = genDF.rename(columns={0: "Frequency"})



genDF['Avg. HP'] = 0

genDF['Avg. Attack'] = 0

genDF['Avg. Defense'] = 0

genDF['Avg. Sp. Atk'] = 0

genDF['Avg. Sp. Def'] = 0

genDF['Avg. Speed'] = 0

genDF['Avg. Total'] = 0



statList = ['HP', 'Speed', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Total']

genList = [1, 2, 3, 4, 5, 6]



for pGen in genList:

    for stat in statList:

        statSum = 0

        monsterCount = 0



        for index, row in df.iterrows():

            if row['Legendary'] == False and 'Mega' not in row['Name'] and row['Generation'] == pGen:

                statSum += row[stat]

                monsterCount += 1



        average = statSum / monsterCount

        genDF.at[pGen, 'Avg. ' + stat] = average

        genDF.at[pGen, 'Generation'] = monsterCount



genDF.sort_values('Avg. Total', inplace = True)

genDF
cleanGenCounts = genDF['Avg. Total']



typePlot = cleanGenCounts.plot.bar()

typePlot.set_title("Avg. total stats by gen (excluding legendaries and megas)", fontsize = plotFontSize)

typePlot.set_xlabel("Gen", fontsize = plotFontSize)

typePlot.set_ylabel("Avg. Total", fontsize = plotFontSize)
df.loc[700:]
from sklearn.model_selection import train_test_split



df_train, df_test = train_test_split(df)



df_train.head()
df_test.head()