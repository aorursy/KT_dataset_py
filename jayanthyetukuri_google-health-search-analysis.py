import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



healthSearchData=pd.read_csv("../input/RegionalInterestByConditionOverTime.csv")

healthSearchData.head(3)
healthSearchData = healthSearchData.drop(['geoCode'],axis=1)
#2004-2017

#cancer cardiovascular stroke depression rehab vaccine diarrhea obesity diabetes

yearWiseMeam = {}

for col in healthSearchData.columns:

    if '+' in col:

        year = col.split('+')[0]

        disease = col.split('+')[-1]

        if not disease in yearWiseMeam:

            yearWiseMeam[disease] = {}

        if not year in yearWiseMeam[disease]:

            yearWiseMeam[disease][year] = np.mean(list(healthSearchData[col]))



plt.figure(figsize=(18, 6))

ax = plt.subplot(111)

plt.title("Year wise google medical search", fontsize=20)



ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13])

ax.set_xticklabels(list(yearWiseMeam['cancer'].keys()))

lh = {}

for disease in yearWiseMeam:

    lh[disease] = plt.plot(yearWiseMeam[disease].values())

plt.legend(lh, loc='best')

plt.figure(figsize=(18, 6))

ax = plt.subplot(111)

plt.title("Year wise google medical search [smoothened]", fontsize=20)



ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13])

ax.set_xticklabels(list(yearWiseMeam['cancer'].keys()))

lh = {}

myLambda = 0.7

for disease in yearWiseMeam:

    tempList = list(yearWiseMeam[disease].values())

    localMean = np.mean(tempList)

    smoothList = []

    for x in tempList:

        smoothList.append(x + myLambda * (localMean - x)) 

    lh[disease] = plt.plot(smoothList)

plt.legend(lh, loc='best')
statesData = pd.DataFrame(healthSearchData.iloc[:,0])

healthSearchData = healthSearchData.drop(['dma'],axis=1)



meanDict = {}

yearList = []

illnessList = []

for col in healthSearchData.columns:

    if '+' in col:

        yearList.append(col.split('+')[0])

        illnessList.append(col.split('+')[-1])

        

for index, row in healthSearchData.iterrows():

    for illness in illnessList:

        searchCountList = []

        for year in yearList:

            searchCountList.append(row[year+ '+' +illness])

        if not illness in meanDict:

            meanDict[illness] = []

        meanDict[illness].append(np.mean(searchCountList))

yearWiseMeanDf = pd.DataFrame.from_dict(meanDict, orient='columns', dtype=None)

heatMapData = statesData.join(yearWiseMeanDf)

heatMapData.set_index('dma', inplace=True, drop=True)



import seaborn as sns

plt.figure(figsize=(10, 25))

plt.title("State wise illness search", fontsize=16)

ax = plt.subplot(111)

ax.spines["top"].set_visible(False)    

ax.spines["bottom"].set_visible(False)    

ax.spines["right"].set_visible(False)    

ax.spines["left"].set_visible(False)

ax.get_xaxis().tick_bottom()    

ax.get_yaxis().tick_left()

ax = sns.heatmap(heatMapData)