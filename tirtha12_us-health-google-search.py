#importing the required libraries
import seaborn as sns # for plotting
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt#for plotting
import os #file handling in os
print(os.listdir("../input")) #list the current data location
#loading dataset using pandas
data = pd.read_csv('../input/RegionalInterestByConditionOverTime.csv')
#looking for random 5 data samoles
data.sample(5)
#checking for the size of the dataset
data.shape
#check missing values
data.columns[data.isnull().any()]
#separate variables into new data frames
data1 = data.select_dtypes(include=[np.number])
sns.jointplot('2004+cancer', '2017+cancer', data=data1)
sns.jointplot('2016+cancer', '2017+cancer', data=data1)
#2004-2017
#cancer cardiovascular stroke depression rehab vaccine diarrhea obesity diabetes
yearWiseMean = {}
for col in data1.columns:
    if '+' in col:
        year = col.split('+')[0]
        disease = col.split('+')[-1]
        if not disease in yearWiseMean:
            yearWiseMean[disease] = {}
        if not year in yearWiseMean[disease]:
            yearWiseMean[disease][year] = np.mean(list(data1[col]))

plt.figure(figsize=(18, 6))
ax = plt.subplot(111)
plt.title("Year wise google medical search", fontsize=20)

ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13])
ax.set_xticklabels(list(yearWiseMean['cancer'].keys()))
lh = {}
for disease in yearWiseMean:
    lh[disease] = plt.plot(yearWiseMean[disease].values())
plt.legend(lh, loc='best')
healthSearchData=data
statesData = pd.DataFrame(healthSearchData.iloc[:,0])
#healthSearchData = healthSearchData.drop(['dma'],axis=1)

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