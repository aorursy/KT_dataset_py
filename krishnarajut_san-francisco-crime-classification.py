# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import sys

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier

#### setting the figure size required

plt.figure(figsize=(100,100))



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



# Any results you write to the current directory are saved as output.
trainMasterDf = pd.read_csv("../input/train.csv")

print(trainMasterDf.head())

print(trainMasterDf.columns)

trainMasterDf = trainMasterDf.drop(['Descript','Resolution','Address'],axis = 1)
testMasterDf = pd.read_csv("../input/test.csv")

print(testMasterDf.head())

testMasterDf = testMasterDf.drop(['Address'],axis = 1)

print(testMasterDf.columns)
####  BarPlot between the crime category and their counts(Univariant analysis)  #####

newdf = trainMasterDf.groupby(['Category'])

catcount = pd.DataFrame([[key,len(newdf.get_group(key))] for key,value in newdf])

tempdf = catcount.sort_values(by=[1])

sns.set(style="white", context="talk",font_scale = 0.6)

sns.barplot(x=tempdf[0], y=tempdf[1])

plt.xticks(rotation= 90)

plt.ylabel("Crime Categorys")

plt.xlabel("Occurences")

plt.title("Number of Incidents per Category")
####  BarPlot to show how many incidents occur perday(Univariant analysis)  #####

newdfWeeks = trainMasterDf.groupby(['DayOfWeek'])

weekcount = pd.DataFrame([[key,len(newdfWeeks.get_group(key))] for key,value in newdfWeeks])

weektempdf = weekcount.sort_values(by=[1])

sns.set(style="white", context="talk",font_scale = 0.6)

sns.barplot(x=weektempdf[0], y=weektempdf[1])

plt.xticks(rotation= 90)

plt.ylabel("Crime Categorys")

plt.xlabel("Occurences")

plt.title("Number of Incidents per Category")
def conversion(masterDf):

	masterDf['Dates']= pd.to_datetime(masterDf['Dates'])

	masterDf['Year'] = masterDf['Dates'].dt.year

	masterDf['Month'] = masterDf['Dates'].dt.month

	masterDf['Day'] = masterDf['Dates'].dt.day

	masterDf['Hour'] = masterDf['Dates'].dt.hour

	masterDf = masterDf.drop(['Dates'],axis=1)

	return masterDf
### figure initialization  ###

sns.set(style="white", context="talk",font_scale = 5)

f, (ax) = plt.subplots(1, 1, figsize=(100, 100), sharex=True)



### Duplicate Removal ###

trainMasterDf = trainMasterDf.drop_duplicates()

### Dates coversion for train dataset ###

trainMasterDf = conversion(trainMasterDf)



#### plotting the incidents with respect to the hour time-series manner  ####

timeGroupDf = trainMasterDf.groupby(['Hour','Category'])

catcount = [[key,len(timeGroupDf.get_group(key))] for key,value in timeGroupDf]

unWrappeddf = pd.DataFrame([[x[0][0],x[0][1],x[1]] for x in catcount],columns=['Hours','Category','crimeCount'])

ax = sns.lineplot(x='Hours', y='crimeCount', hue='Category',data=unWrappeddf)

ax.legend(bbox_to_anchor=(0, 1),prop={'size': 50}, loc=2,ncol=3)
### plotting a FacetGrid for categroy wise crime counts for every hour ###

sns.set(style="white", context="talk",font_scale = 3)

unWrappeddf['crimeCount'] = (unWrappeddf.crimeCount/unWrappeddf.crimeCount.sum()) * 100

g = sns.FacetGrid(unWrappeddf, row="Category", aspect=4, height=20)

g = g.map(plt.bar, "Hours", "crimeCount")

plt.yticks(np.arange(0,1.5,0.1))
### Dates conversing for test dataset ###

testMasterDf = conversion(testMasterDf)
print(testMasterDf.columns)

print(trainMasterDf.columns)
### data transformations for algorithm specific ###

### For DaysofWeek ###

dayofWeekObj = preprocessing.LabelEncoder()

dayofWeekObj.fit(trainMasterDf.DayOfWeek)

trainMasterDf['DayOfWeekConverted'] = dayofWeekObj.transform(trainMasterDf.DayOfWeek)

testMasterDf['DayOfWeekConverted'] = dayofWeekObj.transform(testMasterDf.DayOfWeek)



### for PdDistrict ###

pdDistObj = preprocessing.LabelEncoder()

pdDistObj.fit(trainMasterDf.PdDistrict)

trainMasterDf['PdDistrictConverted'] = pdDistObj.transform(trainMasterDf.PdDistrict)

testMasterDf['PdDistrictConverted'] = pdDistObj.transform(testMasterDf.PdDistrict)



### for Category ###

cateObj = preprocessing.LabelEncoder()

cateObj.fit(trainMasterDf.Category)

trainMasterDf['CategoryConverted'] = cateObj.transform(trainMasterDf.Category)
testMasterDf.columns
randomClassifier = RandomForestClassifier(n_estimators=400, 

                                          max_depth=2,

                                          min_samples_split=50,

                                          random_state=0)

randomClassifier.fit(trainMasterDf[["X","Y","Year",'Month', 'Day','Hour', 'DayOfWeekConverted', 'PdDistrictConverted']],trainMasterDf['CategoryConverted'])
testMasterDf['Predictions'] = randomClassifier.predict(testMasterDf[["X","Y","Year",'Month', 'Day','Hour', 'DayOfWeekConverted', 'PdDistrictConverted']])
testMasterDf['probabilitys'] = randomClassifier.predict_proba(testMasterDf[["X","Y","Year",'Month', 'Day','Hour', 'DayOfWeekConverted', 'PdDistrictConverted']]).tolist()
testMasterDf['Predictions'] = cateObj.inverse_transform(testMasterDf['Predictions'])

submissiondf = testMasterDf[['Id','Predictions','probabilitys']]
submissiondf.head()