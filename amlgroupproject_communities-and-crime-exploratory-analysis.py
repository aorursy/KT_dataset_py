import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as ss
#import vega_datasets
#matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
crimedata = pd.read_csv("../input/communities and crime unnormalized data set/crimedata.csv",sep='\s*,\s*',encoding='latin-1',engine='python',na_values=["?"])
statetoregion = pd.read_csv("../input/usa-states-to-region/states.csv")
statetoregion.head()
statetoregion.drop('Division',axis=1)
statetoregion = statetoregion.rename(columns={'State Code':'statecode'})
crimedata.head()
crimedata = crimedata.rename(columns={'Êcommunityname':'communityName'})
crimedata = crimedata.rename(columns={'state':'statecode'})
crimedata= pd.merge(crimedata, statetoregion, on='statecode', how='outer')

crimedata.head()
#crimedata.to_csv('completedata')
crimedata.to_csv('formatted.csv', mode='a',header=['communityName','statecode','countyCode','communityCode','fold','population','householdsize','racepctblack','racePctWhite','racePctAsian','racePctHisp','agePct12t21','agePct12t29','agePct16t24','agePct65up','numbUrban','pctUrban','medIncome','pctWWage','pctWFarmSelf','pctWInvInc','pctWSocSec','pctWPubAsst','pctWRetire','medFamInc','perCapInc','whitePerCap','blackPerCap','indianPerCap','AsianPerCap','OtherPerCap','HispPerCap','NumUnderPov','PctPopUnderPov','PctLess9thGrade','PctNotHSGrad','PctBSorMore','PctUnemployed','PctEmploy','PctEmplManu','PctEmplProfServ','PctOccupManu','PctOccupMgmtProf','MalePctDivorce','MalePctNevMarr','FemalePctDiv','TotalPctDiv','PersPerFam','PctFam2Par','PctKids2Par','PctYoungKids2Par','PctTeen2Par','PctWorkMomYoungKids','PctWorkMom','NumKidsBornNeverMar','PctKidsBornNeverMar','NumImmig','PctImmigRecent','PctImmigRec5','PctImmigRec8','PctImmigRec10','PctRecentImmig','PctRecImmig5','PctRecImmig8','PctRecImmig10','PctSpeakEnglOnly','PctNotSpeakEnglWell','PctLargHouseFam','PctLargHouseOccup','PersPerOccupHous','PersPerOwnOccHous','PersPerRentOccHous','PctPersOwnOccup','PctPersDenseHous','PctHousLess3BR','MedNumBR','HousVacant','PctHousOccup','PctHousOwnOcc','PctVacantBoarded','PctVacMore6Mos','MedYrHousBuilt','PctHousNoPhone','PctWOFullPlumb','OwnOccLowQuart','OwnOccMedVal','OwnOccHiQuart','OwnOccQrange','RentLowQ','RentMedian','RentHighQ','RentQrange','MedRent','MedRentPctHousInc','MedOwnCostPctInc','MedOwnCostPctIncNoMtg','NumInShelters','NumStreet','PctForeignBorn','PctBornSameState','PctSameHouse85','PctSameCity85','PctSameState85','LemasSwornFT','LemasSwFTPerPop','LemasSwFTFieldOps','LemasSwFTFieldPerPop','LemasTotalReq','LemasTotReqPerPop','PolicReqPerOffic','PolicPerPop','RacialMatchCommPol','PctPolicWhite','PctPolicBlack','PctPolicHisp','PctPolicAsian','PctPolicMinor','OfficAssgnDrugUnits','NumKindsDrugsSeiz','PolicAveOTWorked','LandArea','PopDens','PctUsePubTrans','PolicCars','PolicOperBudg','LemasPctPolicOnPatr','LemasGangUnitDeploy','LemasPctOfficDrugUn','PolicBudgPerPop','murders','murdPerPop','rapes','rapesPerPop','robberies','robbbPerPop','assaults','assaultPerPop','burglaries','burglPerPop','larcenies','larcPerPop','autoTheft','autoTheftPerPop','arsons','arsonsPerPop','ViolentCrimesPerPop','nonViolPerPop','State','Region','Division'])
crimedata.describe()

#Making word cloud for States based on the number of voilent and non voilent crime 

crimedata_state_violent = crimedata.groupby('State').agg({'ViolentCrimesPerPop':'mean'})[['ViolentCrimesPerPop']].reset_index()
crimedata_state_nonviolent = crimedata.groupby('State').agg({'nonViolPerPop':'mean'})[['nonViolPerPop']].reset_index()
crimedata_state_violent.dropna(inplace=True)
crimedata_state_nonviolent.dropna(inplace=True)
state_avg_violent_crime={}
for index,row in crimedata_state_violent.iterrows():   
   state_avg_violent_crime[row['State']]=int(row['ViolentCrimesPerPop']);

state_avg_nonviolent_crime={}
for index,row in crimedata_state_nonviolent.iterrows():   
   state_avg_nonviolent_crime[row['State']]=int(row['nonViolPerPop']);
    

import wordcloud
wc_violent = wordcloud.WordCloud(width=1000, height=500)
wc_violent.generate_from_frequencies(state_avg_violent_crime)

wc_nonviolent = wordcloud.WordCloud(width=1000, height=500)
wc_nonviolent.generate_from_frequencies(state_avg_nonviolent_crime)
plt.figure(figsize=(20,10))
plt.imshow(wc_violent, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud for State based on average Violent Crime")
plt.figure(figsize=(20,10))
plt.imshow(wc_nonviolent, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud for State based on average Non Violent Crime")
#Non violent crime by age%
ax1 = crimedata.plot(x='nonViolPerPop', y='agePct12t21', kind='scatter', c='red', s=2, label='Age%12-21')
ax2 = crimedata.plot(x='nonViolPerPop', y='agePct12t29', kind='scatter', c='green', s=2, label='Age%12-29', ax=ax1)
ax3 = crimedata.plot(x='nonViolPerPop', y='agePct16t24', kind='scatter', c='blue', s=2, label='Age%16-24', ax=ax2)
ax4 = crimedata.plot(x='nonViolPerPop', y='agePct65up', kind='scatter', c='black', s=2, label='Age%65up', ax=ax3)
plt.title('Non Violent Crimes by all age%')
plt.xlabel('Non Violent crimes per pop')
plt.ylabel('age%')
#Violent crime by age%
ax1 = crimedata.plot(x='ViolentCrimesPerPop', y='agePct12t21', kind='scatter', c='red', s=2, label='Age%12-21')
ax2 = crimedata.plot(x='ViolentCrimesPerPop', y='agePct12t29', kind='scatter', c='green', s=2, label='Age%12-29', ax=ax1)
ax3 = crimedata.plot(x='ViolentCrimesPerPop', y='agePct16t24', kind='scatter', c='blue', s=2, label='Age%16-24', ax=ax2)
ax4 = crimedata.plot(x='ViolentCrimesPerPop', y='agePct65up', kind='scatter', c='black', s=2, label='Age%65up', ax=ax3)
plt.title('Violent Crimes by all age%')
plt.xlabel('Violent crimes per pop')
plt.ylabel('age%')
#khaled
sns.stripplot(x='Region', y='ViolentCrimesPerPop', data=crimedata, jitter=True)
sns.stripplot(x='Region', y='nonViolPerPop', data=crimedata, jitter=True)
sns.lmplot(x='ViolentCrimesPerPop', y='nonViolPerPop', data=crimedata,
           fit_reg=True, #  regression line
           hue='Region',x_jitter=.1, y_jitter=0.1)   # Color by Region

import matplotlib.pyplot as plt
plt.figure(figsize=(20,15))
plt.subplot(3,2,1)
sns.swarmplot(x="whitePerCap", y="ViolentCrimesPerPop", hue="Region", data=crimedata)
plt.subplot(3,2,2)
sns.swarmplot(x="blackPerCap", y="ViolentCrimesPerPop", hue="Region", data=crimedata)
plt.subplot(3,2,3)
sns.swarmplot(x="indianPerCap", y="ViolentCrimesPerPop", hue="Region", data=crimedata)
plt.subplot(3,2,4)
sns.swarmplot(x="AsianPerCap", y="ViolentCrimesPerPop", hue="Region", data=crimedata)
plt.subplot(3,2,5)
sns.swarmplot(x="HispPerCap", y="ViolentCrimesPerPop", hue="Region", data=crimedata)
import matplotlib.pyplot as plt
plt.figure(figsize=(20,15))
plt.subplot(3,2,1)
sns.swarmplot(x="whitePerCap", y="nonViolPerPop", hue="Region", data=crimedata)
plt.subplot(3,2,2)
sns.swarmplot(x="blackPerCap", y="nonViolPerPop", hue="Region", data=crimedata)
plt.subplot(3,2,3)
sns.swarmplot(x="indianPerCap", y="nonViolPerPop", hue="Region", data=crimedata)
plt.subplot(3,2,4)
sns.swarmplot(x="AsianPerCap", y="nonViolPerPop", hue="Region", data=crimedata)
plt.subplot(3,2,5)
sns.swarmplot(x="HispPerCap", y="nonViolPerPop", hue="Region", data=crimedata)
import matplotlib.pyplot as plt
plt.figure(figsize=(20,15))
plt.subplot(3,2,1)
plt.title('whitePerCap')
crimedata['whitePerCap'].hist(bins='fd', density=True)
crimedata['whitePerCap'].plot(kind='kde')


plt.subplot(3,2,2)
plt.title('blackPerCap')
crimedata['blackPerCap'].hist(bins='fd', density=True)
crimedata['blackPerCap'].plot(kind='kde')

plt.subplot(3,2,3)
plt.title('indianPerCap')
crimedata['indianPerCap'].hist(bins='fd', density=True)
crimedata['indianPerCap'].plot(kind='kde')



plt.subplot(3,2,4)
plt.title('AsianPerCap')

crimedata['AsianPerCap'].hist(bins='fd', density=True)
crimedata['AsianPerCap'].plot(kind='kde')


plt.subplot(3,2,5)

plt.title('HispPerCap')
crimedata['HispPerCap'].hist(bins='fd', density=True)
crimedata['HispPerCap'].plot(kind='kde')
import matplotlib.pyplot as plt
plt.figure(figsize=(20,15))
plt.subplot(3,2,1)
sns.boxplot(x='Region', y='whitePerCap',data=crimedata)
plt.subplot(3,2,2)
sns.boxplot(x='Region', y='blackPerCap',data=crimedata)
plt.subplot(3,2,3)
sns.boxplot(x='Region', y='indianPerCap',data=crimedata)
plt.subplot(3,2,4)
sns.boxplot(x='Region', y='AsianPerCap',data=crimedata)
plt.subplot(3,2,5)
sns.boxplot(x='Region', y='HispPerCap',data=crimedata)