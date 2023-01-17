# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/covid19-in-india/covid_19_india.csv")
agedata = pd.read_csv("/kaggle/input/covid19-in-india/AgeGroupDetails.csv")
icmrtest = pd.read_csv("/kaggle/input/covid19-in-india/ICMRTestingDetails.csv")
statetest = pd.read_csv("/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv")
beddata = pd.read_csv("/kaggle/input/covid19-in-india/HospitalBedsIndia.csv")
pd.unique(data["State/UnionTerritory"])
beddata.head(10)
import matplotlib.pyplot as plt
plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
plt.plot(data[data["State/UnionTerritory"]=='Kerala']['Date'],data[data["State/UnionTerritory"]=='Kerala']['Confirmed'],label='Confirmed')
plt.plot(data[data["State/UnionTerritory"]=='Kerala']['Date'],data[data["State/UnionTerritory"]=='Kerala']['Deaths'],label='Deaths')
plt.plot(data[data["State/UnionTerritory"]=='Kerala']['Date'],data[data["State/UnionTerritory"]=='Kerala']['Cured'],label='Cured')
plt.legend()
plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
plt.plot(data[data["State/UnionTerritory"]=='Tamil Nadu']['Date'],data[data["State/UnionTerritory"]=='Tamil Nadu']['Confirmed'],label='Confirmed')
plt.plot(data[data["State/UnionTerritory"]=='Tamil Nadu']['Date'],data[data["State/UnionTerritory"]=='Tamil Nadu']['Deaths'],label='Deaths')
plt.plot(data[data["State/UnionTerritory"]=='Tamil Nadu']['Date'],data[data["State/UnionTerritory"]=='Tamil Nadu']['Cured'],label='Cured')
plt.legend()
plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
plt.plot(data[data["State/UnionTerritory"]=='Maharashtra']['Date'],data[data["State/UnionTerritory"]=='Maharashtra']['Confirmed'],label='Confirmed')
plt.plot(data[data["State/UnionTerritory"]=='Maharashtra']['Date'],data[data["State/UnionTerritory"]=='Maharashtra']['Deaths'],label='Deaths')
plt.plot(data[data["State/UnionTerritory"]=='Maharashtra']['Date'],data[data["State/UnionTerritory"]=='Maharashtra']['Cured'],label='Cured')
plt.legend()
plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
plt.plot(data[data["State/UnionTerritory"]=='Delhi']['Date'],data[data["State/UnionTerritory"]=='Delhi']['Confirmed'],label='Confirmed')
plt.plot(data[data["State/UnionTerritory"]=='Delhi']['Date'],data[data["State/UnionTerritory"]=='Delhi']['Deaths'],label='Deaths')
plt.plot(data[data["State/UnionTerritory"]=='Delhi']['Date'],data[data["State/UnionTerritory"]=='Delhi']['Cured'],label='Cured')
plt.legend()
plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
plt.plot(data[data["State/UnionTerritory"]=='West Bengal']['Date'],data[data["State/UnionTerritory"]=='West Bengal']['Confirmed'],label='Confirmed')
plt.plot(data[data["State/UnionTerritory"]=='West Bengal']['Date'],data[data["State/UnionTerritory"]=='West Bengal']['Deaths'],label='Deaths')
plt.plot(data[data["State/UnionTerritory"]=='West Bengal']['Date'],data[data["State/UnionTerritory"]=='West Bengal']['Cured'],label='Cured')
plt.legend()
plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
plt.plot(statetest[statetest['State']=='Kerala']['Date'],statetest[statetest['State']=='Kerala']['TotalSamples'],label='KR-Total')
plt.plot(statetest[statetest['State']=='Karnataka']['Date'],statetest[statetest['State']=='Karnataka']['TotalSamples'],label='KA-Total')
plt.plot(statetest[statetest['State']=='Maharashtra']['Date'],statetest[statetest['State']=='Maharashtra']['TotalSamples'],label='MH-Total')
plt.plot(statetest[statetest['State']=='Delhi']['Date'],statetest[statetest['State']=='Delhi']['TotalSamples'],label='DL-Total')
plt.legend()
plt.show()

plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
plt.plot(statetest[statetest['State']=='Kerala']['Date'],statetest[statetest['State']=='Kerala']['Positive'],label='KR-Positive')
plt.plot(statetest[statetest['State']=='Karnataka']['Date'],statetest[statetest['State']=='Karnataka']['Positive'],label='KA-Positive')
plt.plot(statetest[statetest['State']=='Maharashtra']['Date'],statetest[statetest['State']=='Maharashtra']['Positive'],label='MH-Positive')
plt.plot(statetest[statetest['State']=='Delhi']['Date'],statetest[statetest['State']=='Delhi']['Positive'],label='DL-Positive')

plt.legend()
plt.show()
ndict={}
plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
for state in pd.unique(statetest['State'])[0:4]:
    test2positiveRatio = statetest[statetest['State']==state]['Positive']/statetest[statetest['State']==state]['TotalSamples']
    t2p = test2positiveRatio.tolist()
    plt.plot(statetest[statetest['State']==state]['Date'],np.array(t2p).reshape(len(t2p),1),label=state+"-postive to total tests ratio")
plt.legend()
plt.show()
beddata.columns
beds = beddata[['State/UT','NumPublicBeds_HMIS','NumRuralBeds_NHP18','NumUrbanBeds_NHP18']][0:36]
total = []
for i in range(0,len(beds)):
    total.append(beds.loc[i]['NumPublicBeds_HMIS']+beds.loc[i]['NumRuralBeds_NHP18']+beds.loc[i]['NumUrbanBeds_NHP18'])
beds['total'] = total
beds
beds = beds.sort_values(by=['total'],ascending=False)
beds[0:7]
beds = beds.sort_values(by=['NumUrbanBeds_NHP18'],ascending=False)
beds[0:7]
beds = beds.sort_values(by=['NumRuralBeds_NHP18'],ascending=False)
beds[0:7]
beds = beds.sort_values(by=['NumPublicBeds_HMIS'],ascending=False)
beds[0:7]
# beds.index = beds['State/UT'].values
# beds = beds.drop('State/UT',axis=1)
# beds
beds = beds.sort_values(by='total',ascending=True)
beds[['NumPublicBeds_HMIS','NumRuralBeds_NHP18','NumUrbanBeds_NHP18']].plot(kind='barh',stacked=True,figsize=(8,12),grid=True)
census = pd.read_csv("/kaggle/input/covid19-in-india/population_india_census2011.csv")
import re
den = []
for x in census['Density'].to_list():
    x = x.replace(",","")
    den.append(float(re.findall("\d+[.]?\d*",x)[0]))
census['den']=den
m = census.sort_values(by="Population",ascending=True)
m.index = m['State / Union Territory'].values
m=m.drop('State / Union Territory',axis=1)
m1=m[['Rural population','Urban population']]
m1.plot(kind='barh',stacked=True,grid=True,figsize=(8,12))
m = census.sort_values(by="Rural population",ascending=True)
plt.figure(figsize=(8,10))
plt.barh(m['State / Union Territory'].values,m['Rural population'].values)
plt.show()
m = census.sort_values(by="Urban population",ascending=True)
plt.figure(figsize=(8,10))
plt.barh(m['State / Union Territory'].values,m['Urban population'].values,color="green")
plt.show()
m = census.sort_values(by="den",ascending=True)
plt.figure(figsize=(8,10))
plt.barh(m['State / Union Territory'].values,m['den'].values)
plt.show()
dd = pd.unique(data['Date'])
ld = data[data['Date']==dd[-1]]
ld
mortality_rate = ld['Deaths']/ld['Confirmed']*100
ld['mr']=mortality_rate
curing_rate = ld['Cured']/ld['Confirmed']*100
ld['cr']=curing_rate
ld
import math
ts=[]
for i in ld.index:
    if ld.loc[i]['Confirmed']!=0:
        ts.append(math.log(ld.loc[i]['Confirmed'],10)*(2/(ld.loc[i]['mr']+1))/100*ld.loc[i]['cr']/100)
    else:
        ts.append(0.0)
ts
ld["ts"]=ts
ld.sort_values(by="ts",ascending=False)
left = pd.DataFrame({"Cured":ld['Cured'].values,"Deaths":ld['Deaths'].values,"Confirmed":ld['Confirmed'].values},index=ld['State/UnionTerritory'].values)
right = pd.DataFrame({"Population":census['Population'].values},index=census['State / Union Territory'].values)

left = left.join(right,how="outer")
left
left = left.drop("Nagaland#",axis=0)
left
left
left.fillna(0.0,inplace=True)
infected_per = left['Confirmed'].values/left['Population']*100
# infected_per
left['infected_per']=infected_per
left = left.sort_values(by="infected_per",ascending=True)
plt.figure(figsize=(8,12))
plt.barh(left.index,left['infected_per'].values)