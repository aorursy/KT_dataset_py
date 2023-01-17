import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



dftemp = pd.read_csv('../input/datatokyo/data.csv',encoding='cp932', skiprows=4, usecols=[0,1])

dfcount = pd.read_csv('../input/datatokyo/count.csv',encoding='cp932', usecols=[0,1,3])

dfpositive = pd.read_csv('../input/datatokyo/positive.csv',encoding='cp932', usecols=[0,1])

dftemp
dfcount
dfpositive
dftemp.columns = ['date','energy']

dfcount.columns=['date','studycount1','studycount2']

dfpositive.columns = ['date',"positivecount"]

dftemp['date']=pd.to_datetime(dftemp['date'])

dfcount['date']=pd.to_datetime(dfcount['date'])

dfpositive['date']=pd.to_datetime(dfpositive['date'])

dfcount = dfcount.sort_values('date', ascending=True)

dfpositive = dfpositive.sort_values('date', ascending=True)

dftemp=dftemp[(dftemp['date'] >= pd.datetime(2020,3,15)) & (dftemp['date'] <= pd.datetime(2020,7,26))]

dfcount=dfcount[(dfcount['date'] >= pd.datetime(2020,3,15)) & (dfcount['date'] <= pd.datetime(2020,7,26))]

dfpositive=dfpositive[(dfpositive['date'] >= pd.datetime(2020,3,15)) & (dfpositive['date'] <= pd.datetime(2020,7,26))]

dfcount['studycount']=dfcount['studycount1']+dfcount['studycount2']
dftemp
dfcount
dfpositive
pd.set_option('display.max_rows', 500)

df1 = pd.merge(dftemp, dfcount)

df2 = pd.merge(df1, dfpositive)

df2['energy0']=(df2['energy']-df2['energy'].min())*8

df2['studycount0']=df2['studycount']*0.1

df2
df2['studycountmean']=df2['studycount0'].rolling(window=4).mean()

df2['studycountflat']=df2['studycount0']-df2['studycountmean']



df2['positivecountmean']=df2['positivecount'].rolling(window=4).mean()

df2['positivecountflat']=df2['positivecount']-df2['positivecountmean']





df2['energyflat']=df2['energy0'].rolling(window=4).mean()

df2['sunenergy']=df2['energy0']-df2['energyflat']



df2.plot(x='date', y=['studycount0','positivecount','energy0'],title='[covid-19] study count,positive count, energy of the sun in Tokyo',color=['green','blue','red'],figsize=(12,6))



df2.plot(x='date', y=['studycountflat','positivecountflat','sunenergy'],title='Data Correction : [covid-19] study count,positive count, energy of the sun in Tokyo',color=['green','blue','red'],figsize=(12,6))



df2.plot(x='date', y=['studycountflat','positivecountflat'],title='Data Correction : [covid-19] study count,positive count in Tokyo',color=['green','blue'],figsize=(12,6))

df2.plot(x='date', y=['studycountflat','sunenergy'],title='Data Correction : [covid-19] study count,energy of the sun in Tokyo',color=['green','red'],figsize=(12,6))

df2.plot(x='date', y=['positivecountflat','sunenergy'],title='Data Correction : [covid-19] positive count,energy of the sun in Tokyo',color=['blue','red'],figsize=(12,6))
