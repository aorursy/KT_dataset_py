import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


dftemp = pd.read_csv('/kaggle/input/tokyodata/data.csv',encoding='cp932', skiprows=4, usecols=[0,1,4,7,11,14,17])
dfcovt = pd.read_csv('/kaggle/input/tokyodata/130001_tokyo_covid19_patients.csv', usecols=[4])
dfstudy = pd.read_csv('/kaggle/input/tokyodata/kensasuu.csv', usecols=[0,1,3])

dftemp.columns = ['date','temperature','humidity','sun','wind','energy','rain']
dfcovt.columns = ['date']
dfstudy.columns=['date','studycount1','studycount2']
dftemp['date']=pd.to_datetime(dftemp['date'])
dfcovt['date']=pd.to_datetime(dfcovt['date'])
dfstudy['date']="2020/"+dfstudy['date']
dfstudy['date']=pd.to_datetime(dfstudy['date'])
dfstudy = dfstudy.sort_values('date', ascending=True)
dftemp=dftemp[(dftemp['date'] >= pd.datetime(2020,3,15)) & (dftemp['date'] <= pd.datetime(2020,5,9))]
dfcovt=dfcovt[(dfcovt['date'] >= pd.datetime(2020,3,15)) & (dfcovt['date'] <= pd.datetime(2020,5,9))]
dfstudy=dfstudy[(dfstudy['date'] >= pd.datetime(2020,3,15)) & (dfstudy['date'] <= pd.datetime(2020,5,9))]
dfstudy['studycount']=dfstudy['studycount1']+dfstudy['studycount2']
dftemp
dfstudy
dfcovtcnt=dfcovt.groupby('date').size()
dfcovtcnt
dfcnt = pd.DataFrame(dfcovtcnt)
dfcnt.columns = ['positivecount']
dfcnt = dfcnt.reset_index()
dfcnt
pd.set_option('display.max_rows', 500)
df1 = pd.merge(dftemp, dfcnt)
df2 = pd.merge(df1, dfstudy)
df2['sun0']=(df2['sun']-df2['sun'].min())*6
df2['wind0']=(df2['wind']-df2['wind'].min())*32
df2['energy0']=(df2['energy']-df2['energy'].min())*4
df2['humidity0']=(df2['humidity']-df2['humidity'].min())*2
df2['rain0']=df2['rain']*0.6;
df2['studycount0']=df2['studycount']*0.1
df2
df2['studycountmean']=df2['studycount0'].rolling(window=4).mean()
df2['studycountflat']=df2['studycount0']-df2['studycountmean']

df2['positivecountmean']=df2['positivecount'].rolling(window=4).mean()
df2['positivecountflat']=df2['positivecount']-df2['positivecountmean']


df2['energyflat']=df2['energy0'].rolling(window=4).mean()
df2['sunenergy']=df2['energy0']-df2['energyflat']

df2['sunflat']=df2['sun0'].rolling(window=4).mean()
df2['suntime']=df2['sun0']-df2['sunflat']

df2['humidflat']=df2['humidity0'].rolling(window=4).mean()
df2['humid']=df2['humidity0']-df2['humidflat']

df2.plot(x='date', y=['studycount0','positivecount','energy0'],title='[covid-19] study count,positive count, energy of the sun in Tokyo',color=['green','blue','red'],figsize=(12,6))


df2.plot(x='date', y=['studycountflat','positivecountflat','sunenergy'],title='Data Correction : [covid-19] study count,positive count, energy of the sun in Tokyo',color=['green','blue','red'],figsize=(12,6))


df2.plot(x='date', y=['studycountflat','positivecountflat'],title='Data Correction : [covid-19] study count,positive count in Tokyo',color=['green','blue'],figsize=(12,6))

df2.plot(x='date', y=['studycountflat','sunenergy'],title='Data Correction : [covid-19] study count,energy of the sun in Tokyo',color=['green','red'],figsize=(12,6))

df2.plot(x='date', y=['positivecountflat','sunenergy'],title='Data Correction : [covid-19] positive count,energy of the sun in Tokyo',color=['blue','red'],figsize=(12,6))

