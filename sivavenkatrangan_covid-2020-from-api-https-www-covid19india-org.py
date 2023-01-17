!pip install requests
import requests
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
response = requests.get('https://api.covid19india.org/raw_data.json')
response.status_code
data = response.text
parsed = json.loads(data)
#json.dumps(parsed, indent=4)
data= parsed['raw_data']
thisdict = data[0]
print("-------------------------------------------------------------------")
for x, y in thisdict.items():
    print(x)

print("-------------------------------------------------------------------")
lists = []
for x, y in thisdict.items():
    print("ds['"+x+"']"+"="+x)
    
print("-------------------------------------------------------------------")

lists = []
for x, y in thisdict.items():
    print( x+".append(data[i]['"+x+"'])")
   
lists = []

for x, y in thisdict.items():
    lists.append(x)
    
print(lists)
data = parsed["raw_data"]
ds= pd.DataFrame()


agebracket = []  
backupnotes = []
contractedfromwhichpatientsuspected =[]
currentstatus =[]
dateannounced=[]
detectedcity=[]
detecteddistrict=[]
detectedstate=[]
estimatedonsetdate=[]
gender=[]
nationality=[]
notes=[]
patientnumber=[]
source1=[]
source2=[]
source3=[]
statecode=[]
statepatientnumber=[]
statuschangedate=[]
typeoftransmission=[]

for i in range(len(data)):
    agebracket.append(data[i]['agebracket'])
    backupnotes.append(data[i]['backupnotes'])
    contractedfromwhichpatientsuspected.append(data[i]['contractedfromwhichpatientsuspected'])
    currentstatus.append(data[i]['currentstatus'])
    dateannounced.append(data[i]['dateannounced'])
    detectedcity.append(data[i]['detectedcity'])
    detecteddistrict.append(data[i]['detecteddistrict'])
    detectedstate.append(data[i]['detectedstate'])
    estimatedonsetdate.append(data[i]['estimatedonsetdate'])
    gender.append(data[i]['gender'])
    nationality.append(data[i]['nationality'])
    notes.append(data[i]['notes'])
    patientnumber.append(data[i]['patientnumber'])
    source1.append(data[i]['source1'])
    source2.append(data[i]['source2'])
    source3.append(data[i]['source3'])
    statecode.append(data[i]['statecode'])
    statepatientnumber.append(data[i]['statepatientnumber'])
    statuschangedate.append(data[i]['statuschangedate'])
    typeoftransmission.append(data[i]['typeoftransmission'])

ds['agebracket']=agebracket
ds['backupnotes']=backupnotes
ds['contractedfromwhichpatientsuspected']=contractedfromwhichpatientsuspected
ds['currentstatus']=currentstatus
ds['dateannounced']=dateannounced
ds['detectedcity']=detectedcity
ds['detecteddistrict']=detecteddistrict
ds['detectedstate']=detectedstate
ds['estimatedonsetdate']=estimatedonsetdate
ds['gender']=gender
ds['nationality']=nationality
ds['notes']=notes
ds['patientnumber']=patientnumber
ds['source1']=source1
ds['source2']=source2
ds['source3']=source3
ds['statecode']=statecode
ds['statepatientnumber']=statepatientnumber
ds['statuschangedate']=statuschangedate
ds['typeoftransmission']=typeoftransmission
ds.tail()
ds.head()
ds.info()
ds['agebracket'] = pd.to_numeric(ds['agebracket'], errors='coerce')
ds['dateannounced'] = pd.to_datetime(ds['dateannounced'], dayfirst=True)
#ds['dateannounced'] = ds['dateannounced'].dt.strftime('%m/%d/%Y')
ds.shape
ds.describe(include='all')
ds=ds[ds['detectedstate']!=""]
ds.head()
dsbyState =pd.DataFrame(ds.groupby('detectedstate')['detectedstate'].agg('count'))
dsbyState = dsbyState.rename(columns={'detectedstate':'Count'}).reset_index()
dsbyState.sort_values(by='Count', ascending=False, inplace=True)
plt.figure(figsize=(12,8))
sns.barplot(x='Count', y='detectedstate', data=dsbyState, color='red')
dsbyState.head()
dsbyStateDate =pd.DataFrame(ds.groupby(['dateannounced','detectedstate'])['detectedstate'].agg('count'))
dsbyStateDate = dsbyStateDate.rename(columns={'detectedstate':'Count'})
dsbyStateDate.reset_index(inplace=True)
dsbyState_today = dsbyStateDate[dsbyStateDate['dateannounced']=='2020-04-14'].sort_values('Count', ascending=False)
dsbyState_today
dsbyState_today['Count'].agg('sum')
plt.figure(figsize=(12,8))
sns.barplot(x='Count', y='detectedstate', data=dsbyState_today, color='red')
dsbyDate =pd.DataFrame(ds.groupby(['dateannounced'])['detectedstate'].agg('count'))
dsbyDate = dsbyDate.rename(columns={'detectedstate':'Count'})
dsbyDate.reset_index(inplace=True)
dsbyDate.head()
dsbyDate['Count'][10]
Cumulative = []
n = 0

for i in range(len(dsbyDate)):
    n = dsbyDate['Count'][i]+n
    Cumulative.append(n)
print(Cumulative)    
dsbyDate['Cumulative'] = Cumulative
dsbyDate.head()
dsbyDate['date'] = dsbyDate['dateannounced'].dt.strftime('%m/%d')
plt.figure(figsize=(25,8))
sns.barplot(x='date', y='Count', data=dsbyDate, color='blue')
dsbyDate.tail()
plt.figure(figsize=(25,8))
sns.lineplot(x='date', y='Cumulative', data=dsbyDate, color='red')
plt.figure(figsize=(25,8))
sns.barplot(x='date', y='Count', data=dsbyDate, color='blue')
sns.lineplot(x='date', y='Cumulative', data=dsbyDate, color='red')
(ds['gender']!="").sum()
ds['gender'].value_counts()
ds['agebracket'].value_counts()
ds['currentstatus'].value_counts()
dsbyDate['weekday']= dsbyDate['dateannounced'].dt.strftime('%a')
byweekCovid = pd.DataFrame(dsbyDate.groupby('weekday')['Count'].agg('sum'))
byweekCovid['Average']=dsbyDate.groupby('weekday')['Count'].agg('mean')
byweekCovid['reindex'] = [7,3,1,2,6,4,5]
byweekCovid=byweekCovid.sort_values(['reindex'])
byweekCovid.reset_index(inplace=True)
byweekCovid.drop('reindex', axis=1)

plt.figure(figsize=(13,6))
plt.title('Week by week affected with COVID')
sns.barplot(x='weekday', y='Count', data=byweekCovid)
plt.figure(figsize=(13,6))
plt.title('Average Week by week affected with COVID')
plt.tick_params('both')
plt.xlabel('Testing')
sns.barplot(x='weekday', y='Average', data=byweekCovid)
dsbyStateDate.head()
state = ds['detectedstate'].drop_duplicates()
state.reset_index(drop=True, inplace=True)
state.shape
print(state)
dsbyState_Iv = dsbyStateDate[dsbyStateDate['detectedstate']=='Telangana']
dsbyState_Iv.reset_index(drop=True, inplace=True)
dsbyState_Iv.info()
Cumulative = []
j = 0
for i in range(len(dsbyState_Iv)):
    j = (dsbyState_Iv['Count'][i]+j)
    Cumulative.append(j)
    
print(Cumulative)
dsbyState_Iv['Cumulative']= Cumulative
from datetime import datetime  
from datetime import timedelta
dsbyState_Iv['to_be_recovered']= dsbyState_Iv['dateannounced']+ timedelta(days=14)
dsbyState_Iv
print(dsbyState_Iv['Count'].mean())
print(dsbyState_Iv['Count'].median())
print(dsbyState_Iv['Count'].std())
plt.figure(figsize= (15,5))
sns.distplot(dsbyState_Iv['Count'])
plt.figure(figsize= (15,5))
sns.lineplot('dateannounced', 'Count', data=dsbyState_Iv)
plt.figure(figsize= (15,5))
sns.lineplot(x='dateannounced', y='Cumulative', data=dsbyState_Iv)
sns.lineplot(x='to_be_recovered', y='Cumulative', data=dsbyState_Iv)
dsbyState_Iv['date'] = dsbyState_Iv['dateannounced'].dt.strftime('%m/%d')
plt.figure(figsize= (20,5))
sns.barplot(x='date', y='Count', data=dsbyState_Iv)





