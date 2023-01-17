
from collections import defaultdict
import requests
import operator
import pandas as pd
import matplotlib.pyplot as plt
#import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True) 

r = requests.get('https://api.covid19india.org/raw_data.json')
jsonData = r.json()


listJsonData = jsonData.get('raw_data')
Coviddf = pd.DataFrame.from_dict(listJsonData, orient='columns')
#print(Coviddf)
RawCoviddf= Coviddf
RawCoviddf['dateannounced'] = pd.to_datetime(RawCoviddf['dateannounced'],format ='%d/%m/%Y')
RawCoviddf.dropna(subset=['detectedstate'],inplace = True)

## iyileşen hasta verileri
url1 = 'https://api.covid19india.org/states_daily_csv/recovered.csv'
recDf = pd.read_csv(url1)
recDf = recDf.iloc[:,0:recDf.shape[1]-1]
recDf.fillna(0,inplace = True)
# ölen hasta verileri
url2 = 'https://api.covid19india.org/states_daily_csv/deceased.csv'
decDf =pd.read_csv(url2)
decDf = decDf.iloc[:,0:decDf.shape[1]-1]
decDf.fillna(0,inplace = True)
# onaylanan hasta verileri
url3 = 'http://api.covid19india.org/states_daily_csv/confirmed.csv'
conDf = pd.read_csv(url3)
conDf = conDf.iloc[:,0:conDf.shape[1]-1]
conDf.fillna(0,inplace = True)
combinedDf = pd.DataFrame(columns= ['Date','State','Confirmed','Recovered','Dead'])
index = 0
for i in range(len(recDf)):
    for j in range(2,len(recDf.iloc[i])):
        if conDf.iloc[i,j]==0 and recDf.iloc[i,j]==0 and decDf.iloc[i,j]==0:
            continue
        record = [recDf['date'][i],stateDict.get(recDf.columns[j]),int(conDf.iloc[i,j]),int(recDf.iloc[i,j]),int(decDf.iloc[i,j])]
        combinedDf.loc[index]=record
        index=index+1
index = 0
for i in range(len(recDf)):
    for j in range(2,len(recDf.iloc[i])):
        if conDf.iloc[i,j]==0 and recDf.iloc[i,j]==0 and decDf.iloc[i,j]==0:
            continue
        record = [recDf['date'][i],stateDict.get(recDf.columns[j]),int(conDf.iloc[i,j]),int(recDf.iloc[i,j]),int(decDf.iloc[i,j])]
        combinedDf.loc[index]=record
        index=index+1

obj = datetime.strptime(combinedDf['Date'][len(combinedDf)-1],'%d-%b-%y')

# bir gün öncekine kadar verileri çekiyor
RawCoviddf = RawCoviddf[RawCoviddf['dateannounced']<=datetime.strftime(obj,'%Y-%m-%d')]
combinedDf['Date'] = pd.to_datetime(combinedDf['Date'],format ='%d-%b-%y')
stateCount = defaultdict(list)


for i in range(len(combinedDf)):

    
    value = combinedDf['Confirmed'][i] -(combinedDf['Recovered'][i]+combinedDf['Dead'][i])
    
    if combinedDf['State'][i] not in stateCount:
        stateCount[combinedDf['State'][i]].append(combinedDf['Confirmed'][i])
        stateCount[combinedDf['State'][i]].append(combinedDf['Recovered'][i])
        stateCount[combinedDf['State'][i]].append(combinedDf['Dead'][i])
        stateCount[combinedDf['State'][i]].append(value)
    else:
        stateCount[combinedDf['State'][i]][0]+=combinedDf['Confirmed'][i]
        stateCount[combinedDf['State'][i]][1]+=combinedDf['Recovered'][i]
        stateCount[combinedDf['State'][i]][2]+=combinedDf['Dead'][i]
        stateCount[combinedDf['State'][i]][3]+=value


# maksimum veriyi sıralamak için
stateCount = dict(sorted(stateCount.items(), key = lambda x :x[1][0], reverse=True ))
stateCount
Tabulation = pd.DataFrame.from_dict(stateCount,orient='index',columns=list(combinedDf.columns[2:])+['Active'])
Tabulation