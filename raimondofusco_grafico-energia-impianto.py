# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests
from pandas.io.json import json_normalize
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
startDay = "2020-04-30"
dinoId = "DMETERW030001075"
hgmId= "hgm_0009963649330072029617923040254361835714"
#della giovanna
dinoId = 'DMETERW030000591'
hgmId= "hgm_0009963649330072029617923040254361835714"
#guaitamacchi
#dinoId = 'DMETERW030000431'
#hgmId = "0024649085242906001018691040254361828387"
#sponzilli
#dinoId = 'DMETERW030001075'
#hgmId = 'hgm_0020023682040410774823555040254361824824'
#dinoId = "DMETERW030000431"
task = {"dinoIds":  [dinoId], "startDateTime": startDay+"T00:00:00.000Z",  "endDateTime": startDay+"T23:59:00.000Z" }

resp = requests.post('http://104.199.91.69:8080/v1/energyDelta15M', json=task)

if resp.status_code == 200:
    respJson = resp.json()
    data = respJson["data"]
    dinoId =  data[0]["dinoId"]
    values = data[0]["values"]
    print(dinoId)
    #print(json_normalize(values))      
    df = pd.DataFrame(values)
    df['dateTime'] = pd.to_datetime(df['dateTime'], utc=False).dt.tz_localize(None)
    #df['dateTime'].dt.tz_localize(None)
    df_i = df.set_index('dateTime')    
    print(df_i)
    
    plt.figure(figsize=(20,6))
    sns.lineplot(data=df_i)
    
taskEvo = {"hgmId": hgmId, "startDate": startDay,  "endDate": startDay }

respEvoBess = requests.post('http://104.199.91.69:8082/api/query', json=taskEvo)
if respEvoBess.status_code == 200:
    respJson = respEvoBess.json()
    data = respJson["data"]
    dinoEnergyValues = data[0]['values']  
   
    energyTable= [
        (
        hour, 
         minute, 
         minute_dict['consumption']['powerActive']['delivered'],
         minute_dict['consumption']['powerActive']['absorbed'],
         minute_dict['consumption']['energyActive']['delivered'],
         minute_dict['consumption']['energyActive']['absorbed'],
         minute_dict['storage']['powerActive']['delivered'],
         minute_dict['storage']['powerActive']['absorbed'],
         minute_dict['storage']['energyActive']['delivered'],
         minute_dict['storage']['energyActive']['absorbed'],
         minute_dict['storage']['socPercentage']
        )
        for hour, hour_dict in dinoEnergyValues.items()
        for minute, minute_dict in hour_dict.items()
    ]
    evoBessData = pd.DataFrame.from_records(energyTable,
    columns=['hour', 'minute', 'consPwrDel', 'consPwrAbs','consEngDel', 'consEngAbs','stoPwrDel', 'stoPwrAbs','stoEngDel', 'stoEngAbs','socPercentage'])
    
    
    evoBessData['dateTime']=pd.to_datetime(startDay + ' ' + evoBessData['hour'] + ':' + evoBessData['minute'] )
    evoBessData = evoBessData.set_index('dateTime')
    socPercentage = evoBessData['socPercentage']
    evoBessData = evoBessData.drop(['hour', 'minute', 'consPwrDel', 'consPwrAbs','stoPwrDel', 'stoPwrAbs', 'socPercentage'], axis=1)
    plt.figure(figsize=(20,6))
    sns.lineplot(data=df_i)
plt.figure(figsize=(20,6))
sns.lineplot(dashes=False, data=evoBessData)
plt.figure(figsize=(20,6))
sns.lineplot(data=socPercentage, dashes=False,label="soc")