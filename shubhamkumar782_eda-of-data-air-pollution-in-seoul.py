import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
Image.open('../input/seoul-image/seoul_image.jpg')

data_air= pd.read_csv('../input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv')

data_air.head()
data_air.isnull()
%matplotlib inline
data_measure=pd.read_csv('../input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_item_info.csv')
data_measure # Standard value of all of the factors
# checking weather data has any missing value or not
sns.heatmap(data_air.isnull(),yticklabels=False,cbar=False,cmap='viridis')
data_air.info()
req_data=data_air[['SO2','NO2','O3','CO','PM10','PM2.5']]
req_data.describe()
plt.figure(figsize=(12,8))
sns.boxplot(data=req_data)
# converting 'SO2' value in their standard
def measure(x):
    if x<=0.02:
        return 'good'
    elif 0.02<x<=0.05:
        return 'normal'
    elif 0.05<x<=0.15:
        return 'bad'
    else:
        return 'very bad'
    
SO2_measure=list(map(measure,req_data['SO2']))
# converting 'NO2' value in their standard
def measure(x):
    if x<=0.03:
        return 'good'
    elif 0.03<x<=0.06:
        return 'normal'
    elif 0.06<x<=0.20:
        return 'bad'
    else:
        return 'very bad'
    
NO2_measure=list(map(measure,req_data['NO2']))
# converting 'O3' value in their standard
def measure(x):
    if x<=0.02:
        return 'good'
    elif 0.02<x<=0.05:
        return 'normal'
    elif 0.05<x<=0.15:
        return 'bad'
    else:
        return 'very bad'
    
O3_measure=list(map(measure,req_data['O3']))
# converting 'CO' value in their standard
def measure(x):
    if x<=2.00:
        return 'good'
    elif 2.00<x<=9.00:
        return 'normal'
    elif 9.00<x<=15.00:
        return 'bad'
    else:
        return 'very bad'
    
CO_measure=list(map(measure,req_data['CO']))
# converting 'PM10' value in their standard
def measure(x):
    if x<=30.00:
        return 'good'
    elif 30.00<x<=80.00:
        return 'normal'
    elif 80.00<x<=150.00:
        return 'bad'
    else:
        return 'very bad'
    
PM10_measure=list(map(measure,req_data['PM10']))
# converting 'PM2.5' value in their standard
def measure(x):
    if x<=15.00:
        return 'good'
    elif 15.00<x<=35.00:
        return 'normal'
    elif 35.00<x<=75.00:
        return 'bad'
    else:
        return 'very bad'
    
PM25_measure=list(map(measure,req_data['PM2.5']))
req_data['SO2_standard']=SO2_measure
req_data['NO2_standard']=NO2_measure
req_data['O3_standard']=O3_measure
req_data['CO_standard']=CO_measure
req_data['PM10_standard']=PM10_measure
req_data['PM2.5_standard']=PM25_measure


req_data
sns.countplot(x='SO2_standard',data=req_data)
sns.countplot(x='NO2_standard',data=req_data)
sns.countplot(x='O3_standard',data=req_data)
sns.countplot(x='CO_standard',data=req_data)
sns.countplot(x='PM10_standard',data=req_data)
sns.countplot(x='PM2.5_standard',data=req_data)
plt.figure(figsize=(8,8))
sns.heatmap(req_data.corr(),
            vmin=0,
            cmap='coolwarm',
            annot=True);
req_data['Address']=data_air["Address"]
req_data
data_address=data_air["Address"].value_counts()
data_address
plt.figure(figsize=(19,10))

sns.countplot(x="PM10_standard",data=req_data,hue="Address",orient="v")

plt.figure(figsize=(19,15))

sns.countplot(x="PM2.5_standard",data=req_data,hue="Address",orient="v")



