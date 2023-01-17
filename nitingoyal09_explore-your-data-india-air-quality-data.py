import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input/"))
import seaborn as sns
# Not specifying encoding will lead to Error: UnicodeDecodeError
rawdata = pd.read_csv("../input/data.csv", encoding='Windows-1252')
rawdata.head()
# with open("../input/data.csv", 'rb') as rawdata:
#     result = chardet.detect(rawdata.read(10000))
# print(result)
# 'date' and 'sampling_date' both represents the exact same record
data = rawdata.drop('sampling_date', axis = 1)
print(data.type.nunique())
data.type.unique()
# Replacing the multiple names given to the same area type

data = data.replace({'Residential, Rural and other Areas': 'Residential', 'Residential and others':'Residential','Industrial Area': 'Industrial',
                  'Industrial Areas':'Industrial', 'Sensitive Areas':'Sensitive', 'Sensitive Area': 'Sensitive'})
data.head()
# Rename 'type' to 'area', It sounds more sensible

data.type.unique()
data = data.rename(columns = {'type': 'area'})

data.head()
print('Que: Are dates in the dataset unique?\nAns: ' + str(rawdata.date.is_unique) + ', Only ' + str(rawdata.date.nunique()) + ' dates!')
print('Que: How many number of rows are in the dataset?\nAns: ' + str(rawdata.shape[0]))
# Rolling eyes over the agencies that are recording data in each state.

data.groupby(['state', 'agency']).count()
# Looking into a particular state
# Getting some intuition about the same 'date' values

AP = data[data.state == 'Arunachal Pradesh']
print('There is only ' + str(AP.area.nunique()) + ' area in AP.')
print(AP.date.is_unique)
# print('And the dates are not unique!')

# --- Checking if dates are related to the recording stations --- #

print('Que: How many stations are recordig data in AP?\nAns: ' + str(AP.stn_code.nunique()))
AP.groupby('stn_code').count()
print('Que: Are dates for a particular station Unique? \n' + 'Ans: ' + str(AP[AP.stn_code == 787.0].date.is_unique))
# Let's see if this is true for other states too.
Del = data[data.state == 'Delhi']
print('There are ' + str(Del.area.nunique()) + ' area in Del.')
print(Del.date.nunique())

print('Que: How many stations are recordig data in AP?\nAns: ' + str(Del.stn_code.nunique()))
Del.groupby('date').count()
print('Que: Are dates for a particular station Unique? \n' + 'Ans: ' + str(Del[Del.stn_code == 146].date.is_unique))
data.info()
data.isna().any()
# Format 'date' to datetime type and make a new column 'year'
# This is to plot yearly changes in levels of different pollutants

data['date'] = pd.to_datetime(data['date'],format='%Y-%m-%d') # date parse
data['year'] = data['date'].dt.year #year
print(data.shape)
data['year'] = data.year.dropna()
# print(data.shape) dropping nan
data.head()
df = data[['state', 'area', 'date', 'year']]
df.head()
df.area.fillna('others', inplace = True)
df.area.isna().any()
# Area specific air quality index visualization

#Function to calculate so2 individual pollutant index(si)
def calculate_si(so2):
    si=0
    if (so2<=40):
     si= so2*(50/40)
    if (so2>40 and so2<=80):
     si= 50+(so2-40)*(50/40)
    if (so2>80 and so2<=380):
     si= 100+(so2-80)*(100/300)
    if (so2>380 and so2<=800):
     si= 200+(so2-380)*(100/800)
    if (so2>800 and so2<=1600):
     si= 300+(so2-800)*(100/800)
    if (so2>1600):
     si= 400+(so2-1600)*(100/800)
    return si

#Function to calculate no2 individual pollutant index(ni)
def calculate_ni(no2):
    ni=0
    if(no2<=40):
     ni= no2*50/40
    elif(no2>40 and no2<=80):
     ni= 50+(no2-14)*(50/40)
    elif(no2>80 and no2<=180):
     ni= 100+(no2-80)*(100/100)
    elif(no2>180 and no2<=280):
     ni= 200+(no2-180)*(100/100)
    elif(no2>280 and no2<=400):
     ni= 300+(no2-280)*(100/120)
    else:
     ni= 400+(no2-400)*(100/120)
    return ni

#Function to calculate no2 individual pollutant index(rpi)
def calculate_(rspm):
    rpi=0
    if(rpi<=30):
     rpi=rpi*50/30
    elif(rpi>30 and rpi<=60):
     rpi=50+(rpi-30)*50/30
    elif(rpi>60 and rpi<=90):
     rpi=100+(rpi-60)*100/30
    elif(rpi>90 and rpi<=120):
     rpi=200+(rpi-90)*100/30
    elif(rpi>120 and rpi<=250):
     rpi=300+(rpi-120)*(100/130)
    else:
     rpi=400+(rpi-250)*(100/130)
    return rpi

#Function to calculate no2 individual pollutant index(spi)
def calculate_spi(spm):
    spi=0
    if(spm<=50):
     spi=spm
    if(spm<50 and spm<=100):
     spi=spm
    elif(spm>100 and spm<=250):
     spi= 100+(spm-100)*(100/150)
    elif(spm>250 and spm<=350):
     spi=200+(spm-250)
    elif(spm>350 and spm<=450):
     spi=300+(spm-350)*(100/80)
    else:
     spi=400+(spm-430)*(100/80)
    return spi

def calculate_aqi(si,ni,spi,rpi):
    aqi=0
    if(si>ni and si>spi and si>rpi):
     aqi=si
    if(spi>si and spi>ni and spi>rpi):
     aqi=spi
    if(ni>si and ni>spi and ni>rpi):
     aqi=ni
    if(rpi>si and rpi>ni and rpi>spi):
     aqi=rpi
    return aqi
#AQI calculations
data = data.fillna(0)
df['si']=data['so2'].apply(calculate_si)
df['ni']=data['no2'].apply(calculate_ni)
df['rpi']=data['rspm'].apply(calculate_si)
df['spi']=data['spm'].apply(calculate_spi)
df['AQI']=df.apply(lambda x:calculate_aqi(x['si'],x['ni'],x['spi'],x['rpi']),axis=1)
df.head()
df = df[['state', 'date', 'area', 'AQI', 'year']]
df_res = df[df.area=='Residential'].reset_index().drop('index', axis=1)
# data_ind = df[df.area == 'Industrial']
# data_sens = df[df.area == 'Sensitive']
# data_riruo = df[df.area == 'RIRUO']
df_res.tail()
# Check for Outliers
fig, ax = plt.subplots(2,2, figsize=(30,10))

df1 = df_res[df_res.state == 'Delhi'].groupby('date').mean().reset_index()
df2 = df_res[df_res.state == 'Rajasthan'].groupby('date').mean().reset_index()
df3 = df_res[df_res.state == 'Andhra Pradesh'].groupby('date').mean().reset_index()
df4 = df_res[df_res.state == 'West Bengal'].groupby('date').mean().reset_index()

sns.boxplot(x = df1.AQI, ax = ax[0][0])
sns.boxplot(x = df2.AQI, ax = ax[0][1])
sns.boxplot(x = df3.AQI, ax = ax[1][0])
sns.boxplot(x = df4.AQI, ax = ax[1][1])

plt.show()
# Data has outliers we can see that. 
# but for the sake of data exploration we will leave this task to preprocessing phase where outliers could be analysed.
# top 10 polluted states
pollution_list = df_res.groupby(['state', 'year']).median().sort_values(by='AQI',ascending=False).reset_index()

print(pollution_list.nsmallest(10, 'AQI')['state'])
print(pollution_list.nlargest(10, 'AQI')['state'])

# these are over all those years, one can say that what do you want to check out of these.
# Well, these data points can be visualized in a heat map over the geographical map of India.
# We can visualize excatly how pollution was increased over the geographical regions of India.
