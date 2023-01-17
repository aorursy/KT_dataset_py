import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import math

import warnings

%matplotlib inline

warnings.filterwarnings('ignore')
data=pd.read_csv('../input/covid19-corona-virus-india-dataset/complete.csv')
data.head()
data['Date']=pd.to_datetime(data['Date'])
data.set_index("Date",inplace=True)
data.head()
data['Name of State / UT'].unique()
data.describe()
TN=data[data['Name of State / UT']=='Tamil Nadu']
TN.head()
TN['Today_case']=0

for i in TN.index:

    a=i-pd.DateOffset(days=1)

    if a not in TN.index:

        a=i

    TN['Today_case'].loc[i]=(TN['Total Confirmed cases'].loc[i])-(TN['Total Confirmed cases'].loc[a])
TN.tail(10)
import math

TN['Transmission_rate']=0

for i in TN.index:

    a=i-pd.DateOffset(days=1)

    if a not in TN.index:

        a=i

    TN['Transmission_rate'].loc[i]=(TN['Today_case'].loc[i])/(TN['Today_case'].loc[a])

    if TN['Transmission_rate'].loc[i]==math.inf:

        TN['Transmission_rate'].loc[i]=0
TN.tail()
TN['Transmission_rate'].fillna(value=0)
trans_start=TN['Transmission_rate'].loc['2020-03-23':]

trans_start
overall_trans=trans_start.mean()

overall_trans
TN['No_lock_affected']=0

TN['No_lock_affected'].loc['2020-03-23']=TN['Total Confirmed cases'].loc['2020-03-23']

for i in TN.index:

    if i in pd.date_range('2020-03-24',freq='D',periods=30):

        a=i-pd.DateOffset(days=1)

        if a not in TN.index:

            a=i

        TN['No_lock_affected'].loc[i]=TN['No_lock_affected'].loc[a]*overall_trans
TN.head()
fig=plt.figure(figsize=(20,5))

plt.plot(TN.index,TN['Total Confirmed cases'],'g',label='Actual Case')

plt.plot(TN.index,TN['No_lock_affected'],'r',label='Cases without lockdown')

plt.title('Tamil Nadu COVID-19 Cases')

plt.legend()
def data_inclusion(data):

    data['Today_case']=0

    for i in data.index:

        a=i-pd.DateOffset(days=1)

        if a not in data.index:

            a=i

        data['Today_case'].loc[i]=(data['Total Confirmed cases'].loc[i])-(data['Total Confirmed cases'].loc[a])

        

    data['Transmission_rate']=0

    for i in data.index:

        a=i-pd.DateOffset(days=1)

        if a not in data.index:

            a=i

        data['Transmission_rate'].loc[i]=data['Today_case'].loc[i]/data['Today_case'].loc[a]

        if data['Transmission_rate'].loc[i]==math.inf:

            data['Transmission_rate'].loc[i]=0

    data['Transmission_rate'].fillna(value=0,inplace=True)

    return data
def without_lock(data):

    trans_start=data['Transmission_rate'].loc['2020-03-30':]

    overall_trans=trans_start.mean()

    if overall_trans<1.0:

        overall_trans=overall_trans+0.4

    

    data['No_lock_affected']=0

    data['No_lock_affected'].loc['2020-03-23']=data['Total Confirmed cases'].loc['2020-03-23']

    for i in data.index:

        if i in pd.date_range('2020-03-24',freq='D',periods=30):

            a=i-pd.DateOffset(days=1)

            if a not in data.index:

                a=i

            data['No_lock_affected'].loc[i]=data['No_lock_affected'].loc[a]*overall_trans

    return data,overall_trans
def graph(data):

    fig=plt.figure(figsize=(20,5))

    plt.plot(data.index,data['Total Confirmed cases'],'g',label='Actual Case'),

    plt.plot(data.index,data['No_lock_affected'],'r',label='Cases without lockdown'),

    plt.title(data['Name of State / UT'].unique() +' COVID-19 Cases')

    plt.legend()
def prediction(data):

    data_inclusion(data)

    without_lock(data)

    graph(data)
MH=data[data['Name of State / UT']=='Maharashtra']

MH.head()
prediction(MH)
DL=data[data['Name of State / UT']=='Delhi']

DL.head()
prediction(DL)
RJ=data[data['Name of State / UT']=='Rajasthan']

RJ.head()
prediction(RJ)
TL=data[data['Name of State / UT']=='Telengana']

TL.head()
prediction(TL)
MP=data[data['Name of State / UT']=='Madhya Pradesh']

MP.head()
prediction(MP)
GJ=data[data['Name of State / UT']=='Gujarat']

GJ.head()
prediction(GJ)
UP=data[data['Name of State / UT']=='Uttar Pradesh']

UP.head()
prediction(UP)
KL=data[data['Name of State / UT']=='Kerala']

KL.head()
prediction(KL)
KA=data[data['Name of State / UT']=='Karnataka']

KA.head()
prediction(KA)