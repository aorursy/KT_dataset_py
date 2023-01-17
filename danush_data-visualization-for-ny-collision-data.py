import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

from pandas import read_csv

import warnings

import seaborn as sns

%matplotlib inline

accident_data_NY=read_csv('../input/database.csv')

accident_data_NY.head()

train=accident_data_NY.sample(frac=0.3,random_state=100)

accident_data_NY['DATE']=pd.to_datetime(accident_data_NY['DATE'])

accident_data_NY['Year']=accident_data_NY['DATE'].dt.year

accident_data_NY['Month']=accident_data_NY['DATE'].dt.month

month_map={1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

accident_data_NY['Month'].replace(month_map,inplace=True)

accident_data_NY['Month']=accident_data_NY['DATE'].dt.month

Boroughdata=pd.DataFrame(accident_data_NY['BOROUGH'].value_counts())

accident_data_NY.head()

accident_data_NY.head()

plt.figure(figsize=(12,12))

ax1=plt.subplot2grid((2,2),(0,0))

sns.barplot(x=Boroughdata.index,y='BOROUGH',data=Boroughdata)

YearlyData=pd.DataFrame(accident_data_NY['Year'].value_counts())

ax2=plt.subplot2grid((2,2),(0,1))

sns.barplot(x=YearlyData.index,y='Year',data=YearlyData)

accident_data_NY["TIME"]=pd.to_datetime(accident_data_NY['TIME'])

accident_data_NY['Hour']=accident_data_NY['TIME'].dt.hour 

HourlyData=pd.DataFrame(accident_data_NY["Hour"].value_counts())

plt.figure(figsize=(24,12))

ax1=plt.subplot2grid((2,2),(0,0))

sns.barplot(x=HourlyData.index,y='Hour',data=HourlyData)
df_NYaccidentcause=pd.DataFrame(accident_data_NY['VEHICLE 1 FACTOR'].value_counts()) #create a new data frame containing vehicle 1 type accident cause and the the number of accidents caused

df_NYaccidentcause.tail()

plt.figure(figsize=(12,8))

ax1 =  plt.subplot2grid((1,2),(0,0))

ax1.set_title('Top 10', size=16)

sns.barplot(x=df_NYaccidentcause.head(10).index, y='VEHICLE 1 FACTOR', data=df_NYaccidentcause.head(10))

ax1.set_xticklabels(ax1.xaxis.get_ticklabels(), rotation=90)

ax2 =  plt.subplot2grid((1,2),(0,1))

ax2.set_title('Bottom 10', size=16)

sns.barplot(x=df_NYaccidentcause.tail(10).index, y='VEHICLE 1 FACTOR', data=df_NYaccidentcause.tail(10))

ax2.set_xticklabels(ax2.xaxis.get_ticklabels(), rotation=90)

top10accidentcauses=pd.Series(df_NYaccidentcause.head(10).index)

top10accidentcauses

top10C=accident_data_NY[accident_data_NY['VEHICLE 1 FACTOR'].isin(top10accidentcauses)]

top10C.describe(include='all')

tmp=pd.DataFrame(top10C.groupby(['BOROUGH','VEHICLE 1 FACTOR']).size(), columns=['count'])

tmp.reset_index(inplace=True)

tmp=tmp.pivot(index='BOROUGH',columns='VEHICLE 1 FACTOR',values='count')

fig, axes = plt.subplots(1,1,figsize=(13,15))

tmp.plot(ax=axes,kind='bar', stacked=True)
df_NYaccidents=pd.DataFrame(accident_data_NY['VEHICLE 1 TYPE'].value_counts())

df_NYaccidents.tail()

plt.figure(figsize=(10,10))

ax1 =  plt.subplot2grid((1,2),(0,0))

ax1.set_title('Top 10', size=16)

sns.barplot(x=df_NYaccidents.head(10).index, y='VEHICLE 1 TYPE', data=df_NYaccidents.head(10))

ax1.set_xticklabels(ax1.xaxis.get_ticklabels(), rotation=90)

ax2 =  plt.subplot2grid((1,2),(0,1))

ax2.set_title('Bottom 10', size=16)

sns.barplot(x=df_NYaccidents.tail(10).index, y='VEHICLE 1 TYPE', data=df_NYaccidents.tail(10))

ax2.set_xticklabels(ax2.xaxis.get_ticklabels(), rotation=90)

top10causes=pd.Series(df_NYaccidents.head(10).index)

top10causes

top10V=accident_data_NY[accident_data_NY['VEHICLE 1 TYPE'].isin(top10causes)]

top10V.describe(include='all')

tmp=pd.DataFrame(top10V.groupby(['BOROUGH','VEHICLE 1 TYPE']).size(), columns=['count'])

tmp.reset_index(inplace=True)

tmp=tmp.pivot(index='BOROUGH',columns='VEHICLE 1 TYPE',values='count')

fig, axes = plt.subplots(1,1,figsize=(13,15))

tmp.plot(ax=axes,kind='bar', stacked=True)