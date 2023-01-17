import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import folium

import os
df_carona_in_india=pd.read_csv("https://api.covid19india.org/csv/latest/state_wise.csv",parse_dates=['Last_Updated_Time'])

df_carona_in_india=df_carona_in_india.drop(0)

df_carona_in_india.Last_Updated_Time=df_carona_in_india.Last_Updated_Time.dt.date

df_carona_in_india.head()

df_carona_in_india.info()
df_carona_in_india.isnull().sum()
total_confirmed=df_carona_in_india.Confirmed.sum()

total_recovered=df_carona_in_india.Recovered.sum()

total_Active=df_carona_in_india.Active.sum()

total_deaths=df_carona_in_india.Deaths.sum()

print("Total number of Confirmed cases : ",total_confirmed)

print("Total number of Recovered cases : ",total_recovered)

print("Total number of Active cases : ",total_Active)

print("Total number of Deaths : ",total_deaths)
sns.pairplot(df_carona_in_india)
sns.relplot(x='Confirmed',y='Recovered',data=df_carona_in_india)
Recovered_pct=(df_carona_in_india.Recovered.sum()/df_carona_in_india.Confirmed.sum())*100

Active_pct=(df_carona_in_india.Active.sum()/df_carona_in_india.Confirmed.sum())*100

print("Recovered people Percentage : ",Recovered_pct," %")

print("Active People Percentage : ",Active_pct)
fig=plt.figure(figsize=(16,9),)

plt.pie([Recovered_pct,Active_pct],labels=["Recovered","Active"],shadow=True,autopct ="%0.2f%%",explode=[0.06,0.0])

plt.title("Active vs Recovered",fontsize=22)

plt.show()
india_state=df_carona_in_india[["State","Confirmed"]]

india_state.head()
fig,ax=plt.subplots(figsize=(18,10))

sns.heatmap(df_carona_in_india.corr(),annot=True,annot_kws={'size':12})

plt.show()
fig=plt.figure(figsize=(16,8))

Confirmed=df_carona_in_india.Confirmed.sum()

plt.pie([df_carona_in_india.Confirmed.sum(),df_carona_in_india.Active.sum()],explode=[0.07,0.0],colors=["r","g"],autopct="%0.2f%%",shadow=True,labels=["Confirmed","Active"])

plt.title("Confirmed vs Active",fontsize=22)

plt.show()
fig=plt.figure(figsize=(16,8))

Confirmed=df_carona_in_india.Confirmed.sum()

plt.pie([df_carona_in_india.Confirmed.sum(),df_carona_in_india.Deaths.sum()],explode=[0.5,0],colors=["r","g"],autopct="%0.2f%%",shadow=True,labels=["Confirmed","Active"])

plt.title("Confirmed vs Deaths",fontsize=22)

plt.show()
fig=plt.figure(figsize=(16,8))

Confirmed=df_carona_in_india.Confirmed.sum()

plt.pie([df_carona_in_india.Active.sum(),df_carona_in_india.Deaths.sum()],colors=["r","g"],autopct="%0.2f%%",shadow=True,labels=["Confirmed","Active"],explode=[0.5,0.1])

plt.title("Active vs Deaths",fontsize=22)

plt.show()
plt.figure(figsize=(16,9))

sns.barplot('State',"Active",data=df_carona_in_india,)

plt.xticks(rotation=90)

plt.xlabel('State',fontsize=18)

plt.ylabel('Active',fontsize=18)

plt.title("State with Active cases",fontsize=25)

plt.show()
plt.figure(figsize=(16,9))

sns.barplot('State',"Deaths",data=df_carona_in_india,)

plt.xticks(rotation=90)

plt.xlabel('State',fontsize=18)

plt.ylabel('Deaths',fontsize=18)

plt.title("State with Deaths cases",fontsize=25)

plt.show()
plt.figure(figsize=(16,9))

sns.barplot('State',"Confirmed",data=df_carona_in_india,)

plt.xticks(rotation=90)

plt.xlabel('State',fontsize=18)

plt.ylabel('Confirmed',fontsize=18)

plt.title("State with Confirm cases",fontsize=25)

plt.show()