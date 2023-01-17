import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import math  

import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings("ignore")
data=pd.read_csv("../input/los-angeles-international-airport-air-cargo-volume.csv")

data.info()
data.head()
data.dtypes
data.describe()
data.plot(kind="hist",y="AirCargoTons",figsize=(15,5),bins=30,normed=True)

plt.xlabel("Tons")

plt.ylabel('Probability')

plt.plot()
data=data.drop(["DataExtractDate"],axis=1)
data.ReportPeriod=pd.to_datetime(data.ReportPeriod)

data["Arrival_Departure"]=data["Arrival_Departure"].astype("category")

data["Domestic_International"]=data["Domestic_International"].astype("category")

data["CargoType"]=data["CargoType"].astype("category")
data.dtypes
print(data.index.name)
data1=data.set_index("ReportPeriod")
data1.head()
data1.resample("A").mean()

plt.figure(figsize=(10,5))

plt.plot(data1.resample("A").mean(),color="b",label="Average Air Cargo Tons")

plt.grid()

plt.legend()

plt.xlabel("Year")

plt.ylabel("Ton")

plt.title("Average Air Cargo Tons According to Years")
month= data1.resample("M").mean()

print(month.min())

print(month.idxmin())
print(month.max())

print(month.idxmax())
plt.figure(figsize=(15,8))

plt.plot(data1.resample("M").mean(),color="g",label="Average Air Cargo Tons",linewidth=2)

plt.grid()

plt.legend()

plt.xlabel("Year")

plt.ylabel("Ton")

plt.title("Average Air Cargo Tons According To Months")

plt.show()
data2=data.set_index(["ReportPeriod","Arrival_Departure","Domestic_International","CargoType"])
data2.head(20)
cargotype=pd.concat([data1.groupby("CargoType")["AirCargoTons"].idxmin(),data1.groupby("CargoType")["AirCargoTons"].min(),data1.groupby("CargoType")["AirCargoTons"].idxmax(),data1.groupby("CargoType")["AirCargoTons"].max()],axis=1)

cargotype.columns.values[0:4]="Minimum Weight Date","Minimum Weight","Maximum Weight Date","Maximum Weight"

cargotype
freight=data1[data1.CargoType=="Freight"]

mail=data1[data1.CargoType=="Mail"]
freight.describe()
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(15,5))

freight.plot(kind="hist",bins=30,color="tomato",normed=True,ax=axes[0],title="Freight Cargo Probability")

freight.plot(kind="hist",bins=30,color="tomato",normed=True,ax=axes[1],cumulative=True,title="Freight Cargo Cumulative Probability",label="Tons")

plt.show()
plt.figure(figsize=(15,5))

plt.plot(freight.resample("A").mean(),color="b",label="Average Air Cargo Tons")

plt.grid()

plt.legend()

plt.xlabel("Year")

plt.ylabel("Ton")

plt.title("Average Freight Type Air Cargo Weights According To Years")

plt.show()
plt.figure(figsize=(15,8))

plt.plot(freight.resample("M").mean(),color="r",label="Average Air Cargo Tons")

plt.grid()

plt.legend()

plt.xlabel("Year")

plt.ylabel("Ton")

plt.title("Average Freight Type Air Cargo Weights According To Months")

plt.show()
month_freight= freight.resample("M").mean()

print(month_freight.max())

print(month_freight.idxmax())
print(month_freight.min())

print(month_freight.idxmin())
freight.groupby(["Domestic_International"]).describe()
print("maximum=",freight.groupby(["Domestic_International"]).idxmax())

print("minimum=",freight.groupby(["Domestic_International"]).idxmin())
di=pd.DataFrame(dict(list(freight.groupby("Domestic_International")['AirCargoTons'])))

di.head()
di.resample("M").mean().plot(figsize=(15,8),title="Freight Cargo Tons According to Domestic or International",grid=True,linestyle="-.",marker="8",markersize=3,color=["rebeccapurple","crimson"])

plt.ylabel("Tons")

plt.show()
di.plot(kind="scatter",x="Domestic",y="International",figsize=(10,8),color="red",alpha=0.5,grid=True,title="Scatter Diagram of International and Domestic Freight Cargo",marker="*",s=120)

plt.show()
di.corr()
freight.groupby("Arrival_Departure").describe()
print("maximum",freight.groupby("Arrival_Departure").idxmax())

print("minimum",freight.groupby("Arrival_Departure").idxmin())
ad=pd.DataFrame(dict(list(freight.groupby('Arrival_Departure')['AirCargoTons'])))

ad.head()
ad.resample("M").mean().plot(figsize=(15,8),title="Freight Cargo Weights According to Arrival or Departure",grid=True,marker='o', linestyle='dashed',markersize=5)

plt.ylabel("Tons")
ad.plot(kind="scatter",x="Arrival",y="Departure",figsize=(10,8),color="maroon",alpha=0.5,grid=True,title="Scatter Diagram of Arrival and Departure Freight Cargo",marker="d",s=120)

plt.show()
ad.corr()
freight.drop(["CargoType"],axis=1,inplace=True)
data3=pd.DataFrame(dict(list(freight.groupby(['Arrival_Departure', 'Domestic_International'])["AirCargoTons"])))

data3.head()
table=pd.concat([data3.idxmax(),data3.max(),data3.idxmin(),data3.min()],axis=1)

table.columns=table.columns.astype("str")

table.rename(columns={"0":"Maximum Date","1":"Maximum Weight","2":"Minimum Date","3":"Minimum Weight"},inplace=True)

table
fig,ax=plt.subplots(nrows=2,ncols=1, figsize=(15,15))

data3.Arrival.resample("M").mean().plot(ax=ax[0],title="Arrival Freight Cargo",color=["purple","orange"],linewidth=3,grid=True)

plt.ylabel("Tons")

data3.Departure.resample("M").mean().plot(ax=ax[1],title="Departure Freight Cargo",color=["deeppink","darkcyan"],linewidth=3,grid=True)

plt.ylabel("Tons")

plt.show()
fig,ax=plt.subplots(nrows=2,ncols=1, figsize=(15,15))

data3.Arrival.plot(kind="scatter",x="Domestic",y="International",ax=ax[0],title="Arrival Freight Cargo",color="purple",marker='P', s=80,grid=True)

data3.Departure.plot(kind="scatter",x="Domestic",y="International",ax=ax[1],title="Departure Freight Cargo",color="deeppink",marker='P',s=80,grid=True)

plt.show()

data3.corr()
mail.describe()
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(15,5))

mail.plot(kind="hist",bins=30,color="yellowgreen",normed=True,ax=axes[0],title="Mail Cargo Probability")

mail.plot(kind="hist",bins=30,color="yellowgreen",normed=True,ax=axes[1],cumulative=True,title="Mail Cargo Cumulative Probability",label="Tons")

plt.xlabel("Tons")

plt.show()
mail.resample("A").mean().plot(color="g",figsize=(15,5),title="Average Mail Type Air Cargo Weights According To Years",grid=True,linewidth=3)

plt.xlabel("Year")

plt.ylabel("Ton")

plt.show()
mail.resample("M").mean().plot(color="r",figsize=(18,8),title="Average Mail Type Air Cargo Weights According To Months",grid=True,linewidth=3)

plt.xlabel("Year")

plt.ylabel("Ton")

plt.show()
month_mail= mail.resample("M").mean()

print(month_mail.min())

print(month_mail.idxmin())
print(month_mail.max())

print(month_mail.idxmin())
mail.groupby(["Domestic_International"]).describe()
print("maximum=",mail.groupby(["Domestic_International"]).idxmax())

print("minimum=",mail.groupby(["Domestic_International"]).idxmin())
di_mail=pd.DataFrame(dict(list(mail.groupby("Domestic_International")['AirCargoTons'])))

di_mail.head()
di_mail.resample("M").mean().plot(figsize=(18,10),title="Mail Cargo Weights According to Domestic or International",linewidth=3,linestyle=":",marker="D",markersize=5,grid=True,color=["firebrick","steelblue"])

plt.ylabel("Tons")

plt.show()
di_mail.plot(kind="scatter",x="Domestic",y="International",figsize=(10,8),color="dodgerblue",alpha=0.5,grid=True,title="Scatter Diagram of International and Domestic Mail Cargo",marker="^",s=80)

plt.show()
di_mail.corr()
mail.groupby("Arrival_Departure").describe()
print("maximum",mail.groupby("Arrival_Departure").idxmax())

print("minimum",mail.groupby("Arrival_Departure").idxmin())
ad_mail=pd.DataFrame(dict(list(mail.groupby('Arrival_Departure')['AirCargoTons'])))

ad_mail.head()
ad_mail.resample("M").mean().plot(figsize=(15,8),title="Mail Cargo Weights According to Arrival or Departure",grid=True,marker='p',linestyle="--",markersize=6,colors=["coral","olivedrab"])

plt.ylabel("Tons")

plt.show()
ad_mail.plot(kind="scatter",x="Arrival",y="Departure",figsize=(10,8),color="peru",alpha=0.5,grid=True,title="Scatter Diagram of Arrival and Departure Mail Cargo",marker="s",s=80)

plt.show()
ad_mail.corr()
mail.drop(["CargoType"],axis=1,inplace=True)
data4=pd.DataFrame(dict(list(mail.groupby(['Arrival_Departure', 'Domestic_International'])["AirCargoTons"])))

data4.head()
table_mail=pd.concat([data4.idxmax(),data4.max(),data4.idxmin(),data4.min()],axis=1)

table_mail.columns=table_mail.columns.astype("str")

table_mail.rename(columns={"0":"Maximum Date","1":"Maximum Weight","2":"Minimum Date","3":"Minimum Weight"},inplace=True)

table_mail
fig,ax=plt.subplots(nrows=2,ncols=1, figsize=(18,18))

data4.Arrival.resample("M").mean().plot(ax=ax[0],title="Arrival Mail Cargo",color=["cornflowerblue","navy"], linewidth=3,grid=True)

plt.ylabel("Tons")

data4.Departure.resample("M").mean().plot(ax=ax[1],title="Departure Mail Cargo",color=["hotpink","purple"],linewidth=3,grid=True)

plt.ylabel("Tons")
fig,ax=plt.subplots(nrows=2,ncols=1, figsize=(15,15))

data4.Arrival.plot(kind="scatter",x="Domestic",y="International",ax=ax[0],title="Arrival Mail Cargo",color="mediumvioletred",marker=7,alpha=0.5, s=120,grid=True)

data4.Departure.plot(kind="scatter",x="Domestic",y="International",ax=ax[1],title="Departure Mail Cargo",color="darkslategrey",alpha=0.5,marker=10,s=120,grid=True)

plt.show()
data4.corr()