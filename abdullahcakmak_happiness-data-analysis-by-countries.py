import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sbr

import numpy as np

import os

import warnings

print(os.listdir("../input"))
d2015=pd.read_csv("../input/2015.csv")

d2016=pd.read_csv("../input/2016.csv")

d2017=pd.read_csv("../input/2017.csv")
d2015.info()
d2016.info()
d2017.info()
del d2017["Whisker.high"]
del d2017["Whisker.low"]
d2015.drop(columns="Standard Error",inplace=True,errors="ignore")
d2016.drop(columns="Lower Confidence Interval",inplace=True,errors="ignore")

d2016.drop(columns="Upper Confidence Interval",inplace=True,errors="ignore")
d2015=d2015.rename(columns={"Happiness Rank":"Rank","Happiness Score":"Score","Economy (GDP per Capita)":"Economy","Health (Life Expectancy)":"Health","Trust (Government Corruption)":"Trust","Dystopia Residual":"Dystopia_Residual"})

d2016=d2016.rename(columns={"Happiness Rank":"Rank","Happiness Score":"Score","Economy (GDP per Capita)":"Economy","Health (Life Expectancy)":"Health","Trust (Government Corruption)":"Trust","Dystopia Residual":"Dystopia_Residual"})

d2017=d2017.rename(columns={"Happiness.Rank":"Rank","Happiness.Score":"Score","Economy..GDP.per.Capita.":"Economy","Health..Life.Expectancy.":"Health","Trust..Government.Corruption.":"Trust","Dystopia.Residual":"Dystopia_Residual"})
d2015.head()
d2016.head()
d2017.head()
d2017.drop(d2017.index[154],inplace=True)

d2017.drop(d2017.index[146],inplace=True)

d2017.drop(d2017.index[138],inplace=True)

d2017.drop(d2017.index[112],inplace=True)

d2017.drop(d2017.index[110],inplace=True)

d2017.drop(d2017.index[49],inplace=True)

d2017.sort_values(by="Country").head()
d2015.drop(d2015.index[147],inplace=True)

d2015.drop(d2015.index[139],inplace=True)

d2015.drop(d2015.index[125],inplace=True)

d2015.drop(d2015.index[100],inplace=True)

d2015.drop(d2015.index[98],inplace=True)

d2015.drop(d2015.index[96],inplace=True)

d2015.drop(d2015.index[93],inplace=True)

d2015.drop(d2015.index[39],inplace=True)

d2015.drop(d2015.index[21],inplace=True)

d2015.sort_values(by="Country").head()
d2016.drop(d2016.index[142],inplace=True)

d2016.drop(d2016.index[137],inplace=True)

d2016.drop(d2016.index[112],inplace=True)

d2016.drop(d2016.index[101],inplace=True)

d2016.drop(d2016.index[75],inplace=True)

d2016.drop(d2016.index[51],inplace=True)

d2016.drop(d2016.index[39],inplace=True)

d2016.drop(d2016.index[14],inplace=True)

d2016.sort_values(by="Country").head()
d2015.info()
d2016.info()
d2017.info()
new2015=d2015.sort_values(by="Country").copy()

new2016=d2016.sort_values(by="Country").copy()

new2017=d2017.sort_values(by="Country").copy()
new2017.head()
new2015.index=range(len(new2015))

new2016.index=range(len(new2016))

new2017.index=range(len(new2017))

new2015.head(3)
uni_data=pd.DataFrame()

uni_data["Country"]=new2015.Country

uni_data["Region"]=new2015.Region

uni_data["Rank"]=new2015.Rank

uni_data["Score"]=new2015.Score+new2016.Score+new2017.Score

uni_data["Economy"]=new2015.Economy+new2016.Economy+new2017.Economy

uni_data["Family"]=new2015.Family+new2016.Family+new2017.Family

uni_data["Health"]=new2015.Health+new2016.Health+new2017.Health

uni_data["Freedom"]=new2015.Freedom+new2016.Freedom+new2017.Freedom

uni_data["Trust"]=new2015.Trust+new2016.Trust+new2017.Trust

uni_data["Generosity"]=new2015.Generosity+new2016.Generosity+new2017.Generosity

uni_data["Dystopia_Residual"]=new2015["Dystopia_Residual"]+new2016["Dystopia_Residual"]+new2017["Dystopia_Residual"]

uni_data.head(10)
uni_data.tail(5)
uni_data.info()
uni_data.describe()
uni_data.corr()
f,ax=plt.subplots(figsize=(18,18))

sbr.heatmap(uni_data.corr(),annot=True,fmt=".1f",linewidths=.2,cmap="Spectral",ax=ax,linecolor="black")

plt.show()
uni_data.sort_values(by="Economy",ascending=False).head()
uni_data.sort_values(by="Economy").head()
uni_data.sort_values(by="Score",ascending=False).head()
melted=pd.melt(frame=uni_data,id_vars="Country",value_vars=["Region"])

melted
f,ax=plt.subplots(figsize=(15,20))

sbr.boxplot(x=uni_data.Score,y=uni_data.Region,data=uni_data)

sbr.swarmplot(x=uni_data.Score,y=uni_data.Region,data=uni_data,color=".10",size=8)

warnings.filterwarnings("ignore")
uni_data.set_index("Rank").sort_values(by="Rank",ascending="False").head(10)
uni_data[["Country","Economy","Trust"]].head(20)
uni_data["Economy"].describe()
stdscore=uni_data.Score.std()

scoremean=sum(uni_data.Score)/len(uni_data.Score)

print("Score Average: ",scoremean)

print("Score Standart Deviation: ",stdscore)
stdeco=uni_data["Economy"].std()

ecomean=sum(uni_data.Economy)/len(uni_data.Economy)

print("Economy Average: " ,ecomean)

print("Economy Standard Deviation: ",stdeco)
stdhealth=uni_data["Health"].std()

healthmean=sum(uni_data.Health)/len(uni_data.Health)

print("Health Average: ",healthmean)

print("Health Standart Deviation: ",stdhealth)
stdfamily=uni_data["Family"].std()

familymean=sum(uni_data.Family)/len(uni_data.Family)

print("Family Average: ",familymean)

print("Family Standart Deviation: ",stdfamily)
datamean=pd.DataFrame()

datamean["Country"]=uni_data.Country

datamean["Region"]=uni_data.Region

datamean["Score"]=uni_data.Score

datamean["Score_Level"]=["High" if i>scoremean+stdscore else "Normal" if (scoremean-stdscore)<i<(scoremean+stdscore) else "Low" for i in uni_data.Score]

datamean["Economy"]=uni_data.Economy

datamean["Economic_Level"]=["High" if i>ecomean+stdeco else "Normal" if (ecomean-stdeco)<i<(ecomean+stdeco) else "Low" for i in uni_data.Economy]

datamean["Health"]=uni_data.Health

datamean["Health_Level"]=["High" if i>healthmean+stdhealth else "Normal" if (healthmean-stdhealth)<i<(healthmean+stdhealth) else "Low" for i in uni_data.Health]

datamean["Family"]=uni_data.Family

datamean["Family_Level"]=["High" if i>familymean+stdfamily else "Normal" if (familymean-stdfamily)<i<(familymean+stdfamily) else "Low" for i in uni_data.Family]

datamean.head(10)
datamean.tail(10)
labels=datamean.Score_Level.value_counts().index

colors=("gold","Green","Red")

explode=[0,0.1,0.15]

sizes=datamean.Score_Level.value_counts().values



plt.figure(figsize=(10,10))

plt.pie(sizes,explode=explode,colors=colors,labels=labels,shadow=True,autopct='%1.1f%%')

plt.title("Pie Chart According to Score Level",color="Black",fontsize=15)

warnings.filterwarnings("ignore")
labels=datamean.Economic_Level.value_counts().index

colors=("lightyellow","red","Yellowgreen")

explode=[0,0.1,0.15]

sizes=datamean.Economic_Level.value_counts().values



plt.figure(figsize=(10,10))

plt.pie(sizes,explode=explode,colors=colors,labels=labels,shadow=True,autopct='%1.1f%%')

plt.title("Pie Chart According to Economic Level",color="Black",fontsize=15)

warnings.filterwarnings("ignore")
labels=datamean.Health_Level.value_counts().index

colors=["lightgreen","red","yellowgreen"]

explode=[0,0.1,0.15]

sizes=datamean.Health_Level.value_counts().values



plt.figure(figsize=(10,10))

plt.pie(sizes,explode=explode,colors=colors,labels=labels,shadow=True,autopct="%1.1f%%")

plt.title("Pie Chart According to Health Level",color="Black",fontsize=15)

warnings.filterwarnings("ignore")
labels=datamean.Family_Level.value_counts().index

colors=["lightblue","red","yellowgreen"]

explode=[0,0.1,0.15]

sizes=datamean.Health_Level.value_counts().values



plt.figure(figsize=(10,10))

plt.pie(sizes,explode=explode,colors=colors,labels=labels,shadow=True,autopct="%1.1f%%")

plt.title("Pie Chart According to Family Level",color="Black",fontsize=15)

warnings.filterwarnings("ignore")
datamean.head()
f,ax=plt.subplots(figsize=(15,15))

sbr.swarmplot(x=datamean.Score_Level,y=datamean.Score,hue=datamean.Region,size=12)

warnings.filterwarnings("ignore")
f,ax=plt.subplots(figsize=(15,15))

sbr.swarmplot(x=datamean.Economic_Level,y=datamean.Economy,hue=datamean.Region,size=12)

warnings.filterwarnings("ignore")
f,ax=plt.subplots(figsize=(15,15))

sbr.swarmplot(x=datamean["Health_Level"],y=datamean.Health,hue=datamean.Region,size=12)

warnings.filterwarnings("ignore")
f,ax=plt.subplots(figsize=(15,15))

sbr.swarmplot(x=datamean["Family_Level"],y=datamean.Family,hue=datamean.Region,size=12)

warnings.filterwarnings("ignore")
sbr.jointplot(x=datamean.Health,y=datamean.Economy,data=datamean,kind="kde",space=0,color="g")

warnings.filterwarnings("ignore")
grpdata=datamean.set_index(["Score_Level","Economic_Level","Health_Level","Family_Level"])

grpdata.loc["High","High","High","High"]
grpdata.loc["Normal","Normal","Normal","Normal"]
grpdata.loc["Low","Low","Low","Low"]
f,ax=plt.subplots(figsize=(10,8))

sbr.swarmplot(x=grpdata.loc["High","High","High","High"].Score,y=grpdata.loc["High","High","High","High"].Country,size=10,linewidth=1)

warnings.filterwarnings("ignore")
f,ax=plt.subplots(figsize=(10,6))

sbr.swarmplot(x=grpdata.loc["Low","Low","Low","Low"].Score,y=grpdata.loc["Low","Low","Low","Low"].Country,size=10,linewidth=1)

warnings.filterwarnings("ignore")
filter_eco=uni_data.Economy>sum(uni_data.Economy)/len(uni_data.Economy)

filter_health=uni_data.Health>sum(uni_data.Health)/len(uni_data.Health)

filter_trust=uni_data.Trust>sum(uni_data.Trust)/len(uni_data.Trust)

filter_family=uni_data.Family>sum(uni_data.Family)/len(uni_data.Family)

uni_data[filter_eco & filter_health & filter_trust & filter_family]
f,ax=plt.subplots(figsize=(15,15))

sbr.barplot(x="Economy",y="Country",data=uni_data[filter_eco & filter_health & filter_trust & filter_family].sort_values(by="Economy",ascending=False))

warnings.filterwarnings("ignore")
filter_eco2=uni_data.Economy<sum(uni_data.Economy)/len(uni_data.Economy)

filter_health2=uni_data.Health<sum(uni_data.Health)/len(uni_data.Health)

filter_trust2=uni_data.Trust<sum(uni_data.Trust)/len(uni_data.Trust)

filter_family2=uni_data.Family<sum(uni_data.Family)/len(uni_data.Family)

uni_data[filter_eco2 & filter_health2 & filter_trust2 & filter_family2]
f,ax=plt.subplots(figsize=(15,15))

sbr.barplot(x="Economy",y="Country",data=uni_data[filter_eco2 & filter_health2 & filter_trust2 & filter_family2].sort_values(by="Economy"))

warnings.filterwarnings("ignore")
f,ax=plt.subplots(figsize=(10,10))

p1=sbr.kdeplot(uni_data[filter_eco & filter_health & filter_trust & filter_family].Economy,shade=True,color="g")

p1=sbr.kdeplot(uni_data[filter_eco2 & filter_health2 & filter_family2 & filter_trust2].Economy,shade=True,color="r")

warnings.filterwarnings("ignore")
f,ax=plt.subplots(figsize=(10,10))

p2=sbr.kdeplot(grpdata.loc["High","High","High","High"].Score,color="g",shade=True)

p2=sbr.kdeplot(grpdata.loc["Normal","Normal","Normal","Normal"].Score,color="y",shade=True)

p2=sbr.kdeplot(grpdata.loc["Low","Low","Low","Low"].Score,color="r",shade=True)

warnings.filterwarnings("ignore")
f,ax=plt.subplots(figsize=(10,10))

plt3=sbr.kdeplot(grpdata.loc["High","High","High","High"].Family,shade=True)

plt3=sbr.kdeplot(grpdata.loc["Normal","Normal","Normal","Normal"].Family,shade=True)

plt3=sbr.kdeplot(grpdata.loc["Low","Low","Low","Low"].Family,shade=True)

warnings.filterwarnings("ignore")
sbr.jointplot(x=grpdata.loc["High","High","High","High"].Health,y=grpdata.loc["High","High","High","High"].Economy,kind="kde",color="g")

sbr.jointplot(x=grpdata.loc["Normal","Normal","Normal","Normal"].Health,y=grpdata.loc["Normal","Normal","Normal","Normal"].Economy,kind="kde",color="y")

sbr.jointplot(x=grpdata.loc["Low","Low","Low","Low"].Health,y=grpdata.loc["Low","Low","Low","Low"].Economy,kind="kde",color="r")

warnings.filterwarnings("ignore")
f,ax=plt.subplots(figsize=(15,15))

sbr.swarmplot(x="Economy",y="Score_Level",hue="Country",data=datamean[filter_eco & filter_health & filter_family & filter_trust],size=15)

warnings.filterwarnings("ignore")
f,ax=plt.subplots(figsize=(17,12))

sbr.boxenplot(x="Economic_Level",y="Health",data=datamean,hue="Region")

warnings.filterwarnings("ignore")
f,ax=plt.subplots(figsize=(17,10))

sbr.boxenplot(x="Score_Level",y="Family",data=datamean,hue="Region")

warnings.filterwarnings("ignore")