import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
d=pd.read_csv("../input/air-quality-data-in-india/city_day.csv")

data=pd.DataFrame(d)

data.head()
data_copy=data.copy()

data_copy.head()
data_copy.info()
data_copy["Date"]=pd.to_datetime(data_copy["Date"])

data_copy["Year"]=data_copy["Date"].dt.year
data_copy.info()
data_copy.isnull().sum()
avg1=data_copy["PM2.5"].mean()

avg1

data_copy.fillna(value={"PM2.5":avg1},inplace=True)
avg_NO2=data_copy["NO2"].mean()

avg_NO2

data_copy.fillna(value={"NO2":avg_NO2},inplace=True)
avg_CO=data_copy["CO"].mean()

avg_CO

data_copy.fillna(value={"CO":avg_CO},inplace=True)
avg_SO2=data_copy["SO2"].mean()

avg_SO2

data_copy.fillna(value={"SO2":avg_SO2},inplace=True)
avg_O3=data_copy["O3"].mean()

avg_O3

data_copy.fillna(value={"O3":avg_O3},inplace=True)
avg_NO=data_copy["NO"].mean()

avg_NO

data_copy.fillna(value={"NO":avg_NO},inplace=True)
avg_NOx=data_copy["NOx"].mean()

avg_NOx

data_copy.fillna(value={"NOx":avg_NOx},inplace=True)
avg_AQI=data["AQI"].mean()

data_copy.fillna(value={"AQI":avg_AQI},inplace=True)
sns.set(style="darkgrid")

graph=sns.catplot(x="City",kind="count",data=data_copy,height=5,aspect=3)

graph.set_xticklabels(rotation=90)
sns.set(style="darkgrid")

graph=sns.catplot(x="City",kind="count",data=data_copy,col="AQI_Bucket",col_wrap=2,height=3.5,aspect=3)

graph.set_xticklabels(rotation=90)
graph1=sns.catplot(x="City",y="PM2.5",kind="box",data=data_copy,height=5,aspect=3)

graph1.set_xticklabels(rotation=90)
graph2=sns.catplot(x="City",y="NO2",kind="box",data=data_copy,height=5,aspect=3)

graph2.set_xticklabels(rotation=90)
graph3=sns.catplot(x="City",y="O3",data=data_copy,kind="box",height=5,aspect=3)

graph3.set_xticklabels(rotation=90)
graph4=sns.catplot(x="City",y="SO2",data=data_copy,kind="box",height=5,aspect=3)

graph4.set_xticklabels(rotation=90)
graph5=sns.catplot(data=data_copy,kind="box",x="City",y="NOx",height=6,aspect=3)

graph5.set_xticklabels(rotation=90)
graph6=sns.catplot(data=data_copy,kind="box",x="City",y="NO",height=6,aspect=3)

graph6.set_xticklabels(rotation=90)
graph7=sns.catplot(x="AQI_Bucket",data=data_copy,kind="count",height=6,aspect=3)

graph7.set_xticklabels(rotation=90)
graph8=sns.catplot(x="AQI_Bucket",kind="count",data=data_copy,col="City",col_wrap=4)

graph8.set_xticklabels(rotation=90)
plt.figure(figsize=(12,10))

pivot=pd.pivot_table(index="City",values=["PM2.5","NO2","CO","SO2","O3","NOx","NO"],data=data_copy)

sns.set(font_scale=1.2)

sns.heatmap(pivot,cmap="Reds",annot=True,cbar=False)
plt.figure(figsize=(13,12))

pivot3=data_copy[data_copy["Year"]==2020].pivot_table(index="City",values=["PM2.5","NO2","CO","SO2","O3","NOx","NO"])

sns.heatmap(pivot3,cmap="Reds",cbar=False,annot=True)
plt.figure(figsize=(13,12))

pivot4=data_copy[data_copy["Year"]==2019].pivot_table(index="City",values=["PM2.5","NO2","CO","SO2","O3","NOx","NO"])

sns.heatmap(pivot4,cmap="Reds",cbar=False,annot=True)
plt.figure(figsize=(13,12))

pivot5=data_copy[data_copy["Year"]==2018].pivot_table(index="City",values=["PM2.5","NO2","CO","SO2","O3","NOx","NO"])

sns.heatmap(pivot5,cmap="Reds",cbar=False,annot=True)
plt.figure(figsize=(13,12))

pivot6=data_copy[data_copy["Year"]==2017].pivot_table(index="City",values=["PM2.5","NO2","CO","SO2","O3","NOx","NO"])

sns.heatmap(pivot6,cmap="Reds",cbar=False,annot=True)
plt.figure(figsize=(13,12))

pivot7=data_copy[data_copy["Year"]==2016].pivot_table(index="City",values=["PM2.5","NO2","CO","SO2","O3","NOx","NO"])

sns.heatmap(pivot7,cmap="Reds",cbar=False,annot=True)
plt.figure(figsize=(13,12))

pivot8=data_copy[data_copy["Year"]==2015].pivot_table(index="City",values=["PM2.5","NO2","CO","SO2","O3","NOx","NO"])

sns.heatmap(pivot8,cmap="Reds",cbar=False,annot=True)
graph12=sns.catplot(data=data_copy[data_copy["Year"]==2020],x="City",kind="count",height=6,aspect=3)

graph12.set_xticklabels(rotation=90)
graph13=sns.catplot(data=data_copy[data_copy["Year"]==2019],x="City",kind="count",height=6,aspect=3)

graph13.set_xticklabels(rotation=90)
graph14=sns.catplot(data=data_copy[data_copy["Year"]==2018],x="City",kind="count",height=6,aspect=3)

graph14.set_xticklabels(rotation=90)
graph15=sns.catplot(data=data_copy[data_copy["Year"]==2017],x="City",kind="count",height=6,aspect=3)

graph15.set_xticklabels(rotation=90)
graph16=sns.catplot(data=data_copy[data_copy["Year"]==2016],x="City",kind="count",height=6,aspect=3)

graph16.set_xticklabels(rotation=90)
graph17=sns.catplot(data=data_copy[data_copy["Year"]==2015],x="City",kind="count",height=6,aspect=3)

graph17.set_xticklabels(rotation=90)
sns.relplot(x="PM2.5",y="AQI",kind="scatter",data=data_copy,col="City",col_wrap=4,color="red")
sns.relplot(x="NO2",y="AQI",kind="scatter",data=data_copy,col="City",col_wrap=4,color="green")
sns.relplot(x="SO2",y="AQI",kind="scatter",data=data_copy,col="City",col_wrap=4,color="blue")
sns.relplot(x="PM10",y="AQI",kind="scatter",data=data_copy,col="City",col_wrap=4,color="orange")
sns.relplot(x="NOx",y="AQI",kind="scatter",data=data_copy,col="City",col_wrap=4,color="magenta")