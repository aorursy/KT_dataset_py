import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")

df[:5]
df1 = df.drop("Sno",axis=1)
for i in df1.columns:

    print("Name of Column :",i)

    print("Total No. of unique value:",len(df1[i].unique()))

    print(df1[i].unique(),"\n")
#df["Province/State"].value_counts()

"1/22/2020 12:00".split()[0]

df1['Date'] = df1['Last Update'].apply(lambda x: x.split()[0])

df1 = df1.drop("Last Update",axis=1)

df1[:3]
from datetime import datetime

date_format = "%m/%d/%Y"
df1["Date"] = df1["Date"].replace("1/23/20","1/23/2020")

df1["days interval"] = df1["Date"].apply(lambda x:(datetime.strptime(x, date_format).day)

                                         -(datetime.strptime(df1["Date"].min(), date_format).day))



#df1["percent of Recovered"]= (df1["Recovered"]/df1["Confirmed"])*100

#df1["percent of Deaths"]= (df1["Deaths"]/df1["Confirmed"])*100

df1[:5]
df1.info()
intreval = df1.groupby("days interval").sum().reset_index()[["days interval","Confirmed","Deaths","Recovered"]]

intreval["percent of Recovered"]= (intreval["Recovered"]/intreval["Confirmed"])*100

intreval["percent of Deaths"]= (intreval["Deaths"]/intreval["Confirmed"])*100

intreval
sns.lineplot("days interval", "Confirmed",data=intreval)
intreval.plot("days interval",["percent of Recovered","percent of Deaths"])
intreval["increasing Rate in Confirm"] = round(intreval["Confirmed"].diff(+1)/intreval["Confirmed"]*100,2)

intreval["increasing Rate in Death"] = round(intreval["Deaths"].diff(+1)/intreval["Deaths"]*100,2)

intreval["increasing Rate in Recovered"] = round(intreval["Recovered"].diff(+1)/intreval["Recovered"]*100,2)

intreval
sns.lineplot("days interval","increasing Rate in Confirm",data=intreval)
intreval.plot("days interval",["increasing Rate in Recovered","increasing Rate in Death"])
plt.figure(figsize=(15,8))

sns.countplot(df1["Country"])

plt.xticks(rotation=90)
cou = df1.groupby("Country").sum().reset_index()[["Country","Confirmed","Deaths","Recovered"]]

cou["percent of Recovered"]= (cou["Recovered"]/cou["Confirmed"])*100

cou["percent of Deaths"]= (cou["Deaths"]/cou["Confirmed"])*100

cou[:5]
cou.plot("Country",["percent of Recovered","percent of Deaths"],kind='bar',figsize=(15,8))

plt.figure(figsize=(15,8))
cou.sort_values("Deaths",ascending=False)[:3]
print("Death On Mainland China is",round((802/33986)*100,2),"%")
mc = df1[df1["Country"]=="Mainland China"]

mc[:5]
intreval = mc.groupby("days interval").sum().reset_index()[["days interval","Confirmed","Deaths","Recovered"]]

intreval["percent of Recovered"]= (intreval["Recovered"]/intreval["Confirmed"])*100

intreval["percent of Deaths"]= (intreval["Deaths"]/intreval["Confirmed"])*100

intreval
intreval.plot("days interval",["percent of Recovered","percent of Deaths"])
intreval["increasing Rate in Confirm"] = round(intreval["Confirmed"].diff(+1)/intreval["Confirmed"]*100,2)

intreval["increasing Rate in Death"] = round(intreval["Deaths"].diff(+1)/intreval["Deaths"]*100,2)

intreval["increasing Rate in Recovered"] = round(intreval["Recovered"].diff(+1)/intreval["Recovered"]*100,2)

intreval
sns.lineplot("days interval","increasing Rate in Confirm",data=intreval)
intreval.plot("days interval",["increasing Rate in Recovered","increasing Rate in Death"])
confirm = mc["Confirmed"].sum()

death = mc["Deaths"].sum()

rec = mc["Recovered"].sum()

print("Total No. of Confirmed :",confirm)

print("Total No. of Recovered :",rec)

print("Total No. of Deaths    :",death)

print("Percent of Recovered :",(round((rec/confirm)*100,2)),"%")

print("Percent of Deaths    :",(round((death/confirm)*100,2)),"%")
th = df1[df1["Country"]=="Thailand"]

confirm = th["Confirmed"].sum()

death = th["Deaths"].sum()

rec = th["Recovered"].sum()

print("Total No. of Confirmed :",confirm)

print("Total No. of Recovered :",rec)

print("Total No. of Deaths    :",death)

print("Percent of Recovered :",(round((rec/confirm)*100,2)),"%")

print("Percent of Deaths    :",(round((death/confirm)*100,2)),"%")
confirm = df1["Confirmed"].sum()

death = df1["Deaths"].sum()

rec = df1["Recovered"].sum()

print("Total No. of Confirmed :",confirm)

print("Total No. of Recovered :",rec)

print("Total No. of Deaths    :",death)

print("Percent of Recovered :",(round((rec/confirm)*100,2)),"%")

print("Percent of Deaths    :",(round((death/confirm)*100,2)),"%")