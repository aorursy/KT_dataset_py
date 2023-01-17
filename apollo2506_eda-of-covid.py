import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
country_wise_latest = pd.read_csv("../input/corona-virus-report/country_wise_latest.csv")
country_wise_latest = country_wise_latest.set_index("Country/Region")
country_wise_latest.head()
country_wise_latest.info()
ser = country_wise_latest[["Confirmed","Deaths","Recovered","Active"]].groupby(country_wise_latest["WHO Region"]).sum()
ser.plot(kind="bar")

plt.xlabel("WHO Region")
plt.ylabel("Number of cases in the magnitude of 10^6");
ser1 = country_wise_latest[["Confirmed","Deaths","Recovered","Active"]].groupby(country_wise_latest["WHO Region"]).sum()

fig, ax = plt.subplots(2,2,figsize=(15,15))

ax[0,0].pie(ser1["Confirmed"],autopct="%1.1f%%")
ax[0,0].legend(ser1.index,loc="upper right")
ax[0,0].set_title("Confirmed Cases Per Region")

ax[0,1].pie(ser1["Deaths"],autopct="%1.1f%%")
ax[0,1].legend(ser1.index,loc="upper right")
ax[0,1].set_title("Deaths per Region")

ax[1,0].pie(ser1["Recovered"],autopct="%1.1f%%")
ax[1,0].legend(ser1.index,loc="upper right")
ax[1,0].set_title("Cases Recovered per Region")

ax[1,1].pie(ser1["Active"],autopct="%1.1f%%")
ax[1,1].legend(ser1.index,loc="upper right")
ax[1,1].set_title("Cases still Active per Region");
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df_country_wise_latest = country_wise_latest[["Confirmed","Confirmed last week","1 week change","1 week % increase"]]
df_country_wise_latest = scaler.fit_transform(df_country_wise_latest)

df_country_wise_latest = pd.DataFrame(df_country_wise_latest)
df_country_wise_latest["WHO Region"] = country_wise_latest["WHO Region"].values
df_country_wise_latest.columns = ["Confirmed","Confirmed last week","1 week change","1 week % increase","WHO Region"]
df_country_wise_latest[["Confirmed","Confirmed last week","1 week % increase"]].groupby(df_country_wise_latest["WHO Region"]).mean().plot(kind='bar',legend=True);
day_wise = pd.read_csv("../input/corona-virus-report/day_wise.csv")
day_wise.set_index("Date",drop=False,inplace=True)
day_wise["Date"] = pd.to_datetime(day_wise["Date"])
day_wise.head()
day_wise.info()
day_wise[["Date","Confirmed","Active","Recovered","Deaths"]].groupby(pd.Grouper(key='Date',freq='M')).sum().plot(marker="*",legend=True)
plt.title("Cases per Month\n")
plt.xlabel("Month")
plt.ylabel("Number of cases in all countries");
day_wise[["Date","No. of countries"]].groupby(pd.Grouper(key="Date",freq='W')).mean().plot(marker="*",legend=True)


plt.title("Number of countries in which Covid was confirmed \n")
plt.xlabel("Week")
plt.ylabel("Number of Countries");
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df = day_wise[["Confirmed","Deaths","Recovered","New cases","New deaths","New recovered"]]
df = scaler.fit_transform(df)

df = pd.DataFrame(df)
df["Date"]=day_wise["Date"].values
df.columns = ["Confirmed","Deaths","Recovered","New cases","New deaths","New recovered","Date"]

df.groupby(pd.Grouper(key="Date",freq='M')).sum().plot(kind="bar",legend=True)

plt.title("Cases and Numerb of New Cases recorded per Month")
plt.xlabel("Month")
plt.ylabel("Numerb of Cases and New Cases");
full_grouped = pd.read_csv("../input/corona-virus-report/full_grouped.csv")
full_grouped["Date"] = pd.to_datetime(full_grouped["Date"])
# full_grouped.set_index(["Date","WHO Region"],drop=False,inplace=True)
full_grouped.head()
full_grouped.info()
fig,ax = plt.subplots(6,1,figsize=(6,25))
for i,region in enumerate(full_grouped["WHO Region"].unique()):
    ax[i].plot(full_grouped[["Date","Confirmed","Deaths","Recovered","Active"]][full_grouped["WHO Region"]==region].groupby(pd.Grouper(key="Date",freq='W')).sum(),'*-')
    ax[i].legend(["Confirmed","Deaths","Recovered","Active"],loc='upper left')
    ax[i].set_title(region)
    fig.tight_layout(pad=3.0);
    pass