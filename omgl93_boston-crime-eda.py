# Library import

import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import matplotlib.dates as mdate



import seaborn as sns

import datetime



import folium #Geoploting library

from folium.plugins import HeatMap



import warnings

warnings.filterwarnings("ignore")



#Define plot style

plt.style.use("fivethirtyeight")
#Importing the dataset

df = pd.read_csv(r"../input/crimes-in-boston/crime.csv",encoding = 'unicode_escape')
#List of columns to drop

columns_to_drop = ["INCIDENT_NUMBER", #Unneccecary

                  "OFFENSE_CODE", #Better to use group

                  "OFFENSE_DESCRIPTION", #Better to use group

                  "REPORTING_AREA", #Not needed

                  "UCR_PART", #Not needed

                  "Location"] #Loc and Lat columns are already there



#Apply the column drop

df.drop(columns_to_drop, axis=1, inplace=True)



#Replace the -1 in the Lat & Long column with NaN

df["Lat"].replace(-1,np.nan,inplace=True)

df["Lat"].replace(-1,np.nan,inplace=True)



#Adding a "No" to the shooting column

df["SHOOTING"].fillna("N", inplace=True)



#Converting to datetime

df["OCCURRED_ON_DATE"] = pd.to_datetime(df["OCCURRED_ON_DATE"])



#Renaming some crimes

rename_crimes = {"INVESTIGATE PERSON": "Investigate Person",

                         "HUMAN TRAFFICKING" : "Human Trafficking",

                         "HUMAN TRAFFICKING - INVOLUNTARY SERVITUDE": "Involuntary Servitude"}

df["OFFENSE_CODE_GROUP"].replace(rename_crimes,inplace=True)
fig, ax = plt.subplots(figsize=(10,7))



#Plot

sns.catplot(x="YEAR", kind="count", palette="Blues", data=df, ax=ax)

plt.close(2)



plt.xlabel("Year", alpha=0.75)

plt.ylabel("Total number of crimes", alpha=0.75)



ax.text(x=-1.1,y=112000,s="The total number of crimes committed each year in the Boston area (2015-2018).",weight="bold",alpha=0.75, fontsize=15)
fig, ax = plt.subplots(figsize=(15,5))



#

sns.catplot(x="MONTH", hue="YEAR", kind="count", data=df, ax=ax)

plt.close(2)

plt.xlabel("Month", alpha=0.75)

plt.ylabel("Total number of crimes", alpha=0.75)



ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],alpha=0.75)



#Title

ax.text(x=-1.1,y=10000,s="The total number of crimes committed each month in the Boston area (2015-2018).",weight="bold",alpha=0.75, fontsize=15)

crimes_per_month = df.set_index("OCCURRED_ON_DATE") #Set datetime as index

crimes_m = pd.DataFrame(crimes_per_month.resample("M").agg(dict(OFFENSE_CODE_GROUP="size"))) #Resample on a monthly basis

crimes_m["mean"] = crimes_m["OFFENSE_CODE_GROUP"].mean() #Calculate the mean for the data



# Remove two incomplete dates

incomplete_dates = ["2018-09-30", "2015-06-30"]

crimes_m = crimes_m.drop(pd.to_datetime(incomplete_dates))



fig, ax = plt.subplots(figsize=(14, 7))





plt.plot(crimes_m["OFFENSE_CODE_GROUP"], label="Crimes per month", color="black")

plt.plot(crimes_m["mean"], label="Mean montly crime", color="r", linewidth=4)



#Highs Lows

high = crimes_m[crimes_m.index >= "2017-04-30"]

high = high[high.index <= "2017-11-30"]

low = crimes_m[crimes_m.index >= "2017-11-30"]

low = low[low.index <= "2018-04-30"]



plt.fill_between( high.index, high["OFFENSE_CODE_GROUP"], color="red", alpha=0.75)

plt.fill_between( low.index, low["OFFENSE_CODE_GROUP"], color="green", alpha=0.75)



ax.annotate("6937 Crimes", xy=('2018-02-12', 7000),  xycoords='data',size=15,

            bbox=dict(boxstyle="round", fc="none", ec="gray"),

            xytext=(0, -30), textcoords='offset points', ha='center')

ax.annotate("9026 Crimes", xy=("2017-08-31", 9200),  xycoords='data',size=15,

            bbox=dict(boxstyle="round", fc="none", ec="gray"),

            xytext=(0, 10), textcoords='offset points', ha='center')



plt.xlabel("Date", alpha=0.75)

plt.ylabel("Number of crimes", alpha=0.75)

plt.ylim(6000,10000)



#Title

ax.text(x="2015-03-15",y=10250,s="The total number of crimes committed each month in the Boston area (2015-2018).",weight="bold",alpha=0.75, fontsize=16)



del high, low, crimes_m
#The most crimes happened in a 2017 month so we are going to find temperature data from it

date = ["31.1.2017","28.2.2017","31.3.2017","30.4.2017","31.5.2017","30.6.2017","31.7.2017","31.8.2017","30.9.2017","31.10.2017","30.11.2017"

        ,"31.12.2017"]

avrg_temp = [35.15,36.65,34.05,51.65,56.35,69.5,72.5,72,67.1,61.35,43.75,30.75]

temperature_data = pd.DataFrame(date,columns=["Date"])

temperature_data["Date"] = pd.to_datetime(temperature_data["Date"])

temperature_data["F Temperate"] = avrg_temp
df_month_2017 = df[df["YEAR"]==2017]

df_month_2017 = df_month_2017.set_index(df_month_2017["OCCURRED_ON_DATE"]).resample("M").agg({"OFFENSE_CODE_GROUP" : "size"})



fig, ax1 = plt.subplots(figsize=(10,5))



#Plot



ax2 = ax1.twinx()



ax1.plot(df_month_2017.index,df_month_2017["OFFENSE_CODE_GROUP"],label="Crime",color="black")

ax2.plot(temperature_data["Date"],temperature_data["F Temperate"],label="Temperature",color="r")



ax2.legend()

ax1.legend(loc="upper left")



plt.xlabel("Date", alpha=0.75)

ax1.set_ylabel("Crimes per month", alpha=0.75)

ax2.set_ylabel("Temperature", alpha=0.75)



#Title



ax1.text(x="2016-12-15",y=9650,s="Correlation between crime and temperature (2017)",weight="bold",alpha=0.75, fontsize=17)

ax1.text(x="2016-12-15",y=9500,s="High correlation pattern between the number of crimes and hight of the temperature in the Boston area.",alpha=0.75, fontsize=15)
#Data

day_order = ["Monday", "Tuesday", "Wednesday","Thursday","Friday","Saturday","Sunday"]



fig,(ax1,ax2) = plt.subplots(2,1, figsize=[15,10])



sns.countplot("DAY_OF_WEEK", data=df,order=day_order,palette="RdBu_r",edgecolor="black", ax = ax1)

ax1.set_xlabel("Days of the week", alpha=0.75)

ax1.set_ylabel("Number of crimes", alpha=0.75)



sns.countplot("HOUR", data=df,palette="RdBu_r",edgecolor="black", ax = ax2)

ax2.set_xlabel("Hours in a day", alpha=0.75)

ax2.set_ylabel("Number of crimes", alpha=0.75)



ax1.text(x=-0.75,y=55000,s="Daily and hourly number of crimes in the Boston area.",weight="bold",alpha=0.75, fontsize=20)
crime_per_day = df.set_index("OCCURRED_ON_DATE")
crimes = pd.DataFrame(crime_per_day.resample("D").size())

crimes["mean"] = crime_per_day.resample("D").size().mean()

crimes["std"] = crime_per_day.resample("D").size().std()

UCL = crimes["mean"] + 3 * crimes["std"]

LCL = crimes["mean"] - 3 * crimes["std"]
fig, ax = plt.subplots(figsize=(17, 10))





ax.plot(crimes[[0]],color="red", alpha=0.75, label="Crimes per day")

ax.plot(UCL, color="black",linewidth=4,linestyle="dashed",alpha=0.75,label="Upper control limit")

ax.plot(LCL, color="black",linewidth=4,linestyle="dashed",alpha=0.75,label="Lower control limit")

ax.plot(crimes["mean"], color="black",linewidth=4,alpha=0.75, label="Mean Crime")

ax.tick_params(labelsize=12)



ax.set_xlabel("Date", alpha=0.75)

ax.set_ylabel("Number of crimes", alpha=0.75)



plt.legend(loc="lower right")

ax.text(x="2015-03-01",y=410,s="Number of crimes per day in the Boston area.",weight="bold",alpha=0.75, fontsize=30)

ax.text(x="2015-03-01",y=400,s="The points beyond the upper and lower control limits signal that the data points are uncommon and need to look at further.",alpha=0.75, fontsize=17)
crime_per_day_2017 = crime_per_day[crime_per_day["YEAR"] == 2017]

crimes_2017 = pd.DataFrame(crime_per_day_2017.resample("D").size())
fig, ax = plt.subplots(figsize=(17, 10))



#Plot

ax.plot(crimes_2017[[0]], label="Crimes per day")



#Annotations

ax.annotate("Independence Day", xy=('2017-7-4', 250),  xycoords='data',size=15,

            bbox=dict(boxstyle="round", fc="none", ec="gray"),

            xytext=(10, -40), textcoords='offset points', ha='center',

            arrowprops=dict(arrowstyle="->",color="black"))

ax.annotate("New Year's Day", xy=('2017-1-1', 220),  xycoords='data',size=15,

            bbox=dict(boxstyle="round", fc="none", ec="gray"),

            xytext=(10, -40), textcoords='offset points', ha='left',

            arrowprops=dict(arrowstyle="->",color="black"))

ax.annotate("Labor Day", xy=('2017-9-4', 220),  xycoords='data',size=15,

            bbox=dict(boxstyle="round", fc="none", ec="gray"),

            xytext=(10, -40), textcoords='offset points', ha='left',

            arrowprops=dict(arrowstyle="->",color="black"))

ax.annotate("Halloween", xy=('2017-10-31', 270),  xycoords='data',size=15,

            bbox=dict(boxstyle="round", fc="none", ec="gray"),

            xytext=(-80, -50), textcoords='offset points', ha='left',

            arrowprops=dict(arrowstyle="->",color="black"))

ax.annotate("Thanksgiving", xy=('2017-11-25', 150),  xycoords='data',size=15,

            bbox=dict(boxstyle="round", fc="none", ec="gray"),

            xytext=(-80, -40), textcoords='offset points', ha='left',

            arrowprops=dict(arrowstyle="->",color="black"))

ax.annotate("Christmas", xy=('2017-12-25', 150),  xycoords='data',size=15,

            bbox=dict(boxstyle="round", fc="none", ec="gray"),

            xytext=(10, -40), textcoords='offset points', ha='left',

            arrowprops=dict(arrowstyle="->",color="black"))



#Plot elements

ax.set_xlabel("Date", alpha=0.75)

ax.set_ylabel("Crime per day", alpha=0.75)



#Title

ax.text(x="2016-12-10",y=380,s="Number of crimes per day in the Boston area.",weight="bold",alpha=0.75, fontsize=22)

ax.text(x="2016-12-10",y=370,s="Some spikes and valleys can be explained by certain holidays that occurred.",alpha=0.75, fontsize=17)
ax,fig = plt.subplots(figsize=(15,7))



#Plot

sns.countplot(y = df["OFFENSE_CODE_GROUP"],order=df["OFFENSE_CODE_GROUP"].value_counts()[:10].index, palette="RdBu_r",edgecolor="black")



plt.ylabel("Crime Type", fontsize=22, alpha=.75)

plt.xlabel("Number of crimes committed", fontsize=22, alpha=.75)



plt.yticks(alpha=0.75,weight="bold")

plt.xticks(alpha=0.75)



#Title

ax.text(x=-0.10,y=0.95,s="Most frequent crimes in the Boston area (2015-2017).",weight="bold",alpha=0.75, fontsize=22)
ax,fig = plt.subplots(figsize=(15,7))



#Plot

sns.countplot(y = df["OFFENSE_CODE_GROUP"],order=df["OFFENSE_CODE_GROUP"].value_counts()[-10:].index, palette="RdBu_r",edgecolor="black")



plt.ylabel("Crime Type", fontsize=22, alpha=.75)

plt.xlabel("Number of crimes committed", fontsize=22, alpha=.75)



plt.yticks(alpha=0.75,weight="bold")

plt.xticks(alpha=0.75)



#Title

ax.text(x=-0.05,y=0.95,s="Least frequent crimes in the Boston area (2015-2017).",weight="bold",alpha=0.75, fontsize=22)
christmas_crime = df[df["OCCURRED_ON_DATE"] >= "2017-12-25"]

christmas_crime = christmas_crime[christmas_crime["OCCURRED_ON_DATE"] < "2017-12-26"]
ax,fig = plt.subplots(figsize=(15,7))



#Plot

sns.countplot(y = christmas_crime["OFFENSE_CODE_GROUP"],order=christmas_crime["OFFENSE_CODE_GROUP"].value_counts()[:10].index, palette="RdBu_r",edgecolor="black")



plt.ylabel("Crime Type", fontsize=22, alpha=.75)

plt.xlabel("Number of crimes committed", fontsize=22, alpha=.75)



plt.yticks(alpha=0.75,weight="bold")

plt.xticks(alpha=0.75)



#Title

ax.text(x=-0.05,y=0.95,s="Most frequent crimes on Christmas in the Boston area (2017).",weight="bold",alpha=0.75, fontsize=22)
christmas_crime_map = folium.Map(location=[42.3601, -71.0589],

              zoom_start = 12)

christmas_crime_loc = christmas_crime[["Lat", "Long"]]

christmas_crime_loc.dropna(inplace=True)

HeatMap(christmas_crime_loc, radius = 20).add_to(christmas_crime_map)

display(christmas_crime_map)
#Data

thank_crimes = df[df["OCCURRED_ON_DATE"] >= "2017-11-23"]

thank_crimes = thank_crimes[thank_crimes["OCCURRED_ON_DATE"] < "2017-11-24"]



ax,fig = plt.subplots(figsize=(15,7))



#Plot

sns.countplot(y = thank_crimes["OFFENSE_CODE_GROUP"],order=thank_crimes["OFFENSE_CODE_GROUP"].value_counts()[:10].index, palette="RdBu_r",edgecolor="black")



plt.ylabel("Crime Type", fontsize=22, alpha=.75)

plt.xlabel("Number of crimes committed", fontsize=22, alpha=.75)



plt.yticks(alpha=0.75,weight="bold")

plt.xticks(alpha=0.75)



#Title

ax.text(x=-0.05,y=0.95,s="Most frequent crimes on Thanksgiving Day in the Boston area (2017).",weight="bold",alpha=0.75, fontsize=22)

thank_crime_map = folium.Map(location=[42.3601, -71.0589],

              zoom_start = 12)

thank_crime_loc = thank_crimes[["Lat", "Long"]]

thank_crime_loc.dropna(inplace=True)

HeatMap(thank_crime_loc, radius = 20).add_to(thank_crime_map)

display(thank_crime_map)
gun_crime_df = crime_per_day[crime_per_day["SHOOTING"] == "Y"]

gun_crime_per_day = pd.DataFrame(gun_crime_df.resample("M").size())

gun_crime_per_day["mean"] = gun_crime_df.resample("M").size().mean()
#Ploting with an window = 5 or SMA = 5 rolling mean.

fig, ax = plt.subplots(figsize=(15, 7))



ax.plot(gun_crime_per_day[[0]], label = "Gun crime per day")

ax.plot(gun_crime_per_day[[0]].rolling(window=5).mean(), color = "red", label="Gun crime rolling mean")



plt.ylabel("Number of gun crimes", fontsize=17, alpha=0.75)

plt.xlabel("Date", fontsize=17,alpha=0.75)



plt.xticks(fontsize=14,alpha=0.75,weight="bold")

plt.yticks(fontsize=14,alpha=0.75,weight="bold")



ax.text(x="2018-08-01",y=27,s="Rolling 5-day mean",weight="bold",color="red",alpha=0.75, fontsize=18)

ax.text(x="2018-08-01",y=19,s="Daily mean",weight="bold",color="blue",alpha=0.75, fontsize=18)



#Title

ax.text(x="2015-05-01",y=50,s="Daily gun crime in the Boston area(2015-2018).",weight="bold",color="black",alpha=0.75, fontsize=22)
boston_shooting_map = folium.Map(location=[42.3601, -71.0589],

              zoom_start = 12)

shooting_location = gun_crime_df[["Lat", "Long"]]

shooting_location.dropna(inplace=True)
HeatMap(shooting_location, radius = 20).add_to(boston_shooting_map)

display(boston_shooting_map)
crimes_with_guns = gun_crime_df["OFFENSE_CODE_GROUP"].value_counts()[:10]

top_gun_crimes = gun_crime_df[gun_crime_df["OFFENSE_CODE_GROUP"].isin(crimes_with_guns.index)]
ax,fig = plt.subplots(figsize=(15,7))



#Plot

sns.countplot(y = top_gun_crimes["OFFENSE_CODE_GROUP"], order=crimes_with_guns.index, data=top_gun_crimes, palette="RdBu_r",edgecolor="black")



plt.ylabel("Crime Type", fontsize=22, alpha=.75)

plt.xlabel("Number of crimes committed", fontsize=22, alpha=.75)



plt.yticks(alpha=0.75,weight="bold")

plt.xticks(alpha=0.75)



#Title

ax.text(x=-0.05,y=0.95,s="Most frequent crimes that involve gun violence (2015-2017).",weight="bold",alpha=0.75, fontsize=22)
ten_freq_crimes = df["OFFENSE_CODE_GROUP"].value_counts()[:12]

df_top_crimes = df[df["OFFENSE_CODE_GROUP"].isin(ten_freq_crimes.index)]
df_tp = df_top_crimes.pivot_table(index=df_top_crimes["OCCURRED_ON_DATE"],

                                                      columns=["OFFENSE_CODE_GROUP"],aggfunc="size", fill_value=0).resample("M").sum()
#palette = plt.get_cmap('Set2')

num=0

ax,fix = plt.subplots(figsize=(15,7))

for column in df_tp:

    num+=1

    plt.subplot(3,4, num)

    for v in df_tp:

        plt.plot(df_tp.index,v,data=df_tp,marker='', color='grey', linewidth=0.9, alpha=0.3)

        plt.tick_params(labelbottom=False)

        plt.plot(df_tp.index,column, data=df_tp,color="red", linewidth=2.4, alpha=0.75, label=column)

        plt.title(column, loc='left', fontsize=12, fontweight=0, color="black", alpha=0.75,weight="bold")

        #plt.suptitle("Most frequent crimes from 2015 - 2018", fontsize=20, fontweight=0, color='black', style='italic', y=1.02)

ax.text(x=0.05,y=0.95,s="Timeline of the most frequent crimes(2015-2018).",weight="bold",alpha=0.75, fontsize=22)