# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import folium

import plotly.express as ply

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_dt = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")

test_dt = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv")

submission_dt = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/submission.csv")

print("Number of rows in the training dataset:",len(train_dt))

print("Number of rows in the test dataset:",len(test_dt))

print("Test and Training Dataset Ratio:",round(len(test_dt)*100/len(train_dt),2),"%")

train_dt.head(5)

print("No of confirmed cases:",len(train_dt[train_dt["ConfirmedCases"]>=1]),"\n")

print("Percentage of corfirmed cases in the training dataset:",round(len(train_dt[train_dt["ConfirmedCases"]>=1])*100/len(train_dt),2),"%")

train_dt_confirmed = train_dt[train_dt["ConfirmedCases"]>=1]

train_dt_confirmed=train_dt_confirmed[["Country/Region","Date"]]

print("\nNumber of Corona Affected Countries:",train_dt_confirmed["Country/Region"].nunique(),"\n")

print("\nList of Corona Affected Countries....\n")

print(train_dt_confirmed.head(10))

df= train_dt_confirmed.groupby(["Country/Region"]).count()

df=df.sort_values("Date",ascending=False)

country_name = df.index.get_level_values('Country/Region')

corona_victims=[]

for i in range(len(df)):

    corona_victims.append(df["Date"][i])

cl = pd.DataFrame(corona_victims,columns=["Victim"]) # Converting List to Dateframe

df=df.head(80)

xlocs=np.arange(len(df))

df.plot.barh(color=[np.where(cl["Victim"]>20,"r","y")],figsize=(12,16))

plt.xlabel("Number of Corona Affected Patients",fontsize=12,fontweight="bold")

plt.ylabel("Country/Region",fontsize=12,fontweight="bold")

plt.title("Global Effect of Corona",fontsize=14,fontweight="bold")

for i, v in enumerate(df["Date"][:]):

    plt.text(v+0.01,xlocs[i]-0.25,str(v))

plt.legend(country_name) # top affected country

plt.show()

df_61 = []

number_countries = 0

total_victims=0

for i in range(df["Date"].shape[0]):

    if df["Date"][i] > 55:

        df_61.append(df["Date"][i])

        total_victims = total_victims + df["Date"][i]

        number_countries=number_countries+1

print("Number of countries where Corona Victims are more than 55 :", number_countries,"\n")

print("Total Number of Victims:",total_victims,"\n")        

explode=np.zeros(number_countries)

explode[0]=0.1

explode[1]=0.1

explode[2]=0.2

fig = plt.gcf() # gcf stands for Get Current Figure

fig.set_size_inches(10,10)

plt.pie(df_61,explode=explode,autopct='%1.1f%%',shadow=True, labels=country_name[0:number_countries])

title = "Contribution of Top "+str(number_countries) +" Countries" 

plt.title(title,fontsize=12, fontweight="bold")

plt.legend(loc="lower right",bbox_to_anchor=(1.1,0),bbox_transform=plt.gcf().transFigure) # bbx required to place legend without overlapping

plt.show()

train_dt_fatalities = train_dt[train_dt["Fatalities"]>=1]

print("Number of Countries where Death Toll is more than or equal to 1 is  ",len(train_dt_fatalities),"\n")

print("Out of ", len(train_dt)," Countries ",len(train_dt_fatalities)," Countries got fatalities\n")

print("Percentage of Countries got fatalities ",round(len(train_dt_fatalities)*100/len(train_dt),2),"%")
import datetime

Previous_Date = datetime.datetime.today() - datetime.timedelta(days=1)

Previous_Date=Previous_Date.strftime("%Y-%m-%d")

train_dt_fatalities_sort = train_dt_fatalities.sort_values("Fatalities",ascending=False)

train_dt_fatalities_sort = train_dt_fatalities_sort[train_dt_fatalities_sort["Date"]==Previous_Date]

train_dt_fatalities_sort 

fig=plt.gcf()

fig.set_size_inches(10,10)

xlocs=np.arange(len(train_dt_fatalities_sort["Country/Region"][0:10]))

plt.bar(train_dt_fatalities_sort["Country/Region"][0:10],train_dt_fatalities_sort["Fatalities"][0:10],alpha=0.5)

for i,v in enumerate(train_dt_fatalities_sort["Fatalities"][0:10]):

    plt.text(xlocs[i]-0.25,v+1,str(v))

title = "Number of Fatalities on :"+str(Previous_Date)

plt.title(title,fontsize=12,fontweight="bold")

plt.xlabel("Countries",fontsize=12,fontweight="bold")

plt.ylabel("Number of Death Toll Due To Corona",fontsize=12,fontweight="bold")

plt.legend(train_dt_fatalities_sort["Country/Region"])

fig.autofmt_xdate() # make space for and rotate the x-axis tick labels
fig,ax1 = plt.subplots()

ax2 = ax1.twinx()

df_china = train_dt[train_dt["Country/Region"]=="China"]

df_china = df_china.groupby("Date").sum().reset_index()

ax1.plot(df_china["Date"],df_china["ConfirmedCases"],linestyle="solid",linewidth=2,color='b')

ax2.plot(df_china["Date"],df_china["Fatalities"],linestyle="solid",linewidth=2,color='r')

ax1.set_xticklabels(df_china["Date"],rotation=90,fontsize="x-small",fontweight="bold")

ax1.set_ylabel("Confirmed Cases",fontweight="bold")

ax2.set_ylabel("Fatalities",fontweight="bold")

plt.legend(fontsize=10,fancybox=True, framealpha=1, shadow=True, borderpad=1,loc="upper right")

plt.title("China")

plt.show()
fig,ax1 = plt.subplots()

ax2 = ax1.twinx()

df_Italy = train_dt[train_dt["Country/Region"]=="Italy"]

df_Italy = df_Italy.groupby("Date").sum().reset_index()

ax1.plot(df_Italy["Date"],df_Italy["ConfirmedCases"],linewidth=2,linestyle="-",color='b')

ax2.plot(df_Italy["Date"],df_Italy["Fatalities"],linewidth=2,linestyle="-",color="r")

ax1.set_xticklabels(df_Italy["Date"],rotation=90,fontsize="x-small",fontweight="bold")

ax1.set_ylabel("Confirmed Cases",fontweight="bold")

ax2.set_ylabel("Fatalities",fontweight="bold")

plt.legend(fontsize=10,fancybox=True, framealpha=1, shadow=True, borderpad=1,loc="upper right")

plt.title("Italy")

plt.show()
fig,ax1=plt.subplots()

ax2 = ax1.twinx()

df_Australia = train_dt[train_dt["Country/Region"]=="Australia"]

df_Australia = df_Australia.groupby("Date").sum().reset_index()

ax1.plot(df_Australia["Date"],df_Australia["ConfirmedCases"],linewidth=2,linestyle="solid",color='b')

ax2.plot(df_Australia["Date"],df_Australia["Fatalities"],linewidth=2,linestyle="solid",color='r')

ax1.set_xticklabels(df_Australia["Date"],rotation=90,fontsize="x-small",fontweight="bold")

ax1.set_ylabel("Confirmed Cases",fontweight="bold")

ax2.set_ylabel("Fatalities",fontweight="bold")

plt.legend(fontsize=10,fancybox=True, framealpha=1, shadow=True, borderpad=1,loc="upper right")

plt.title("Australia")

plt.show()
fig,ax1=plt.subplots()

ax2=ax1.twinx()

df_US = train_dt[train_dt["Country/Region"]=="US"]

df_US = df_US.groupby("Date").sum().reset_index()

ax1.plot(df_US["Date"],df_US["ConfirmedCases"],linewidth=2,linestyle="solid",color='b')

ax2.plot(df_US["Date"],df_US["Fatalities"],linewidth=2,linestyle="solid",color='r')

ax1.set_xticklabels(df_US["Date"],rotation=90,fontsize="x-small",fontweight="bold")

ax1.set_ylabel("Confirmed Cases",fontweight="bold")

ax2.set_ylabel("Fatalities",fontweight="bold")

plt.legend(fontsize=10,fancybox=True, framealpha=1, shadow=True, borderpad=1,loc="upper right")

plt.title("US")

plt.show()
fig,ax1 = plt.subplots()

ax2=ax1.twinx()

df_Canada = train_dt[train_dt["Country/Region"]=="Canada"]

df_Canada = df_Canada.groupby("Date").sum().reset_index()

ax1.plot(df_Canada["Date"],df_Canada["ConfirmedCases"],linewidth=2,linestyle="solid",color='b')

ax2.plot(df_Canada["Date"],df_Canada["Fatalities"],linewidth=2,linestyle="solid",color='r')

ax1.set_xticklabels(df_Canada["Date"],rotation=90,fontsize="x-small",fontweight="bold")

ax1.set_ylabel("Confirmed Cases",fontweight="bold")

ax2.set_ylabel("Fatalities",fontweight="bold")

plt.legend(fontsize=10,fancybox=True, framealpha=1, shadow=True, borderpad=1,loc="upper right")

plt.title("Canada")

plt.show()
df = train_dt.groupby(["Date","Country/Region"])["ConfirmedCases","Fatalities"].max().reset_index()

fig= ply.scatter_geo(df,locations="Country/Region", locationmode='country names', 

                     color="ConfirmedCases", hover_name="Country/Region", 

                     range_color= [0, 1000], 

                     projection="natural earth", animation_frame="Date", 

                     title='Confirmed Cases of Corona Over Time', color_continuous_scale="portland")

fig.update(layout_coloraxis_showscale=False)

fig.show()