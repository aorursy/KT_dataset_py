# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import datetime

import requests

import warnings

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
icovid = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')

icovid['Date'] = pd.to_datetime(icovid['Date'],dayfirst = True)

age_details = pd.read_csv('../input/covid19-in-india/AgeGroupDetails.csv')

hospital_beds = pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')

individual_details = pd.read_csv('../input/covid19-in-india/IndividualDetails.csv')

icovid.head(3)
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

confirmed_df.head()

deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-04-2020.csv')
world_confirmed = confirmed_df.iloc[:,-1].sum()

world_recovered = recovered_df.iloc[:,-1].sum()

world_deaths = deaths_df.iloc[:,-1].sum()

world_active = world_confirmed - (world_recovered - world_deaths)

world_active

labels = ['Active','Recovered','Deceased']

sizes = [world_active,world_recovered,world_deaths]

color= ['orange','pink','yellow']

explode = [0,0,0.2]

plt.figure(figsize= (15,10))

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=9, explode =explode,colors = color)

plt.title('World COVID-19 Cases',fontsize = 30)

plt.figure(figsize=(10,10))

sizes,labels=[],['Missing', 'Male', 'Female']

sizes.append(individual_details["gender"].isnull().sum())

sizes.append(individual_details["gender"].value_counts()[0])

sizes.append(individual_details["gender"].value_counts()[1])

plt.pie(sizes,labels=labels,autopct='%1.1f%%',explode=[0,0,0.2],colors=["pink","cyan","yellow"])

plt.title('Percentage of Gender',fontsize = 20)
plt.figure(figsize=(10,10))

sizes,labels=[],['Male', 'Female']

sizes.append(individual_details["gender"].value_counts()[0])

sizes.append(individual_details["gender"].value_counts()[1])

plt.pie(sizes,labels=labels,autopct='%1.1f%%',explode=[0,0],colors=["cyan","yellow"],startangle=90)

plt.title('Percentage of Gender',fontsize = 20)
dates = list(confirmed_df.columns[4:])

dates=list(pd.to_datetime(dates))

dates_india = dates[8:]

dates_india

df1 = confirmed_df.groupby('Country/Region').sum()

df1=df1.reset_index()

df2 = deaths_df.groupby('Country/Region').sum().reset_index()

df3 = recovered_df.groupby('Country/Region').sum().reset_index()

k = df1[df1['Country/Region']=='India'].loc[:,'1/30/20':]

india_confirmed = k.values.tolist()[0] 



k = df2[df2['Country/Region']=='India'].loc[:,'1/30/20':]

india_deaths = k.values.tolist()[0] 



k = df3[df3['Country/Region']=='India'].loc[:,'1/30/20':]

india_recovered = k.values.tolist()[0] 



plt.figure(figsize= (15,10))

plt.xticks(rotation = 90 ,fontsize = 11)

plt.yticks(fontsize = 10)

plt.xlabel("Dates",fontsize = 20)

plt.ylabel('Total cases',fontsize = 20)

plt.title("Total Confirmed, Active, Death in India" , fontsize = 20)



ax1 = plt.plot_date(y= india_confirmed,x= dates_india,label = 'Confirmed',linestyle ='-',color = 'b')

ax2 = plt.plot_date(y= india_recovered,x= dates_india,label = 'Recovered',linestyle ='-',color = 'g')

ax3 = plt.plot_date(y= india_deaths,x= dates_india,label = 'Death',linestyle ='-',color = 'r')

plt.legend();
icovid = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')

#icovid['Date'] = pd.to_datetime(icovid['Date'],dayfirst = True)

icovid.drop(["ConfirmedIndianNational","ConfirmedForeignNational","Time","Sno"],axis=1,inplace=True)

icovid.rename(columns={"State/UnionTerritory":"State"},inplace=True)

icovid.head()
datelist=icovid["Date"].unique()

iconfirm=pd.DataFrame({'Date':datelist})

for state in icovid.State.unique():

        datalist=[]

        for date in datelist:

            d=icovid[(icovid["Date"]==date) & (icovid["State"]==state)]

            if(len(d["State"])>0):

                datalist.append(d["Confirmed"].values[0])

            else:

                datalist.append(0)

        iconfirm[state]=datalist
iconfirm.head()
iconfirmed=iconfirm.set_index("Date")

iconfirmed=iconfirmed.T

iconfirmed.tail()
icovid.State.unique()
plt.figure(figsize=(20,5))

#qu=input("\nEnter State to know the maximum number of confirmed case in a particular day:")

qu="Maharashtra"

r=iconfirmed.loc[qu].diff().tolist()

r.pop(0)

print("\nFor",qu,"State The maximum number of confirmed cases is",iconfirmed.loc[qu].diff().max(),"on",datelist[r.index(max(r))+1])
iconfirmed.loc["Kerala"]
#list1=list(input("Enter 2 space separated State name for comparative study: ").split(","))

list1=["West Bengal","Tamil Nadu"]

plt.figure(figsize=(20,5))

iconfirmed.loc[list1[1]].plot()

iconfirmed.loc[list1[0]].plot()

plt.legend()