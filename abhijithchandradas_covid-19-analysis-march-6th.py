# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Data visualization

import folium #for mapping 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_conf=pd.read_csv('../input/coronavirus-6th-mar-2020-johns-hopkins-university/time_series_19-covid-Confirmed.csv')

df_death=pd.read_csv('../input/coronavirus-6th-mar-2020-johns-hopkins-university/time_series_19-covid-Deaths.csv')

df_rec=pd.read_csv('../input/coronavirus-6th-mar-2020-johns-hopkins-university/time_series_19-covid-Recovered.csv')



print("conf_shape:", df_conf.shape)                     

print("death_shape:", df_death.shape)

print("rec_shape:", df_rec.shape)



df_death.head()
import folium

world_map = folium.Map(location=[0,0], zoom_start=2,tiles='CartoDB positron')



for lat, lon,conf in zip(df_conf['Lat'], df_conf['Long'],df_conf.iloc[:,-1]):

    folium.CircleMarker([lat, lon], 

                        radius=5,color='red',

                        fill_color='red',fill_opacity=0.5,

                        tooltip="Confirmed Cases :"+str(conf)).add_to(world_map)

world_map
df_chn_full=df_conf[df_conf["Country/Region"]=="Mainland China"] #Data within China



china_map = folium.Map(location=[35,100], zoom_start=4,tiles='Stamen Toner')



for lat, lon,state, cases in zip(df_chn_full['Lat'], df_chn_full['Long'],df_chn_full['Province/State'], df_chn_full.iloc[:,-1]):

    folium.CircleMarker([lat, lon],

                        radius=5,

                        color='red',

                        tooltip =('Province: ' + str(state))+ '<br>' +"Cases :"+str(cases),

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(china_map)

china_map
def stats_calculator(df_conf,df_death,df_rec):

    """

    Docstring:

    Calculate the number of Total Deaths, Confirmed Cases, Recoveries, New Cases and Mortality Rate each day

    

    Takes in confirmed, death and recovered dataframes as argument

    Return a dataframe

    """

    #Extract dates

    date=df_death.keys()[4:]

    

    #Initialize Stats

    total_death_date=[]

    total_rec_date=[]

    total_conf_date=[]

    mortality_rate=[]

    

    #Append Stats

    for i in date:

        total_death_date.append(df_death[i].sum())

        total_rec_date.append(df_rec[i].sum())

        total_conf_date.append(df_conf[i].sum())

        mortality_rate.append(df_death[i].sum()*100/(df_death[i].sum()+df_conf[i].sum()))



    #Make stats into a dataframe

    df_temp=pd.DataFrame(data=date, columns=["date"])

    df_temp["death"]=total_death_date

    df_temp["rec"]=total_rec_date

    df_temp["conf"]=total_conf_date

    df_temp["mort_rate"]=mortality_rate

    df_temp['new_cases']=df_temp.conf

    for i in range(1,df_temp.shape[0]):

        df_temp.new_cases[i]=df_temp.conf[i]-df_temp.conf[i-1]

    

    #Return dataframe

    return(df_temp)
def stats_plotter(df_temp):

    """

    Docstring: 

    Input dataframe object

    Plots Total Confirmed Cases, Recovered Cases, Total Deaths Vs Date 

    """

    

    #Print Latest Stats

    print("Total Confirmed Cases :",df_temp.conf.iloc[-1])

    print("Total Recovered Patients :",df_temp.rec.iloc[-1])

    print("Total Deaths :",df_temp.death.iloc[-1])

    

    #Plot Stats

    plt.figure(figsize=(10,6))

    plt.title("COVID-19 Time Series")

    plt.plot(df_temp.date,df_temp.death, label="Death")

    plt.plot(df_temp.date,df_temp.conf, label="Confirmed Cases")

    plt.plot(df_temp.date,df_temp.rec, label="Recovered")

    plt.xticks(rotation=90)

    plt.legend()

    plt.grid()

    plt.show()

    

    #Plot Mortality Rate

    print("Mortality Rate: "+ str(round(df_temp.mort_rate.iloc[-1],2))+"%")

    plt.figure(figsize=(10,4))

    plt.title("COVID-19 Mortality Rate (in %) Vs Time")

    plt.plot(df_temp.date,df_temp.mort_rate, label="Mortality Rate")

    plt.xticks(rotation=90)

    plt.legend()

    plt.grid()

    plt.show()
def only_country(country_name):

    """

    Input- Country Name

    Output- Data of that country

    """

    df_death_ctry=df_death[df_death["Country/Region"]==country_name]

    df_rec_ctry=df_rec[df_rec["Country/Region"]==country_name]

    df_conf_ctry=df_conf[df_conf["Country/Region"]==country_name]



    return(stats_calculator(df_conf_ctry,df_death_ctry,df_rec_ctry))



def except_country(country_name):

    """

    Input- Country Name

    Output- Data outside that country

    """

    df_death_ctry=df_death[df_death["Country/Region"]!=country_name]

    df_rec_ctry=df_rec[df_rec["Country/Region"]!=country_name]

    df_conf_ctry=df_conf[df_conf["Country/Region"]!=country_name]



    return(stats_calculator(df_conf_ctry,df_death_ctry,df_rec_ctry))
df_total=stats_calculator(df_conf,df_death,df_rec)

stats_plotter(df_total)
stats_plotter(only_country("South Korea"))

df_chn=only_country("Mainland China")  #dataframe within China

df_oc=except_country("Mainland China") #dataframe outside China



plt.figure(figsize=(12,8))

plt.title("COVID-19 Spred: China Vs Rest of the World", fontsize=18)

plt.plot(df_total.date,df_total.conf,'k--', label="Global")

plt.plot(df_total.date,df_chn.conf, label="Inside China")

plt.plot(df_total.date,df_oc.conf, label="Outside China")

plt.legend()

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(12,8))

plt.title("COVID-19 New Cases: China Vs Rest of the World", fontsize=18)

plt.plot(df_total.date,df_total.new_cases,'k--', label="Global")

plt.plot(df_total.date,df_chn.new_cases, label="Inside China")

plt.plot(df_total.date,df_oc.new_cases, label="Outside China")

plt.legend()

plt.xticks(rotation=90)

plt.show()
c=df_chn.tail(5).new_cases.sum()

print("New Cases in China in last 5 days: ",c)

o=df_oc.tail(5).new_cases.sum()

print("New Cases outside China in last 5 days: ",o)

print("Percentage of Cases outside China : ", round(o*100/(o+c),2),"%")