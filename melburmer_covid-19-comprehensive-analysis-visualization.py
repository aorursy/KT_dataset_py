# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import warnings

warnings.filterwarnings('ignore')

import pandas as pd 

import plotly.express as px

import math

%matplotlib inline 
df = pd.read_csv(r"/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

df_line = pd.read_csv(r"/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")



df_open = pd.read_csv(r"/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")





#preprocess

df_line = df_line.iloc[:,:-6]

df_line.drop(['Unnamed: 3',"source","link"],axis=1,inplace=True)
confirmed = df.groupby('ObservationDate').sum()['Confirmed'].reset_index()

deaths = df.groupby('ObservationDate').sum()['Deaths'].reset_index()

recovered = df.groupby('ObservationDate').sum()['Recovered'].reset_index()





fig = go.Figure()

fig.add_trace(go.Scatter(x=confirmed['ObservationDate'], 

                         y=confirmed['Confirmed'],

                         mode='lines+markers',

                         name='Confirmed',

                         line=dict(color='blue', width=2)

                        ))

fig.add_trace(go.Scatter(x=deaths['ObservationDate'], 

                         y=deaths['Deaths'],

                         mode='lines+markers',

                         name='Deaths',

                         line=dict(color='Red', width=2)

                        ))

fig.add_trace(go.Scatter(x=recovered['ObservationDate'], 

                         y=recovered['Recovered'],

                         mode='lines+markers',

                         name='Recovered',

                         line=dict(color='Green', width=2)

                        ))

fig.update_layout(

    title='Worldwide Corona Virus Cases - Confirmed, Deaths, Recovered (Line Chart)',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Number of Cases',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    )

)

fig.show()

# China



china = df[df["Country/Region"] == "Mainland China"]

confirmed_china = china.groupby('ObservationDate').sum()['Confirmed'].reset_index()

deaths_china = china.groupby('ObservationDate').sum()['Deaths'].reset_index()

recovered_china = china.groupby('ObservationDate').sum()['Recovered'].reset_index()



# USA

usa = df[df["Country/Region"] == "US"]

confirmed_usa = usa.groupby('ObservationDate').sum()['Confirmed'].reset_index()

deaths_usa = usa.groupby('ObservationDate').sum()['Deaths'].reset_index()

recovered_usa = usa.groupby('ObservationDate').sum()['Recovered'].reset_index()



# ıtaly



italy = df[df["Country/Region"] == "Italy"]

confirmed_italy = italy.groupby('ObservationDate').sum()['Confirmed'].reset_index()

deaths_italy = italy.groupby('ObservationDate').sum()['Deaths'].reset_index()

recovered_italy = italy.groupby('ObservationDate').sum()['Recovered'].reset_index()



# Germany



germany = df[df["Country/Region"] == "Germany"]

confirmed_germany = germany.groupby('ObservationDate').sum()['Confirmed'].reset_index()

deaths_germany =germany.groupby('ObservationDate').sum()['Deaths'].reset_index()

recovered_germany = germany.groupby('ObservationDate').sum()['Recovered'].reset_index()
# Visualization

plt.rc('ytick', labelsize=20) 

plt.rc('xtick', labelsize=10)

fig, axes = plt.subplots(2,2,sharex=True,figsize=(20,20))



axes[0,0].plot(confirmed_china['ObservationDate'],confirmed_china['Confirmed'],label="Confirmed",color="Blue",linewidth=5)

axes[0,0].plot(recovered_china['ObservationDate'],recovered_china['Recovered'],label="Recovered",color="Green",linewidth=4)

axes[0,0].plot(deaths_china['ObservationDate'],deaths_china['Deaths'],label="Deaths",color="Red",linewidth=4)

axes[0,0].legend(prop={'size': 20},shadow=True)

axes[0,0].set_title("China Corona Virus Cases - Confirmed, Deaths, Recovered",weight="bold",fontsize=15)





axes[0,1].plot(confirmed_usa['ObservationDate'],confirmed_usa['Confirmed'],label="Confirmed",color="Blue",linewidth=5)

axes[0,1].plot(recovered_usa['ObservationDate'],recovered_usa['Recovered'],label="Recovered",color="Green",linewidth=4)

axes[0,1].plot(deaths_usa['ObservationDate'],deaths_usa['Deaths'],label="Deaths",color="Red",linewidth=4)

axes[0,1].legend(prop={'size': 20},shadow=True)

axes[0,1].set_title("USA Corona Virus Cases - Confirmed, Deaths, Recovered",weight="bold",fontsize=15)





axes[1,0].plot(confirmed_italy['ObservationDate'],confirmed_italy['Confirmed'],label="Confirmed",color="Blue",linewidth=5)

axes[1,0].plot(recovered_italy['ObservationDate'],recovered_italy['Recovered'],label="Recovered",color="Green",linewidth=4)

axes[1,0].plot(deaths_italy['ObservationDate'],deaths_italy['Deaths'],label="Deaths",color="Red",linewidth=4)

axes[1,0].legend(loc='upper left', bbox_to_anchor=(0, 1),

           fancybox=True, shadow=True,prop={'size': 20})

axes[1,0].set_title("Italy Corona Virus Cases - Confirmed, Deaths, Recovered",weight="bold",fontsize=15)



axes[1,1].plot(confirmed_germany['ObservationDate'],confirmed_germany['Confirmed'],label="Confirmed",color="Blue",linewidth=5)

axes[1,1].plot(recovered_germany['ObservationDate'],recovered_germany['Recovered'],label="Recovered",color="Green",linewidth=4)

axes[1,1].plot(deaths_germany['ObservationDate'],deaths_germany['Deaths'],label="Deaths",color="Red",linewidth=4)

axes[1,1].legend(loc='upper left', bbox_to_anchor=(0, 1),

           fancybox=True, shadow=True,prop={'size': 20})

axes[1,1].set_title("Germany Corona Virus Cases - Confirmed, Deaths, Recovered",weight="bold",fontsize=15)





axes[1][0].xaxis.set_tick_params(rotation=90)

axes[1][1].xaxis.set_tick_params(rotation=90)



plt.show()
deaths = df_line[df_line["death"]!="0"]



print("Average age of death: " + str(int(round(deaths["age"].mean()))))

print("\n=> One of the biggest reasons for the high average age of death is low immunity of elderly people.")
male_number = deaths["gender"].value_counts()["male"]



female_number = deaths["gender"].value_counts()["female"]



total = female_number  + male_number



print("% of deaths by gender => Male: %{0}, Female: %{1}".format(int(round(male_number*100/total)),int(round(female_number*100/total))))
only_death_date = deaths[df_line["death"]!="1"]



only_death_date.dropna(subset=['symptom_onset'],inplace=True)



only_death_date.reset_index(inplace=True)



only_death_date["death_time(day)"] = (pd.to_datetime(only_death_date['death']) - pd.to_datetime(only_death_date['symptom_onset']))



only_death_date["death_time(day)"] = only_death_date["death_time(day)"].apply(lambda x: int(x.days))





only_death_date = only_death_date[["location","country","gender","age","symptom_onset","death","symptom","death_time(day)"]]



only_death_date.drop(["location"],axis=1,inplace=True)

# there are not enough data, lets use df_open dataset to gain more data.



df_open["outcome"] = df_open["outcome"].apply(lambda x:"death" if x== "died" else x)



deaths_df2 = df_open[df_open["outcome"]=="death"]

deaths_df2.dropna(subset=['date_death_or_discharge',"date_onset_symptoms"],inplace=True)



deaths_df2 = deaths_df2[["province","country","sex","age","date_onset_symptoms","symptoms",'date_death_or_discharge']]

deaths_df2.reset_index(inplace=True)



deaths_df2['date_death_or_discharge'][0] = "02.09.2020"

deaths_df2["death_time(day)"] = (pd.to_datetime(deaths_df2['date_death_or_discharge']) - pd.to_datetime(deaths_df2['date_onset_symptoms']))

deaths_df2["death_time(day)"] = deaths_df2["death_time(day)"].apply(lambda x: int(x.days))



deaths_df2["country"][2] =deaths_df2["province"][2]

deaths_df2.drop(["index","province"],axis=1,inplace=True)





deaths_df2.columns = ["country","gender","age","symptom_onset","symptom","death","death_time(day)"]



only_death_date = only_death_date.append(deaths_df2)

only_death_date.reset_index(inplace=True)



only_death_date.drop(["index"],axis=1,inplace=True)
import seaborn as sns

only_death_date = only_death_date[['country','age','gender','symptom_onset','death','death_time(day)','symptom']]



cm = sns.light_palette("red", as_cmap=True)

s = only_death_date .style.background_gradient(cmap=cm)

s
print("Average death time after symptom onset:{0} days".format(round(only_death_date["death_time(day)"].mean())))
print("Average death time after symptom onset by gender (Note: There is not enough data to make a good inference.)\n=> male:{} days, female:{} days"

      .format(int(only_death_date.groupby("gender").mean()["death_time(day)"]["male"]),int(only_death_date.groupby("gender").mean()["death_time(day)"]["female"])))
recovered = df_line[df_line["recovered"] != "0"]



print("Average recovered age: " + str(round(recovered["age"].mean())))
male_number = recovered["gender"].value_counts()["male"]



female_number = recovered["gender"].value_counts()["female"]



total = female_number  + male_number



print("% of recovered by gender => Male: %{0}, Female: %{1}".format(int(round(male_number*100/total)),int(round(female_number*100/total))))

print("\nImportant Note: As we have analyzed before, the number of infected male is higher than the number of infected female. So it is necessary to be careful when evaluating this result.")
recovered = recovered[recovered["recovered"] != "1"]



recovered = recovered[[ 'reporting date', 'country', 'gender', 'age','recovered','symptom']]



# deleting wrong entries (like recovered date: 12/30/1899 etc.)

recovered.reset_index(inplace=True,drop=True)



for index,date in enumerate(recovered["recovered"].values):

    if(date.split("/")[2] == "1899"):

        recovered.drop(index,axis=0,inplace=True)



    

recovered["recover_time"] = (pd.to_datetime(recovered['recovered']) - pd.to_datetime(recovered['reporting date']))

recovered["recover_time"] = recovered["recover_time"].apply(lambda x: int(x.days))





#  some fixes

recovered = recovered[recovered["recover_time"]>=0]



recovered["recover_time"] = recovered["recover_time"].apply(lambda x: 1 if x== 0 else x)



recovered = recovered[['country', 'gender', 'age','reporting date', 'recovered','recover_time', 'symptom']]

print("Average treatment time(days): {0}".format(round(recovered["recover_time"].mean())))
print("\nAverage treatment time(days) (by gender) => Male: {0} Female: {1}"

      .format(round(recovered.groupby("gender").mean()["recover_time"]["male"])

      ,round(recovered.groupby("gender").mean()["recover_time"]["female"])))
recovered_by_country = recovered.groupby("country").mean()



recovered_by_country["age"] = recovered_by_country["age"].apply(lambda x:int(x) if(math.isnan(x)==False) else x)



recovered_by_country["recover_time"] = recovered_by_country["recover_time"].apply(lambda x:int(x) if(math.isnan(x)==False) else x)



recovered_by_country.rename(columns={'age':"avarage age","recover_time":"avarage recover time(days)"},inplace=True)



cm = sns.light_palette("green", as_cmap=True)

s2 = recovered_by_country.style.background_gradient(cmap=cm)

s2
def set_interval(age):

    if(age>0 and age<=10):

        return "0-10"

    elif(age> 10 and age<=18):

        return "10-18"

    elif(age>18 and age <30 ):

        return "18-30"

    elif(age>=30 and age <60):

        return "30-60"

    elif(math.isnan(age)):

        return np.nan

    else :

        return "60+"

    



recovered["age_interval"] = recovered["age"].apply(set_interval)



recover_by_age = recovered.groupby("age_interval").mean()["recover_time"]



recover_by_age = recover_by_age.astype(int)



recover_by_age = pd.DataFrame(recover_by_age.values,index =recover_by_age.index,columns=["avg recover time(days)"])



cm = sns.light_palette("seagreen", as_cmap=True)

s3 = recover_by_age.style.background_gradient(cmap=cm)

s3
# lets use df_open



symptoms = df_open.copy()

symptoms.dropna(subset=['symptoms'],inplace=True)



symptoms.drop(11233,axis=0,inplace=True)



# creating symptoms dict



symp_dict = {"fever":0,"cough":0,"pneumonia":0,"fatigue":0,"headache":0,

             "chills":0,"sputum":0,"joint_pain":0,"diarrhea":0,

             "runny_nose":0,"malaise":0,"vomiting":0,"nausea":0}



for sym in symptoms['symptoms'].values:

    if("fever" in sym.lower()):

        symp_dict["fever"]+=1

    if("cough" in sym.lower()):

        symp_dict["cough"]+=1 

    if("pneumonitis" in sym.lower() or "pneumonia" in sym.lower() ):

        symp_dict["pneumonia"]+=1        

    if("fatigue" in sym.lower()):

        symp_dict["fatigue"]+=1 

    if("headache" in sym.lower()):

        symp_dict["headache"]+=1    

        

    if("chills" in sym.lower()):

        symp_dict["chills"]+=1 

    if("sputum" in sym.lower()):

        symp_dict["sputum"]+=1

    if("joint" in sym.lower()):

        symp_dict["joint_pain"]+=1

        

    if("diar" in sym.lower()):

        symp_dict["diarrhea"]+=1 

        

    if("runny" in sym.lower()):

        symp_dict["runny_nose"]+=1 

        

    if("mala" in sym.lower()):

        symp_dict["malaise"]+=1 

       

    if("vomit" in sym.lower()):

        symp_dict["vomiting"]+=1      

    if("nau" in sym.lower()):

        symp_dict["nausea"]+=1         



sympts = pd.DataFrame(data = symp_dict.items(),columns=["symptom","symptom_count"])  # dict to dataframe



sympts.set_index("symptom",inplace=True)

        

sympts["Occurance Rate"] = sympts["symptom_count"]/symptoms.shape[0]*100



sympts["Occurance Rate"] = sympts["Occurance Rate"].apply(lambda x: "% "+str(int(round(x))))



cm = sns.light_palette("#c90000", as_cmap=True)

s = sympts.style.background_gradient(cmap=cm)

s
after_days = df_open.dropna(subset=["date_onset_symptoms","date_admission_hospital"])





after_days = after_days[["country","age","sex","outcome","date_onset_symptoms","date_admission_hospital"]]





after_days.drop([476,477,478,479,480,1379,101],inplace=True,axis=0) # deleting some rows



after_days["hospital_admission_time"]  = (pd.to_datetime(after_days["date_admission_hospital"],utc=True,dayfirst=True) - pd.to_datetime(after_days["date_onset_symptoms"],utc=True,dayfirst=True))



after_days["hospital_admission_time"] = after_days["hospital_admission_time"].apply(lambda x: int(x.days))





after_days = after_days[after_days["hospital_admission_time"]>=0]

after_days.reset_index(inplace=True,drop=True)



after_days["hospital_admission_time"] = after_days["hospital_admission_time"].apply(lambda x: 1 if x==0 else x)



cm = sns.light_palette("#c99100", as_cmap=True)

s = after_days.sample(n=10).style.background_gradient(cmap=cm)

s
print("\nAverage hospital admission time(days) after symptoms on set:{0} ".format(int(round(after_days["hospital_admission_time"].mean()))))
admission_by_country = after_days.groupby("country").mean()["hospital_admission_time"].apply(lambda x: int(round(x)) )

admission_by_country = pd.DataFrame(admission_by_country)

admission_by_country.columns = ["Average hospital_admission_time (day)"]

cm = sns.light_palette("#ed6e6d", as_cmap=True)

s = admission_by_country.sample(n=10).style.background_gradient(cmap=cm)

s
df = pd.read_csv(r"/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")
total_confirmed = df.groupby('Country/Region')['3/14/20'].sum()



total_confirmed = pd.DataFrame(data=total_confirmed,index=total_confirmed.index)



total_confirmed.columns = ["Total Confirmed"]



total_confirmed.sort_values("Total Confirmed",axis=0,inplace=True,ascending=False)



# top five

cm = sns.light_palette("#5c60c7", as_cmap=True)

s = total_confirmed.head().style.background_gradient(cmap=cm)

s
ax1 = total_confirmed[0:10].plot(kind="barh",colormap="coolwarm",width=0.7,stacked=True,figsize=(10,10))

ax1.set_title("Top 10 Confirmed Case Count by Country",fontweight="bold")

ax1.set_xlabel("Total Confirmed")



for p in ax1.patches:

    left, bottom, width, height = p.get_bbox().bounds

    if(width<4000):

        ax1.annotate(str(int(width)), xy=(left+3*width, bottom+height/2),ha='center', va='center',fontweight="bold")

    else:

        ax1.annotate(str(int(width)), xy=(left+width/2, bottom+height/2),ha='center', va='center',fontweight="bold")
df.drop([104,105],axis=0,inplace=True) # removing duplicate records

df["Province/State"].fillna(df["Country/Region"],inplace=True)



total_confirmed = pd.DataFrame(df.groupby(["Country/Region",'Province/State'])['3/14/20'].sum())



total_confirmed.columns = ["Total Confirmed"]



total_confirmed["Lat"] = np.nan

total_confirmed["Long"] = np.nan







for loc in total_confirmed.index:

    

    temp = df[(df["Country/Region"] == loc[0]) & (df["Province/State"]== loc[1])]

    total_confirmed["Lat"][loc] = temp["Lat"]

    total_confirmed["Long"][loc] = temp["Long"]





total_confirmed = total_confirmed[total_confirmed["Total Confirmed"]>0]     


fig = px.density_mapbox(total_confirmed, 

                        lat="Lat", 

                        lon="Long", 

                        hover_name=total_confirmed.index.get_level_values(1), 

                        hover_data=["Total Confirmed"], 

                        color_continuous_scale="Portland",

                        radius=7, 

                        zoom=1,height=600,width=1200)    

fig.update_layout(title="World Wide Corona Virus Cases Confirmed",

                  font=dict(family="Courier New, monospace",

                            size=18,

                            color="#7f7f7f")

                 )

fig.update_layout(mapbox_style="open-street-map", mapbox_center_lon=0)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
df = pd.read_csv(r"/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")
# ülkelere göre toplam ölüm sayıları



total_deaths = df.groupby('Country/Region')['3/14/20'].sum()



total_deaths = pd.DataFrame(data=total_deaths,index=total_deaths.index)



total_deaths.columns = ["Total Deaths"]



total_deaths.sort_values("Total Deaths",axis=0,inplace=True,ascending=False)



# top five

cm = sns.light_palette("#cb504e", as_cmap=True)

s = total_deaths.head().style.background_gradient(cmap=cm)

s
# Top 10 country



ax1 = total_deaths[0:10].plot(kind="barh",colormap="Set1",width=0.7,stacked=True,figsize=(10,10))

ax1.set_title("Top 10 Death Count by Country",fontweight="bold")

ax1.set_xlabel("Total Deaths")



for p in ax1.patches:

    left, bottom, width, height = p.get_bbox().bounds

    if(width<60):

        ax1.annotate(str(int(width)), xy=(left+3*width+10, bottom+height/2),ha='center', va='center',fontweight="bold")

    else:

        ax1.annotate(str(int(width)), xy=(left+width/2, bottom+height/2),ha='center', va='center',fontweight="bold")
df.drop([104,105],axis=0,inplace=True) # dublike kayıtlar 

df["Province/State"].fillna(df["Country/Region"],inplace=True)



total_deaths = pd.DataFrame(df.groupby(["Country/Region",'Province/State'])['3/14/20'].sum())



total_deaths.columns = ["Total Deaths"]



total_deaths["Lat"] = np.nan

total_deaths["Long"] = np.nan





for loc in total_deaths.index:

    temp = df[(df["Country/Region"] == loc[0]) & (df["Province/State"]== loc[1])]

    total_deaths["Lat"][loc] = temp["Lat"]

    total_deaths["Long"][loc] = temp["Long"]





total_deaths = total_deaths[total_deaths["Total Deaths"]>0]   
fig = px.density_mapbox(total_deaths, 

                        lat="Lat", 

                        lon="Long", 

                        hover_name=total_deaths.index.get_level_values(1), 

                        hover_data=["Total Deaths"], 

                        color_continuous_scale="Portland",

                        radius=8, 

                        zoom=1,height=600,width=1200)    

fig.update_layout(title="World Wide Corona Virus Death Cases",

                  font=dict(family="Courier New, monospace",

                            size=18,

                            color="#7f7f7f")

                 )

fig.update_layout(mapbox_style="open-street-map", mapbox_center_lon=0)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
df = pd.read_csv(r"/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
# all world



total_recovered = df.groupby('Country/Region')['3/14/20'].sum()



total_recovered = pd.DataFrame(data=total_recovered,index=total_recovered.index)



total_recovered.columns = ["Total Recovered"]



total_recovered.sort_values("Total Recovered",axis=0,inplace=True,ascending=False)





# top five

cm = sns.light_palette("#68a05f", as_cmap=True)

s = total_recovered.head().style.background_gradient(cmap=cm)

s
# Top 10 country



ax1 = total_recovered[0:10].plot(kind="barh",colormap="summer",width=0.7,stacked=True,figsize=(10,10))

ax1.set_title("Top 10 Recovered Count by Country",fontweight="bold")

ax1.set_xlabel("Total Recovered")



for p in ax1.patches:

    left, bottom, width, height = p.get_bbox().bounds

    if(width<2000):

        ax1.annotate(str(int(width)), xy=(left+4*width+5, bottom+height/2),ha='center', va='center',fontweight="bold")

    else:

        ax1.annotate(str(int(width)), xy=(left+width/2, bottom+height/2),ha='center', va='center',fontweight="bold")
df.drop([104,105],axis=0,inplace=True) # dublike kayıtlar 



df["Province/State"].fillna(df["Country/Region"],inplace=True)



total_recovered = pd.DataFrame(df.groupby(["Country/Region",'Province/State'])['3/14/20'].sum())



total_recovered.columns = ["Total Recovered"]



total_recovered["Lat"] = np.nan

total_recovered["Long"] = np.nan



for loc in total_recovered.index:

    temp = df[(df["Country/Region"] == loc[0]) & (df["Province/State"]== loc[1])]

    total_recovered["Lat"][loc] = temp["Lat"]

    total_recovered["Long"][loc] = temp["Long"]





total_recovered = total_recovered[total_recovered["Total Recovered"]>0]
for loc in total_recovered.index:

    temp = df[(df["Country/Region"] == loc[0]) & (df["Province/State"]== loc[1])]

    total_recovered["Lat"][loc] = temp["Lat"]

    total_recovered["Long"][loc] = temp["Long"]





total_recovered = total_recovered[total_recovered["Total Recovered"]>0]     

 

fig = px.density_mapbox(total_recovered, 

                        lat="Lat", 

                        lon="Long", 

                        hover_name=total_recovered.index.get_level_values(1), 

                        hover_data=["Total Recovered"], 

                        color_continuous_scale="Portland",

                        radius=7, 

                        zoom=1,height=600,width=1200)    

fig.update_layout(title="World Wide Corona Virus Cases Deaths",

                  font=dict(family="Courier New, monospace",

                            size=18,

                            color="#7f7f7f")

                 )

fig.update_layout(mapbox_style="open-street-map", mapbox_center_lon=0)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
df = pd.read_csv(r"/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv",parse_dates=['Last Update'])



df_confirmed =  pd.read_csv(r"/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")





df.rename(columns={'ObservationDate':'Date'}, inplace=True)





df_confirmed = df_confirmed[["Province/State","Lat","Long","Country/Region"]]



df['Country/Region'].replace({'Mainland China': 'China'}, inplace=True)





df_cordi = pd.merge(df, df_confirmed, on=["Country/Region", "Province/State"])



fig = px.density_mapbox(df_cordi, 

                        lat="Lat", 

                        lon="Long", 

                        hover_name="Province/State", 

                        hover_data=["Confirmed","Deaths","Recovered"], 

                        animation_frame="Date",

                        color_continuous_scale="Portland",

                        radius=7, 

                        zoom=1,height=700,width=1200)

fig.update_layout(title='Worldwide Corona Virus Cases Time Lapse - Confirmed, Deaths, Recovered',

                  font=dict(family="Courier New, monospace",

                            size=18,

                            color="#7f7f7f")

                 )

fig.update_layout(mapbox_style="open-street-map", mapbox_center_lon=0)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})





fig.show()