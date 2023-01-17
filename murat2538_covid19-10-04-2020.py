import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
covid_19_confirmed_global=pd.read_csv("/kaggle/input/cssegisanddata-covid19/time_series_covid19_confirmed_global.csv")
covid_19_deaths_global=pd.read_csv("/kaggle/input/cssegisanddata-covid19/time_series_covid19_deaths_global.csv")
covid_19_recovered_global=pd.read_csv("/kaggle/input/cssegisanddata-covid19/time_series_covid19_recovered_global.csv")
death=np.array(covid_19_deaths_global["4/10/20"])
confirmed=np.array(covid_19_confirmed_global["4/10/20"])
Country_Region=np.array(covid_19_deaths_global["Country/Region"])

death_rate=death/(confirmed)

df=pd.DataFrame({"Country/Region":Country_Region,
                 "observ_time":"10/04/2020",
                 "Deaths":death,
                 "Confirmed":confirmed,
                 "Death_Rate":death_rate})
death.shape
confirmed.shape
df.sort_values(by="Confirmed",ascending=False).head(10)
df.shape
df.info()
df.isnull().sum()
df.describe().T
confirmed_total_cases=np.sum(df["Confirmed"])
deaths_total_cases=np.sum(df["Deaths"])
recovered_total_cases=np.sum(covid_19_recovered_global["4/10/20"])

print("total number of confirmed cases in the world as of April 10, 2020:",confirmed_total_cases)
print("total number of deaths cases in the world as of April 10, 2020:",deaths_total_cases)
print("total number of recovered cases in the world as of April 10, 2020:",recovered_total_cases)
cases10000=df[df["Confirmed"]>10000]
cases10000=cases10000.sort_values(by="Confirmed",ascending=False)
cases10000.head()
fig, axs = plt.subplots(1, 2, figsize=(20,18), sharey=True)
axs[0].set_title("Confirmed Cases")
axs[0].pie(cases10000["Confirmed"].head(10),labels=cases10000["Country/Region"].head(10),autopct='%1.2f%%');
axs[1].set_title("Deaths Cases")
axs[1].pie(cases10000["Deaths"].head(10),labels=cases10000["Country/Region"].head(10),autopct='%1.2f%%');
plt.figure(figsize=(10,10))
sns.barplot(x="Confirmed", y="Country/Region", data=cases10000.head(10));
plt.title("number of cases > 10000")
plt.xlabel("number of cases");
cases10000_death=df[df["Confirmed"]>10000]
cases10000_death=cases10000.sort_values(by="Deaths",ascending=False).head(10)
cases10000_death.head()
plt.figure(figsize=(10,10))
sns.barplot(x="Deaths", y="Country/Region", data=cases10000_death);
plt.title("10 Countries with the Most Deaths")
plt.ylabel("Country/Region")
plt.xlabel("Number of Deaths");
df.head()
fig = px.choropleth(df, 
                    locations="Country/Region", 
                    locationmode = "country names",
                    color="Confirmed",
                    hover_name="Country/Region", 
                    animation_frame="observ_time"
                   )

fig.update_layout(
    title_text = 'Distribution of Coronavirus Confirmed by Countries',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))
    
fig.show()
fig = px.choropleth(df, 
                    locations="Country/Region", 
                    locationmode = "country names",
                    color="Deaths",
                    hover_name="Country/Region", 
                    animation_frame="observ_time"
                   )

fig.update_layout(
    title_text = 'Distribution of Coronavirus Deaths by Countries',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))
    
fig.show()
recovered=covid_19_recovered_global.copy()
recovered_number=np.array(recovered["4/10/20"])
df_recovered=pd.DataFrame({"Country/Region":recovered["Country/Region"],
                           "observ_time":"10/04/2020",
                            "Recovered":recovered_number})
df_recovered.head()
fig = px.choropleth(df_recovered, 
                    locations="Country/Region", 
                    locationmode = "country names",
                    color="Recovered",
                    hover_name="Country/Region", 
                    animation_frame="observ_time"
                   )

fig.update_layout(
    title_text = 'Distribution of Coronavirus Recovered by Countries',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))
    
fig.show()
confirmed_total_cases=np.array(np.sum(df["Confirmed"]))
deaths_total_cases=np.array(np.sum(df["Deaths"]))
recovered_total_cases=np.array(np.sum(covid_19_recovered_global["4/10/20"]))
Country_region="All_World"

df_all_world=pd.DataFrame([confirmed_total_cases,
                           deaths_total_cases,
                           recovered_total_cases], index=["Confirmed","Deaths","Recovered"],
                           columns=["All_World"])
df_all_world
df_all_world.plot.pie(y="All_World",figsize=(10,10),autopct='%1.2f%%');
df_all_world=df_all_world.sort_values(by="All_World")
df_all_world.plot.barh(y="All_World",figsize=(10,10),color="blue");
death_china=covid_19_deaths_global[covid_19_deaths_global["Country/Region"]=="China"]
death_china=np.sum(death_china["4/10/20"])

death_italy=covid_19_deaths_global[covid_19_deaths_global["Country/Region"]=="Italy"]
death_italy=np.sum(death_italy["4/10/20"])
recovered_china=covid_19_recovered_global[covid_19_recovered_global["Country/Region"]=="China"]
recovered_china=np.sum(recovered_china["4/10/20"])

recovered_italy=covid_19_recovered_global[covid_19_recovered_global["Country/Region"]=="Italy"]
recovered_italy=np.sum(recovered_italy["4/10/20"])
recovered_italy
confirmed_china=covid_19_confirmed_global[covid_19_confirmed_global["Country/Region"]=="China"]
confirmed_china=np.sum(confirmed_china["4/10/20"])

confirmed_italy=covid_19_confirmed_global[covid_19_confirmed_global["Country/Region"]=="Italy"]
confirmed_italy=np.sum(confirmed_italy["4/10/20"])

df_china_italy=pd.DataFrame([[confirmed_china,confirmed_italy],[recovered_china,recovered_italy],
                       [death_china,death_italy]],
                      index=["Confirmed","Recovered","Death"],columns=["China","Italy"])
df_china_italy
df_china_italy.plot.pie( subplots=True ,figsize=(20,20),autopct='%1.2f%%');
fig = px.treemap(covid_19_confirmed_global, path=['Country/Region'], values='4/10/20',
                title="CoVid-19 Confirmed cases")
fig.show()
fig = px.treemap(covid_19_deaths_global, path=['Country/Region'], values='4/10/20',
                title="CoVid-19 deaths cases ")
fig.show()
fig = px.treemap(covid_19_recovered_global, path=['Country/Region'], values='4/10/20',
                title="CoVid-19 recovered cases ")
fig.show()