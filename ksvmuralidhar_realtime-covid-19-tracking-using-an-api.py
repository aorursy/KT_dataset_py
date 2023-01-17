import requests

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

import datetime

import plotly.express as px

import plotly.graph_objects as go

%matplotlib inline

pd.set_option("display.max_rows",None)
req = requests.get("https://api.covid19api.com/summary")

req.status_code

js = req.json()





country_data = pd.DataFrame()

for i in range(len(js["Countries"])):

    js["Countries"][i].pop("Premium")

    country_data = country_data.append(pd.DataFrame(js["Countries"][i],index=[i])).copy()



country_data_main = country_data.copy()

country_data.head()
as_on = js["Countries"][0]["Date"]

as_on = datetime.datetime.strptime(as_on,"%Y-%m-%dT%H:%M:%SZ")

as_on = datetime.datetime.strftime(as_on,"%d %b %Y")

print("Last updated on: "+ str(as_on))
iso_2_3 = pd.read_csv("https://gist.githubusercontent.com/tadast/8827699/raw/3cd639fa34eec5067080a61c69e3ae25e3076abb/countries_codes_and_coordinates.csv")

iso_2_3 = iso_2_3[["Alpha-2 code","Alpha-3 code"]].copy()

iso_2_3.columns = ["CountryCode","iso3"]



iso_2_3["CountryCode"] = iso_2_3["CountryCode"].str.replace('"',"").str.strip()

iso_2_3["iso3"] = iso_2_3["iso3"].str.replace('"',"").str.strip()



country_data = country_data.merge(iso_2_3,on="CountryCode",how="left").copy()

px.choropleth(country_data,locations="iso3",color="TotalConfirmed",hover_name="Country",color_continuous_scale=px.colors.sequential.Reds)

country_data.drop(columns="iso3",inplace=True)
sns.set_style("darkgrid")

country_data = country_data.sort_values(by="TotalConfirmed",ascending=False)

country_data = country_data.reset_index(drop=True)





fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,20))

top_20 = country_data.iloc[:20,:].copy()

top_cat_plot_df = top_20.loc[:,["Country","TotalConfirmed","TotalDeaths","TotalRecovered"]].melt(id_vars="Country").copy()

sns.stripplot(x="value",y="Country",hue="variable",data=top_cat_plot_df,alpha=0.6,sizes=[160,160],ax=ax1,jitter=False)

ax1.set_title("Top 20 countries by confirmed cases as on "+str(as_on))

ax1.set_xlabel("Total confirmed cases / Total deaths / Total recovered")







bot_20 = country_data.iloc[-20:,:].copy()

cat_plot_df = bot_20.loc[:,["Country","TotalConfirmed","TotalDeaths","TotalRecovered"]].melt(id_vars="Country").copy()

sns.stripplot(x="value",y="Country",hue="variable",data=cat_plot_df,alpha=0.6,sizes=[160,160],ax=ax2,jitter=False)

ax2.set_title("Bottom 20 countries by confirmed cases as on "+str(as_on))

ax2.set_xlabel("Total confirmed cases / Total deaths / Total recovered")

ax2.set_ylabel("")

plt.show()
country_data["Mortality_rate"] = np.round((country_data["TotalDeaths"] / country_data["TotalConfirmed"]) * 100,2)

country_data["Recovery_rate"] = np.round((country_data["TotalRecovered"] / country_data["TotalConfirmed"]) * 100,2)

country_data = country_data.sort_values(by="Mortality_rate",ascending=False)

country_data = country_data.reset_index(drop=True)





fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,22))

top_20 = country_data.iloc[:20,:].copy()

top_cat_plot_df = top_20.loc[:,["Country","Recovery_rate","Mortality_rate"]].melt(id_vars="Country").copy()

#sns.stripplot(x="value",y="Country",hue="variable",data=top_cat_plot_df,sizes=[160,160],ax=ax1)

sns.barplot(x="value",y="Country",hue="variable",data=top_cat_plot_df,ax=ax1)

ax1.set_title("Top 20 countries by mortality rate as on "+str(as_on))

ax1.set_xlabel("Mortality rate / Recovery rate")

#ax1.legend().remove()





bot_20 = country_data.iloc[-20:,:].copy()

cat_plot_df = bot_20.loc[:,["Country","Recovery_rate","Mortality_rate"]].melt(id_vars="Country").copy()

#sns.stripplot(x="value",y="Country",hue="variable",data=cat_plot_df,sizes=[160,160],ax=ax2)

sns.barplot(x="value",y="Country",hue="variable",data=cat_plot_df,ax=ax2)

ax2.set_title("Bottom 20 countries by mortality rate as on "+str(as_on))

ax2.set_xlabel("Mortality rate / Recovery rate")

ax2.set_ylabel("")

plt.show()
country_data = country_data.sort_values(by="TotalConfirmed",ascending=False)

country_data = country_data.reset_index(drop=True)

sns.relplot(x="TotalConfirmed",y="TotalDeaths",data=country_data,s=100,alpha=0.5,height=6,aspect=1.5)

plt.title("Total confirmed cases vs Total deaths as on "+str(as_on))

for i in range(5):

    plt.text(x=country_data.loc[i,"TotalConfirmed"],y=country_data.loc[i,"TotalDeaths"],s=country_data.loc[i,"CountryCode"])

country_data = country_data.sort_values(by="TotalDeaths",ascending=False)

country_data = country_data.reset_index(drop=True)

for i in range(5):

    plt.text(x=country_data.loc[i,"TotalConfirmed"],y=country_data.loc[i,"TotalDeaths"],s=country_data.loc[i,"CountryCode"])
country_data_c = country_data.loc[country_data["CountryCode"].isin(["IN","US","MX","GB","BR"])==False,:].copy()

country_data_c = country_data_c.sort_values(by="TotalConfirmed",ascending=False)

country_data_c = country_data_c.reset_index(drop=True)

sns.relplot(x="TotalConfirmed",y="TotalDeaths",data=country_data_c,s=100,alpha=0.5,height=6,aspect=1.5)

plt.title("Total confirmed cases vs Total deaths (excluding visual outliers) as on "+str(as_on))

for i in range(10):

    plt.text(x=country_data_c.loc[i,"TotalConfirmed"],y=country_data_c.loc[i,"TotalDeaths"],s=country_data_c.loc[i,"CountryCode"])

country_data_c = country_data_c.sort_values(by="TotalDeaths",ascending=False)

country_data_c = country_data_c.reset_index(drop=True)

for i in range(10):

    plt.text(x=country_data_c.loc[i,"TotalConfirmed"],y=country_data_c.loc[i,"TotalDeaths"],s=country_data_c.loc[i,"CountryCode"])
#Cases, deaths and recoveries are all moving together, which is obvious

cor = country_data.corr()

mask = np.triu(cor)

fig = plt.figure(figsize=(10,10))

sns.heatmap(country_data.corr(),mask=mask,annot=True,fmt="0.1f",cbar=False)

plt.show()
def to_date(x):

    x = datetime.datetime.strptime(x,"%Y-%m-%dT%H:%M:%SZ")

    x = datetime.datetime.strftime(x,"%d %b %Y")

    return x





def country_plot(Country):

    res = requests.get("https://api.covid19api.com/total/country/" + Country)

    js = res.json()



    country_wise = pd.DataFrame()

    for i in range(len(js)):

        country_wise = country_wise.append(pd.DataFrame(js[i],index=[i]))

    country_wise["Date"] = country_wise["Date"].map(to_date)

    country_wise = country_wise.loc[:,["Confirmed","Deaths","Recovered","Active","Date"]].copy()

    country_wise = country_wise.set_index(country_wise["Date"])

    country_wise.drop(columns=["Date"],inplace=True)

    country_wise.plot(figsize=(17,6))

    plt.title(Country.upper())

    plt.ylabel("Total confirmed / Total recovered / Total active / Total deaths")

    plt.show()



country_plot("united-states")

country_plot("brazil")

country_plot("india")

country_plot("russia")

country_plot("peru")
country_plot("yemen")

country_plot("italy")

country_plot("united-kingdom")

country_plot("belgium")

country_plot("mexico")