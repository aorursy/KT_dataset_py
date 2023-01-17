#This Notebook aims to explain the analysis of coronavirus spread in the world and across majorly affected countries using python. Seaborn and Matplotlib libraries are used to get beautiful visualizations. One can expect to learn how to code data manipulation and visualization from this file. 
#Dataset used here is available on GitHub, Filename: "countries-aggregated.csv". Data here gets updated on a daily basis. 
#This Dataset contains the record of 188 countries in terms of confirmed cases, recovered cases, and death cases from 22nd January 2020 to 26th May 2020.
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker
%matplotlib inline
import numpy as np
import seaborn as sns
df=pd.read_csv("https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv") # Importing directly from the url
df.tail(20)  # returns last 20 records.


df["Country"].nunique() # checking the no. of countries.
df.groupby("Date")["Confirmed"].sum()
# To get the Total no. of confirmed cases grouped by date.
plt.figure(figsize=(10,5))
plt.plot(df.groupby("Date")["Confirmed"].sum(),color="Red")
plt.xticks(["2020-01-26", "2020-02-26","2020-03-26","2020-04-26" ,"2020-05-26",],rotation=60,
           horizontalalignment='right',fontsize='15')
plt.title("Total confirmed cases trend in the world")

plt.xlabel("Date")
plt.ylabel("Confirmed(amoount. in 10 lakhs)")
plt.show()
df.groupby("Date")["Recovered"].sum()
# To get the recovered cases grouped by date.
plt.figure(figsize=(10,7))
plt.plot(df.groupby("Date")["Recovered"].sum(),color="Black")
plt.xlabel("Date")
plt.ylabel("Recovered (amount in 10 lakhs)")
plt.title("Total Recovered cases trend in the world")
plt.xticks(["2020-01-26", "2020-02-26","2020-03-26","2020-04-26" ,"2020-05-26",],rotation=45,
           horizontalalignment='right',fontsize='15')
plt.show() 
df.groupby("Date")["Deaths"].sum()
# To get the death cases grouped by date.
plt.figure(figsize=(10,5)) 
plt.plot(df.groupby("Date")["Deaths"].sum(),color="Green")
plt.xlabel("Date")
plt.ylabel("Deaths")
plt.xticks(["2020-01-26", "2020-02-26","2020-03-26","2020-04-26" ,"2020-05-26",],rotation=45,
           horizontalalignment='right',fontsize='15')
plt.show()

df1=df.query("Date=='2020-05-26'")  # to get the latest data available for the 26th may 2020.
df1
df1["Confirmed"].max() 
print("Maximum no. of confirmed cases for a country is {}".format(df1["Confirmed"].max()))
df1["Recovered"].max() 
print("Maximum no. of cases of Recovered cases for a country is {}".format(df1["Recovered"].max()))
df1["Deaths"].max() 
print("Maximum no. of death cases for a country is {}".format(df1["Deaths"].max()))
df_sub= df1[df1['Confirmed'] > 50000 ]
df_sub # to get records for those countries having more than 50K cases as on 26th may.
df_sub.count() # 19 countries out of 188 have more than 50k confirmed cases.
df_sorted1=df1.sort_values(by=["Confirmed"], ascending=False)
df_sorted1.head(10) # get the top 10 records after sorting confirmed cases in descending order
df_sorted1.iloc[0:10,1:3] # get only top 10 countries with highest no. of confirmed cases and excluding other columns. 
df1.sort_values(by=["Recovered"], ascending=False) 
# Not storing this into different data frame as we dont need it later for the vsiualization part.

df1.sort_values(by=["Recovered"], ascending=False).iloc[0:10,1:4:2] 
# get only top 10 countries with highest no. of recovered cases. 
df1.sort_values(by=["Deaths"], ascending=False)

df1.sort_values(by=["Deaths"], ascending=False).iloc[0:10,1:5:3] #get top 10 countries with highest no. deaths
# recovery rate= Recovered/Confirmed.
df_sorted1["Recovery rate"]=df_sorted1["Recovered"]/df_sorted1["Confirmed"]
df_sorted1["Death rate"]=df_sorted1["Deaths"]/df_sorted1["Confirmed"]
df_top10confirmed=df_sorted1.head(10) # recovery rates and death rates of top 10 countries with highest coronavirus confirmed cases. 
df_top10confirmed
dff1=df_top10confirmed.sort_values(by=["Death rate"],ascending=False)
print(dff1.iloc[:,1:7:5])# to  get the countries and their respective death rates only
dff2=df_top10confirmed.sort_values(by=["Recovery rate"],ascending=False)
print(dff2.iloc[:,1:6:4])#to  get the countries and their respective recovery rates only
plt.figure(figsize=(12,10))
sns.barplot(x="Country",y="Confirmed",data=df_top10confirmed)
plt.xlabel("Country")
plt.ylabel("Confirmed (in 10 lakhs)")
plt.title("No. of Confirmed cases of 10 most affected countries")
plt.show()
plt.savefig("CORONO.png")
#SCATTERPLOT
plt.figure(figsize=(12,10))
sns.stripplot(x="Country",y="Confirmed",data=df_top10confirmed,size=12)
plt.xlabel("Country")
plt.ylabel("Confirmed (in 10 lakhs)")
plt.title("Scatter plot Confirmed cases of 10 most affected countries")

plt.figure(figsize=(12,10))
sns.barplot(x="Country",y="Recovered",data=df_top10confirmed)
plt.xlabel("Country")
plt.ylabel("Recovered")
plt.title("No. of Recovered cases of 10 most affected countries")
plt.figure(figsize=(12,10))
sns.barplot(x="Country",y="Deaths",data=df_top10confirmed)
plt.xlabel("Country")
plt.ylabel("Deaths")
plt.title("No. of  Death cases of most affected countries")
plt.figure(figsize=(12,10))
sns.barplot(x="Country",y="Death rate",data=df_top10confirmed)
plt.xlabel("Country")
plt.ylabel("Death rate")
plt.title("Death rate")
plt.figure(figsize=(12,10))
sns.stripplot(x="Country",y="Death rate",data=df_top10confirmed,size=12)
plt.title("Death rate Scatter plot")
plt.figure(figsize=(12,10))
sns.barplot(x="Country",y="Recovery rate",data=df_top10confirmed)
plt.xlabel("Country")
plt.ylabel("Recovery rate")
plt.title("Recovery rate bar chart")
plt.figure(figsize=(12,10))
sns.stripplot(x="Country",y="Recovery rate",data=df_top10confirmed,size=12)
plt.title("Recovery rate Scatter plot")

countries = [ 'United Kingdom', 'US', 'France', "Germany", "Brazil","Spain","Italy","Russia","Turkey","India"] # LIST OF MOST AFFECTED COUNTRIES MAINLY IN TERMS OF NO. OF CONFIRMED CASES.


df_main_countries = df[df['Country'].isin(countries)]
df_main_countries
df_main_countries["Country"].nunique() # to recheck that we get data for only 10 countries as required.
countries_1 = [ 'United Kingdom', 'US', 'France', "Germany","Russia"]   # getting only 5 countries at a time to use it to make the graph more transparent and clearer.
plt.figure(figsize=(11,9))

sns.set_style("darkgrid")
sns.lineplot(x="Date", y="Confirmed", hue="Country", data=df_main_countries[df_main_countries['Country'].isin(countries_1)])
plt.xticks(["2020-01-22", "2020-02-22","2020-03-22","2020-04-22" ,"2020-05-22",],rotation=45, 
    horizontalalignment='right',
    fontsize='15')
plt.title("Confirmed cases trend since 22nd jan")
plt.show()
countries_2=["Brazil","Spain","Italy","India","Turkey"]

import matplotlib.dates as md
plt.figure(figsize=(11,9)) 
sns.lineplot(x="Date", y="Confirmed", hue="Country", data=df_main_countries[df_main_countries['Country'].isin(countries_2)])

plt.xticks(["2020-01-22", "2020-02-22","2020-03-22","2020-04-22" ,"2020-05-22",],rotation=45,
           horizontalalignment='right',fontsize='15')
plt.title("Confirmed cases trend since 22nd jan")
plt.show()

countries_1 = [ 'United Kingdom', 'US', 'France', "Germany","Russia"]   # getting only 5 countries at a time to use it to make the graph more transparent and clearer.
plt.figure(figsize=(11,9))

sns.set_style("darkgrid")
sns.lineplot(x="Date", y="Recovered", hue="Country", data=df_main_countries[df_main_countries['Country'].isin(countries_1)])
plt.xticks(["2020-01-22", "2020-02-22","2020-03-22","2020-04-22" ,"2020-05-22",],rotation=45, 
    horizontalalignment='right',
    fontsize='15')


plt.title("Recovered cases trend since 22nd jan")
plt.ylabel(" No. of Recovered cases ")
plt.show()
countries_2=["Brazil","Spain","Italy","India","Turkey"]

import matplotlib.dates as md
plt.figure(figsize=(11,9)) 
sns.lineplot(x="Date", y="Recovered", hue="Country", data=df_main_countries[df_main_countries['Country'].isin(countries_2)])

plt.xticks(["2020-01-22", "2020-02-22","2020-03-22","2020-04-22" ,"2020-05-22",],rotation=45,
           horizontalalignment='right',fontsize='15')
plt.title("Recovered cases trend since 22nd jan")
plt.show()

countries_1 = [ 'United Kingdom', 'US', 'France', "Germany","Russia"]   # getting only 5 countries at a time to use it to make the graph more transparent and clearer.
plt.figure(figsize=(11,9))

sns.set_style("darkgrid")
sns.lineplot(x="Date", y="Deaths", hue="Country", data=df_main_countries[df_main_countries['Country'].isin(countries_1)])
plt.xticks(["2020-01-22", "2020-02-22","2020-03-22","2020-04-22" ,"2020-05-22",],rotation=45, 
    horizontalalignment='right',
    fontsize='15')
plt.title("Death cases trend since 22nd jan")
plt.show()
countries_2=["Brazil","Spain","Italy","India","Turkey"]

import matplotlib.dates as md
plt.figure(figsize=(11,9)) 
sns.lineplot(x="Date", y="Deaths", hue="Country", data=df_main_countries[df_main_countries['Country'].isin(countries_2)])

plt.xticks(["2020-01-22", "2020-02-22","2020-03-22","2020-04-22" ,"2020-05-22",],rotation=45,
           horizontalalignment='right',fontsize='15')
plt.title("Death cases trend since 22nd jan")
plt.show()


