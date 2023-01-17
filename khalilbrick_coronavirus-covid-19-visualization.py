import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import datetime as dt

%matplotlib inline
country_wise_latest = pd.read_csv("../input/corona-virus-report/country_wise_latest.csv")

country_wise_latest = country_wise_latest.set_index("Country/Region")

country_wise_latest_Confirmed_Deaths_Recovered = country_wise_latest[["Confirmed","Recovered","Deaths",]]

country_wise_latest_Confirmed_Deaths_Recovered = country_wise_latest_Confirmed_Deaths_Recovered.rename(index = {"United Kingdom":"UK"})





country_wise_latest_Confirmed_Deaths_Recovered_top_10_by_Deaths = country_wise_latest_Confirmed_Deaths_Recovered.sort_values(by = "Deaths")[-1:-11:-1]

country_wise_latest_Confirmed_Deaths_Recovered_top_10_by_confirmed = country_wise_latest_Confirmed_Deaths_Recovered.sort_values(by = "Confirmed")[-1:-11:-1]

country_wise_latest_Confirmed_Deaths_Recovered_top_10_by_recovered = country_wise_latest_Confirmed_Deaths_Recovered.sort_values(by = "Recovered")[-1:-11:-1]

country_wise_latest_Confirmed_Deaths_Recovered_top_10_by_recovered
country_wise_latest_Confirmed_Deaths_Recovered_top_10_by_confirmed["Confirmed"][-1:-11:-1].plot(kind = "barh",figsize = (15,8))

plt.xticks(size = 15)

plt.yticks(size = 15)

plt.ylabel(None)





plt.legend(fontsize = 15,ncol = 3)

plt.title("Top 10 Countries with the most confirmed cases in the World",fontdict = {"fontsize":25})

plt.savefig("Top 10 Countries with the most confirmed cases in the World",dpi = 300)

plt.show()
country_wise_latest_Confirmed_Deaths_Recovered_top_10_by_Deaths["Deaths"][-1:-11:-1].plot(kind = "barh",figsize = (15,8),color = "red")

plt.xticks(size = 15)

plt.yticks(size = 15)

plt.ylabel(None)





plt.legend(fontsize = 15,ncol = 3)

plt.title("Top 10 Countries with the most Deaths in the World",fontdict = {"fontsize":25})

plt.savefig("Top 10 Countries with the most Deaths in the World",dpi = 300)



plt.show()
country_wise_latest_Confirmed_Deaths_Recovered_top_10_by_recovered["Recovered"][-1:-11:-1].plot(kind = "barh",figsize = (15,8),color = "green")

plt.xticks(size = 15)

plt.yticks(size = 15)

plt.ylabel(None)





plt.legend(fontsize = 15,ncol = 3)

plt.title("Top 10 Countries with the most Recovered in the World",fontdict = {"fontsize":25})

plt.savefig("Top 10 Countries with the most Recovered in the World",dpi = 300)



plt.show()
country_wise_latest_active = country_wise_latest.Active.sort_values()[-1:-11:-1]

country_wise_latest_active = country_wise_latest_active.rename(index = {"United Kingdom":"UK"})

country_wise_latest_active
country_wise_latest_active[-1:-11:-1].plot(kind = "barh",figsize = (15,8),color = "purple")

plt.xticks(size = 15)

plt.yticks(size = 15)

plt.ylabel(None)





plt.legend(fontsize = 15,ncol = 3)

plt.title("Top 10 Countries with the most Active Cases in the World",fontdict = {"fontsize":25})

plt.savefig("Top 10 Countries with the most Active Cases in the World",dpi = 300)



plt.show()
day_wise = pd.read_csv("../input/corona-virus-report/day_wise.csv")

day_wise = day_wise.set_index("Date")

day_wise
plt.figure(figsize = (15,8))

plt.plot(day_wise.Confirmed,label = "Confirmed Cases")

plt.plot(day_wise.Deaths,label = "Confirmed Deaths")

plt.plot(day_wise.Recovered,label = "Confirmed Recovered")

plt.xticks(['2020-01-22','2020-02-22','2020-03-22','2020-04-22','2020-05-22','2020-06-22'],size = 15)

plt.yticks([i for i in range(0,10000000,1000000)],size = 15)

plt.ylabel("Number of Cases",size = 15)





plt.legend(fontsize = 15,ncol = 3)

plt.title("Covid-19 in the World from (2020-01-22) to (2020-06-30)",fontdict = {"fontsize":25})

plt.savefig("Covid-19 in the World from 2020-01-22 to 2020-06-30",dpi = 300)



plt.show()
plt.figure(figsize = (15,8))

plt.plot(day_wise["No. of countries"],label = "Number of Countries infected",marker = "_")



plt.xticks(['2020-01-22','2020-02-22','2020-03-22','2020-04-22','2020-05-22','2020-06-22'],size = 15)

plt.yticks(size = 15)

plt.ylabel("Number of Countries",size = 15)





plt.legend(fontsize = 15,ncol = 3)

plt.title("Covid-19 Number of Countries infected from (2020-01-22) to (2020-06-30)",fontdict = {"fontsize":20})

plt.savefig("Covid-19 Number of Countries infected from 2020-01-22 to 2020-06-30",dpi = 300)

plt.show()
plt.figure(figsize = (15,8))

plt.scatter(day_wise["Deaths / 100 Cases"].index,day_wise["Deaths / 100 Cases"].values,label = "Covid-19 Mortality")

plt.xticks(['2020-01-22','2020-02-22','2020-03-22','2020-04-22','2020-05-22','2020-06-22'],size = 15)

plt.yticks(size = 15)

plt.ylabel("Covid-19 Mortality %",size = 15)





plt.legend(fontsize = 15,ncol = 3)

plt.title("Covid-19 Mortality in The World from (2020-01-22) to (2020-06-30)",fontdict = {"fontsize":20})

plt.savefig("Covid-19 Mortality in The World from 2020-01-22 to 2020-06-30",dpi = 300)

plt.show()
plt.figure(figsize = (15,8))

plt.plot(day_wise["New cases"].index,day_wise["New cases"].values)

plt.scatter(day_wise["New cases"].index,day_wise["New cases"].values,label = "New cases")



plt.plot(day_wise["New deaths"].index,day_wise["New deaths"].values,label = "New deaths")





plt.xticks(['2020-01-22','2020-02-22','2020-03-22','2020-04-22','2020-05-22','2020-06-22'],size = 15)

plt.yticks(size = 15)

plt.ylabel("New cases/deaths",size = 20)





plt.legend(fontsize = 15,ncol = 3)

plt.title(" Covid-19 New cases_Deaths in The World (2020-01-22) to (2020-06-30)",fontdict = {"fontsize":20})

plt.savefig("Covid-19 New cases_Deaths in The World 2020-01-22) to 2020-06-30",dpi = 300)

plt.show()
covid_19_clean_complete = pd.read_csv("../input/covid19-data/owid-covid-data.csv")

covid_19_clean_complete = covid_19_clean_complete.set_index(["location"])

covid_19_clean_complete = covid_19_clean_complete.rename(index = {"United States":"US","United Kingdom":"UK"})

covid_19_clean_complete_top_10_confiremed = covid_19_clean_complete.loc[country_wise_latest_Confirmed_Deaths_Recovered_top_10_by_Deaths.index]

covid_19_clean_complete_top_10_Deaths = covid_19_clean_complete.loc[country_wise_latest_Confirmed_Deaths_Recovered_top_10_by_Deaths.index]
plt.figure(figsize = (15,8))

plt.scatter(day_wise.Confirmed.index,day_wise.Confirmed.values,label = "The World")



for country in country_wise_latest_Confirmed_Deaths_Recovered_top_10_by_Deaths.index:   

    plt.plot(covid_19_clean_complete_top_10_confiremed.loc[country]["date"],covid_19_clean_complete_top_10_confiremed.loc[country]["total_cases"],label = country)





plt.ylabel("Number of Cases",size = 15)



plt.xticks(['2020-01-22','2020-02-22','2020-03-22','2020-04-22','2020-05-22','2020-06-22',"2020-07-02"],size = 15)

plt.yticks([i for i in range(0,10000000,1000000)],size = 15)

plt.title("Covid-19 Top 10 Countries Confirmed Cases Every Day from (2019-12-31) to (2020-07-02)",fontdict = {"fontsize":18})

plt.legend(fontsize = 15,ncol = 2)

plt.savefig("Covid-19 Top 10 Countries Confirmed Cases Every Day from 2019-12-31 to 2020-07-02",dpi = 300)

plt.show()
plt.figure(figsize = (15,8))

plt.scatter(day_wise.Deaths.index,day_wise.Deaths.values,label = "The World")



for country in country_wise_latest_Confirmed_Deaths_Recovered_top_10_by_Deaths.index:   

    plt.plot(covid_19_clean_complete_top_10_Deaths.loc[country]["date"],covid_19_clean_complete_top_10_Deaths.loc[country]["total_deaths"],label = country)





plt.ylabel("Number of Deaths",size = 15)



plt.xticks(['2020-01-22','2020-02-22','2020-03-22','2020-04-22','2020-05-22','2020-06-22',"2020-07-02"],size = 15)

plt.yticks([i for i in range(0,800000,100000)],size = 15)

plt.title("Covid-19 Top 10 Countries  Deaths Every Day from (2019-12-31) to (2020-07-02)",fontdict = {"fontsize":18})

plt.savefig("Covid-19 Top 10 Countries  Deaths Every Day from 2019-12-31 to 2020-07-02",dpi = 300)

plt.legend(fontsize = 15,ncol = 2)

plt.show()
covid_19_clean_complete_cont_top_cases = (covid_19_clean_complete.drop(["International","World"],axis = 0)).groupby("continent").sum().sort_values(by = "new_cases")[-1::-1]
plt.figure(figsize = (15,8))



for cont in covid_19_clean_complete_cont_top_cases.index:   

    covid_19_clean_complete_cont_top_cases["new_cases"].plot(kind = "bar",label = cont)



plt.plot(covid_19_clean_complete_cont_top_cases.new_cases.index,covid_19_clean_complete_cont_top_cases.new_cases.values,color = "black",marker = "o")



plt.ylabel("Number of Cases",size = 15)

plt.xlabel(None)

plt.xticks(rotation = 20,size = 15)

plt.title("Covid-19 Top Continent Confirmed Cases",fontdict = {"fontsize":18})

plt.savefig("Covid-19 Top Continent Confirmed Cases",dpi = 300)

plt.show()
covid_19_clean_complete_cont_top_deaths = (covid_19_clean_complete.drop(["International","World"],axis = 0)).groupby("continent").sum().sort_values(by = "new_deaths")[-1::-1]
plt.figure(figsize = (15,8))



for cont in covid_19_clean_complete_cont_top_deaths.index:   

    covid_19_clean_complete_cont_top_deaths["new_deaths"].plot(kind = "bar",label = cont,color = "red")



plt.plot(covid_19_clean_complete_cont_top_deaths.new_deaths.index,covid_19_clean_complete_cont_top_deaths.new_deaths.values,color = "black",marker = "o")



plt.ylabel("Number of deaths",size = 15)

plt.xlabel(None)

plt.xticks(rotation = 20,size = 15)



plt.title("Covid-19 Top Continent deaths",fontdict = {"fontsize":18})

plt.savefig("Covid-19 Top Continent deaths Cases",dpi = 300)



plt.show()