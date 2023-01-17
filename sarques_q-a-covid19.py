import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns
BASE_PATH = "/kaggle/input/corona-virus-report/"

data_complete = pd.read_csv("{}covid_19_clean_complete.csv".format(BASE_PATH))

data_usa = pd.read_csv("{}usa_county_wise.csv".format(BASE_PATH))

data_test = pd.read_csv("{}country_wise_latest.csv".format(BASE_PATH))

data_world = pd.read_csv("{}worldometer_data.csv".format(BASE_PATH))

data_gp = pd.read_csv("{}full_grouped.csv".format(BASE_PATH))

data_gdp = pd.read_csv("/kaggle/input/uscountiesgdp/us_counties_gdp.csv")

data_covid_usa = pd.read_csv("/kaggle/input/covid-usa/covid_usa.csv")

data_continent = pd.read_csv("/kaggle/input/conticountry/conti_country.csv")
data_complete.head(3)
data_usa.head(3)
data_test.head(3)
data_gdp.head(3)
data_covid_usa.head(3)
data_continent.head(3)
data_gp.head(3)
data_world.head(3)
temp = data_world.sort_values(["Deaths/1M pop", "Tot Cases/1M pop", "Tests/1M pop"], ascending = False).iloc[:15]
sns.set(style = "darkgrid")

plt.rcParams["figure.figsize"] = [20, 15]

temp.plot(x = "Country/Region", y = ["Deaths/1M pop", "Tot Cases/1M pop", "Tests/1M pop"], kind = "bar")

plt.xlabel("Country")

plt.ylabel("Ratio")

plt.title("Deaths/Cases/Tests per 1M population in different countries")
temp = data_world.sort_values(["Deaths/1M pop", "Tot Cases/1M pop", "Tests/1M pop"], ascending = True).iloc[:15]
temp.plot(x = "Country/Region", y = ["Deaths/1M pop", "Tot Cases/1M pop", "Tests/1M pop"], kind = "bar")

plt.xlabel("Country")

plt.ylabel("Ratio")

plt.title("Deaths/Cases/Tests per 1M population in different countries")
temp = data_world[:10]

plt.plot(temp["Country/Region"], temp["Deaths/1M pop"]/temp["Tot Cases/1M pop"], c = "Red")

plt.ylabel("Death rate per case")

plt.xlabel("Top 10 countries with most number of tests")

plt.title("Death rate in countries with most number of tests conducted")
temp = data_complete.sort_values(["Date"], ascending = False)

temp = temp[temp["Date"] == temp.Date[len(temp)-1]]
print("Global death rate stands at about {}%".format((temp.Deaths.sum()/temp.Confirmed.sum())*100))

print("Global recovery rate stands at about {}%".format((temp.Recovered.sum()/temp.Confirmed.sum())*100))
temp = data_world.copy()

temp.columns = ['Country', 'Continent', 'Population', 'TotalCases', 'NewCases',

       'TotalDeaths', 'NewDeaths', 'TotalRecovered', 'NewRecovered',

       'ActiveCases', 'Serious,Critical', 'Tot Cases/1M pop', 'Deaths/1M pop',

       'TotalTests', 'Tests/1M pop', 'WHO Region']
temp = temp[["Country", "Deaths/1M pop", "Continent"]].groupby(["Continent"])["Deaths/1M pop"].mean()
plt.plot(temp)

plt.xlabel("Continents")

plt.ylabel("Deaths per 1M population")

plt.title("Deaths per 1M pop in different continents")
data_gdp.columns = ["Rank", "County", "Contribution", "%age_of_total_contribution", "gdp_per_capita", "Region"]

temp = data_gdp.sort_values(["Contribution"], ascending = False)[1:11]
plt.rcParams["figure.figsize"] = [20, 10]

ax = sns.barplot(x = "County", y = "Contribution", data = temp)

plt.title("Top 10 contributors to the USA's GDP")
temp = data_covid_usa.sort_values(["Deaths", "Confirmed"], ascending = False)[:10]
sns.barplot(x = "Province_State", y = "Deaths", data = temp)

plt.title("Top 10 worst affected counties of USA")
temp = pd.merge(data_covid_usa, data_gdp, how = "left", left_on = "Province_State", right_on = "County")

temp = temp.sort_values(["Deaths"], ascending = False)[["County", "Deaths", "Region"]].groupby(["Region"])["Deaths"].sum()



plt.plot(temp)

plt.xlabel("Region")

plt.ylabel("Deaths")

plt.title("Number of deaths in different regions of the USA")
data_gdp[data_gdp["Region"] == "Northeast"]