# Load the dataset

xls_dataset = "/kaggle/input/COVID-19-geographic-disbtribution-worldwide.xlsx"

import pandas as pd

import numpy as np



covid_dataset = pd.read_excel(xls_dataset)
# create two new columns:

# - deaths per 100_000 population

# - deaths per 100 cases

# (note: population data updated in 2018)

covid_dataset["deaths_per_100k"] = covid_dataset["deaths"] / covid_dataset["popData2018"] * 100_000

covid_dataset["deaths_per_100cases"] = covid_dataset["deaths"] / covid_dataset["cases"] * 100



# useful snippets:

# print data for a specific date

# print(covid_dataset[covid_dataset["dateRep"] == "2020-04-17"][["deaths", "countriesAndTerritories"]])



# print data for a range of dates and a list of countries

# select_dates = covid_dataset["dateRep"].isin(["2020-04-17", "2020-04-16"])

# select_countries = covid_dataset["countriesAndTerritories"].isin(["Italy", "Germany"])

# print(covid_dataset[select_dates & select_countries][["dateRep", "countriesAndTerritories", "deaths", "deaths_per_100k"]].sort_values(["countriesAndTerritories", "dateRep"], ascending=[True, True]))



# print data for the UK

# print(covid_dataset[covid_dataset["countriesAndTerritories"] == "United_Kingdom"].head()

#

# print data for the UK after certain date

# country_uk = covid_dataset["countriesAndTerritories"] == "United_Kingdom"

# after = covid_dataset["dateRep"] > "2020-03-01"

# print(covid_dataset[country_uk & after].sort_values("dateRep", ascending=True).head())



# plot a simple graph

country_uk = covid_dataset["countriesAndTerritories"] == "United_Kingdom"

from_01_04 = covid_dataset["dateRep"] >= "2020-04-01"

covid_dataset[country_uk & from_01_04].sort_values("dateRep", ascending=True).plot(kind="bar", y=["deaths", "cases"], x="dateRep")

covid_dataset[country_uk & from_01_04].sort_values("dateRep", ascending=True).plot(y="deaths_per_100cases", x="dateRep")
import matplotlib.pyplot as plt

import seaborn as sns



sns.set()



# plot_this contains the data that we want to plot:

# deaths, cases, deaths_per_100k, deaths_per_100cases in a single country ("United_Kingdom")

# after March 1st 2020



country_uk = covid_dataset["countriesAndTerritories"] == "United_Kingdom"

from_01_03 = covid_dataset["dateRep"] >= "2020-03-01"

data_uk_march_01 = covid_dataset[country_uk & from_01_03]

bins = int(np.sqrt(len(data_uk_march_01)))

plot_this = data_uk_march_01[["dateRep", "cases", "deaths", "deaths_per_100k", "deaths_per_100cases"]].sort_values("dateRep")



# very simple graph of deaths per 100k population using seaborn



_ = plt.plot(plot_this["dateRep"], plot_this["deaths_per_100k"])

_ = plt.ylabel("deaths per 100k population")

_ = plt.xlabel("date")

_ = plt.xticks(rotation=45)

plt.show()
# create a graph title which also contains the date of last updated



from datetime import date

_today = date.today().isoformat()

graph_title = "UK - COVID-19 Deaths after March 1st 2020 - Updated: " + _today
# create a graph using seaborn which display separately deaths per 100k and deaths per 100 cases



fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(14, 10))

fig.suptitle(graph_title)



ax1.set_title("deaths_per_100k")

ax1.plot(plot_this["dateRep"], plot_this["deaths_per_100k"])



ax2.set_title("deaths_per_100cases")

ax2.plot(plot_this["dateRep"], plot_this["deaths_per_100cases"])



plt.xlabel("date")

plt.xticks(rotation=45)

plt.show()
# plot deaths and cases in a single graph using seaborn



plt.figure(figsize=(14, 10))

g1 = sns.lineplot(x="dateRep", y="deaths", data=plot_this, palette="blue", label="deaths")

g2 = sns.lineplot(x="dateRep", y="cases", data=plot_this, palette="orange", label="cases")

plt.ylabel("")

plt.xlabel("")

plt.suptitle(graph_title)

plt.xticks(rotation=45)

plt.show()
# total number of deaths for COVID-19 in the UK

print(plot_this.tail())



print("\n\nCumulative sum of the number of deaths")

plot_this["total_deaths"] = plot_this["deaths"].cumsum()

print(plot_this[["dateRep", "deaths", "total_deaths"]].tail())

pivot_df = covid_dataset.pivot_table(index="dateRep", values=["deaths_per_100k", "deaths_per_100cases"],columns="countriesAndTerritories")

print(pivot_df[("deaths_per_100cases", "United_Kingdom")])

# pivot_df.loc[lambda x: x > "2020-04"]

pivot_df.loc["2020-04":].head()