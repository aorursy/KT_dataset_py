## COVID-19 Data Visualization Study



## Study done by Özgür Can Arıcan

## Accessed to the data set from "EU Open Data Portal"

## link to the data set: https://data.europa.eu/euodp/en/data/dataset/covid-19-coronavirus-data/resource/55e8f966-d5c8-438e-85bc-c7a5a26f4863
import pandas as pd

import matplotlib.pyplot as plt



#imported the required libraries
data = pd.read_excel("https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide.xlsx")

data.head(10)
data.shape
#For cleaning the dataframe, rows that includes NA values are erased.



data.dropna(inplace = True)

data.shape
#Get the worldwide total cases



data["cases"].sum()
#Get the worldwide total deaths



data["deaths"].sum()
#Demonstrate the total cases and deaths by countries in horizontal bar plot



byCountrySum = data.groupby(["countriesAndTerritories"]).sum()[["cases", "deaths"]]

byCountrySum.plot.barh(figsize = (15, 500))
#Demonstrate the top 5 countries that have the most cases in bar plot



mostFiveCase = data.groupby(["countriesAndTerritories"]).sum()["cases"].sort_values(ascending = False).head(5)

mostFiveCase.plot.bar()
#Demonstrate the top 5 countries that have the least cases in bar plot



leastFiveCase = data.groupby(["countriesAndTerritories"]).sum()["cases"].sort_values(ascending = True).head(5)

leastFiveCase.plot.bar()
#Demonstrate the top 5 countries that have the most deaths in bar plot



mostFiveDeaths = data.groupby(["countriesAndTerritories"]).sum()["deaths"].sort_values(ascending = False).head(5)

mostFiveDeaths.plot.bar()
#Demonstrate the continental cases and deaths by pie charts



continentData = data.groupby(["continentExp"]).sum()[["cases", "deaths"]]

continentData.plot.pie(subplots = True, figsize = (10, 10))



#As it is shown in pie charts;

#There is no any significant changes in death and case ratios of America, Africa and Oceania

#Altough the deaths ratio is much greater than the cases ratio in Europe

#And the deaths ratio is much lower than the cases ratio in Asia
#Demonstrate the daily cases and deaths in Turkey in line graph



TurkeyData = data[data["countriesAndTerritories"] == "Turkey"].sort_values(by = "dateRep").set_index("dateRep")[["cases", "deaths"]]

TurkeyData.plot.line(figsize = (12, 7))
#Get the Turkey total cases



TurkeyData["cases"].sum()
#Get the Turkey total deaths

TurkeyData["deaths"].sum()