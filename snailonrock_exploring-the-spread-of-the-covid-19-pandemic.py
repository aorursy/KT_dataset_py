%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import matplotlib

# Reading data
population_data = pd.read_csv("../input/global-population/population_by_country_2020.csv")
# 
population_data.head()
population_data.shape
# Renaming the column names to look more pythonic
population_data = population_data.rename(columns = {"Country (or dependency)": "country_dependency", 
                                                   "Population (2020)": "population_2020", "Yearly Change": "yearly_change", 
                                                   "Net Change": "net_change", "Density (P/Km²)": "density_p_km_sq", 
                                                   "Land Area (Km²)": "land_area_km_sq", "Migrants (net)": "migrants",
                                                   "Fert. Rate": "fert_rate", "Med. Age": "med_age", 
                                                    "Urban Pop %": "urban_pop_percent", "World Share": "world_share_percent",
                                                   })
population_data.dtypes
population_data.head()
population_data.describe().T
population_data[population_data.country_dependency == "China"]
population_data[population_data.country_dependency == "United States"]
covid_19_clean = pd.read_csv("../input/covid-19-clean-complete/covid_19_clean_complete.csv")
covid_19_clean.head()
covid_19_clean.columns
covid_19_clean.shape
covid_19_clean = covid_19_clean.rename(columns = {"Province/State": "province_state", "Country/Region": "country_region",
                                                 "Lat": "lat", "Long": "long", "Date": "date", "Confirmed": "confirmed",
                                                 "Deaths": "deaths", "Recovered": "recovered", "Active": "active",
                                                 "WHO Region": "who_region"})
covid_19_clean.head()
covid_19_clean.dtypes
covid_19_clean.describe().T
worldometer_data = pd.read_csv("../input/worldometer/datasets_494766_1402868_worldometer_data.csv")
worldometer_data.head()
worldometer_data.columns
worldometer_data.shape
worldometer_data = worldometer_data.rename(columns = {"Country/Region": "country_region", "Continent": "continent", 
                                                     "Population": "population", "TotalCases": "total_cases", 
                                                      "NewCases": "new_cases", "TotalDeaths": "total_deaths",
                                                      "NewDeaths": "new_deaths", "TotalRecovered": "total_recovered",
                                                     "NewRecovered": "new_recovered", "ActiveCases": "active_cases", 
                                                     "Serious,Critical": "serius_critical", 
                                                      "Tot Cases/1M pop": "top_cases_1m_pop", "Deaths/1M pop": "deaths_1m_pop",
                                                     "TotalTests": "total_tests", "Tests/1M pop": "tests_1m_pop", 
                                                     "WHO Region": "who_region"})
worldometer_data.head()
worldometer_data.dtypes
worldometer_data.describe().T
worldometer_data[worldometer_data.country_region == "USA"]
worldometer_data[worldometer_data.country_region == "China"]
worldometer_data = worldometer_data.append({"country_region": "China", "continent": "Asia","population": 1440070587,          
                                            "total_cases": 84895, "total_deaths": 4634, "total_recovered": 79745,
                                            "active_cases": 516, "total_tests": 90410000, 
                                            "who_region": "WesternPacific"}, ignore_index = True)
worldometer_data[worldometer_data.country_region == "China"]
covid_all_data = pd.read_csv("../input/covid19all/covid-19-all.csv")
covid_all_data.head()
covid_all_data.shape
covid_all_data = covid_all_data.rename(columns = {"Country/Region": "country_region", "Province/State": "province_state",
                                                 "Latitude": "latitude", "Longitude": "longitude", "Confirmed": "confirmed",
                                                 "Recovered": "recovered", "Deaths": "deaths", "Date": "date"})
covid_all_data.head()
covid_all_data.dtypes
covid_all_data
population_data.head()
# Renaming the United States and USA values to US for consistency
population_data = population_data.replace("United States", "US")
worldometer_data = worldometer_data.replace("USA", "US")
worldometer_data.head()
population_data.head()
population_data.dtypes
# Removing the % symbol and checking if it is removed
population_data.urban_pop_percent = population_data.urban_pop_percent.str.replace("%","")
population_data.head()
urban_pop_missing_data = population_data[population_data.urban_pop_percent == "N.A."]
urban_pop_missing_data
top_10_countries_check = population_data.sort_values(by = ["population_2020"], ascending = False).head(10)
top_10_countries_check
# Checking if any of the urban_pop_missing_data values are in the top 10 list
set_top_10 = set(top_10_countries_check.country_dependency)
set_urban_pop_na = set(urban_pop_missing_data.country_dependency)
set_top_10.intersection(urban_pop_missing_data)
population_data = population_data.drop(urban_pop_missing_data.index)
# Changing the dtype of urban_pop_percent to int
population_data.urban_pop_percent = population_data.urban_pop_percent.astype("int")
population_data.dtypes

countries_population = population_data[["country_dependency", 
                                        "population_2020", 
                                        "density_p_km_sq", 
                                        "urban_pop_percent"]].sort_values(by = ["population_2020"], ascending = False)
countries_population.shape
countries_population.columns
def plot_population_data(column_1, column_2, number, title, label):
    
    plt.figure(figsize = (12, 6))
    plt.barh(countries_population[column_1].head(number), countries_population[column_2].head(number))
    plt.title(title, size = 15)
    plt.xlabel(label, size = 15)
    plt.yticks(fontsize = 14)
    plt.gca().invert_yaxis()
    plt.show()
plot_population_data("country_dependency", "population_2020", 10, "Countries with Highest Population", "Population Count")
plot_population_data("country_dependency", "density_p_km_sq", 10, "Population Density of Highest Populated Countries", 
                 "Count of People per $Km^2$")
plot_population_data("country_dependency", "urban_pop_percent", 10, 
                           "Urban Population Percentage of Highest Populated Countries", "Percentage of Urban Population")
top_10_countries_covid_data = worldometer_data.sort_values(by = ["population"], ascending = False).head(10)
top_10_countries_covid_data = top_10_countries_covid_data.sort_values(by = ("total_cases"), ascending = False).reset_index(drop = True)
def plot_countries_covid_data(dataset, column_1, column_2, color, show = True):
    
    if show:
        alpha = 1
    else:
        alpha = 0.5
        
    plt.barh(dataset[column_1], dataset[column_2], alpha =alpha, color = color, label = column_2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize = 14)
    plt.xlabel("COVID-19 Cases Count", size = 15)
    plt.yticks(fontsize = 15)
    plt.gca().invert_yaxis()
    if show:
        plt.show()
plt.figure(figsize=(12,6))
plot_countries_covid_data(top_10_countries_covid_data, "country_region", "total_cases", "blue", show = False)
#plot_countries_covid_data(top_10_countries_covid_data, "country_region", "total_recovered", "green", show = False)
plot_countries_covid_data(top_10_countries_covid_data, "country_region", "active_cases", "orange", show = False)
#plot_countries_covid_data(top_10_countries_covid_data, "country_region", "total_deaths", "red", show = False)
plt.title("Highest Populated Countries Total Cases vs Active Cases", size = 15)
#plt.xlabel("COVID-19 Cases Count", size = 15)
plt.gca().invert_yaxis()

plt.figure(figsize=(12,6))

plot_countries_covid_data(top_10_countries_covid_data, "country_region", "total_cases", "b", show = False)
plot_countries_covid_data(top_10_countries_covid_data, "country_region", "total_recovered", "g", show = False)

plt.title("Highest Populated Coutries Total Cases vs Recovered Cases", size = 15)
plt.gca().invert_yaxis()
plt.figure(figsize=(12,6))

plot_countries_covid_data(top_10_countries_covid_data, "country_region", "total_recovered", "g", show = False)
plot_countries_covid_data(top_10_countries_covid_data, "country_region", "total_deaths", "r", show = False)


plt.title("Highest Populated Coutries Total Cases vs Death Cases", size = 15)
plt.gca().invert_yaxis()
plt.figure(figsize = (12, 6))

plt.bar("Confirmed", worldometer_data.total_cases.sum(), color = "royalblue")
plt.bar("Recovered", worldometer_data.total_recovered.sum(), color = "mediumseagreen")
plt.bar("Deaths", worldometer_data.total_deaths.sum(), color = "firebrick")
plt.bar("Active", worldometer_data.active_cases.sum(), color = "purple")

plt.title("Worldwide Cases", size = 15)
plt.xticks(fontsize = 15)
plt.ylabel("Number of Cases", size = 15)
plt.show()
worldometer_data.head()
countries_by_total_cases = worldometer_data.sort_values(by = ["total_recovered"], ascending = False).head(10)
countries_by_total_deaths = worldometer_data.sort_values(by = ["total_deaths"], ascending = False).head(10)
countries_by_active_cases = worldometer_data.sort_values(by = ["active_cases"], ascending = False).head(10)
countries_by_recovered_cases = worldometer_data.sort_values(by = ["total_recovered"], ascending = False).head(10)
plt.figure(figsize = (12, 6))
plt.title("Countries with most Total Cases", size = 15)
plot_countries_covid_data(countries_by_total_cases, "country_region", "total_cases", "royalblue")

countries_by_total_cases_set = set(countries_by_total_cases.country_region)
print("Intersection: {}".format(set_top_10.intersection(countries_by_total_cases_set)))
print("Difference: {}".format(set_top_10.difference(countries_by_total_cases_set)))
plt.figure(figsize = (12, 6))
plt.title("Countries with most Recovered Cases", size = 15)
plot_countries_covid_data(countries_by_recovered_cases, "country_region", "total_recovered", "mediumseagreen")
countries_by_recovered_cases_set = set(countries_by_recovered_cases.country_region)
print("Intersection: {}".format(set_top_10.intersection(countries_by_recovered_cases_set)))
print("Difference: {}".format(set_top_10.difference(countries_by_recovered_cases_set)))
plt.figure(figsize = (12, 6))
plt.title("Countries with most Death Cases", size = 15)
plot_countries_covid_data(countries_by_total_deaths, "country_region", "total_deaths", "firebrick")
countries_by_total_deaths_set = set(countries_by_total_deaths.country_region)
print("Intersection: {}".format(set_top_10.intersection(countries_by_total_deaths_set)))
print("Difference: {}".format(set_top_10.difference(countries_by_total_deaths_set)))
plt.figure(figsize = (12, 6))
plt.title("Countries with most Acive Cases", size = 15)
plot_countries_covid_data(countries_by_active_cases, "country_region", "active_cases", "orange")
countries_by_active_cases_set = set(countries_by_active_cases.country_region)
print("Intersection: {}".format(set_top_10.intersection(countries_by_active_cases_set)))
print("Difference: {}".format(set_top_10.difference(countries_by_active_cases_set)))
worldometer_data.head()
countries_by_population = worldometer_data.sort_values(by = ["population"], ascending = False).head(10).reset_index(drop = True)
countries_by_tests = worldometer_data.sort_values(by = ["total_tests"], ascending = False).head(10).reset_index(drop = True)
countries_by_population
def plot_tests_bar(feature, value, title, df, size, show = True):
    f, ax = plt.subplots(1, 1, figsize = (size * 4, size))
    df = df.sort_values([value], ascending=False).reset_index(drop = True)
    g = sns.barplot(df[feature][0: 10], df[value][0: 10])
    g.set_title(title)
    g.set_ylabel("Number of Tests", size = 13)
    g.set_xlabel("Country", size = 13)
    if show:
        plt.show()
plot_tests_bar("country_region", "total_tests", "Highest Populated Countries Tests Count", countries_by_population, 4)
plot_tests_bar("country_region", "total_tests", "Countries with largest number of tests", countries_by_tests, 4)
def plot_who_regions(column, color):
    cases_by_who_regions = worldometer_data.groupby(["who_region"])[column].sum().sort_values(ascending = False)
    
    plt.barh(cases_by_who_regions.index, cases_by_who_regions, color = color)
    plt.title("WHO regions {}".format(column), size = 14)
    plt.xlabel("Cases", size = 14)
    plt.yticks(fontsize = 14)
    plt.gca().invert_yaxis()
    plt.show()
plot_who_regions("total_cases", "royalblue")
plot_who_regions("total_deaths", "firebrick")
plot_who_regions("total_recovered", "mediumseagreen")
covid_all_data.head()
def plot_covid_curve(dataset, column_1, column_2, show = True):
    
    cases = []
    date = []
    current_cases = 0
    for (curr_date, conf), data in dataset.groupby([column_1, column_2]):
    
        if len(date) == 0:
        
            date.append(curr_date)
            current_cases += conf
        else:
        
            if date[-1] == curr_date:
                current_cases += conf
            else:
                date.append(curr_date)
                cases.append(current_cases)
                current_cases = 0
                current_cases += conf
    cases.append(current_cases)
    if column_2 == "confirmed":
        color = "royalblue"
    elif column_2 == "recovered":
        color = "mediumseagreen"
    elif column_2 == "deaths":
        color = "firebrick"
    elif column_2 == "active":
        
        color = "orange"
    

    plt.scatter(pd.to_datetime(date), cases, color = color, label = column_2)
    plt.plot(pd.to_datetime(date), cases, color = color, linewidth = 2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize = 14)
    plt.xlabel("Date", size = 14)
    plt.ylabel("Number of Cases", size = 14)
    #plt.yscale("log")
    if show:
        plt.show()
worldometer_data[worldometer_data.who_region == "WesternPacific"].country_region.unique()
western_pacific_region_data = worldometer_data[worldometer_data.who_region == "WesternPacific"]
western_pacific_region_data = western_pacific_region_data.sort_values(by = ["total_cases"], ascending = False)
western_pacific_region_data.head(5)
plt.figure(figsize = (12, 6))
plot_covid_curve(covid_all_data[covid_all_data.country_region == "China"], "date", "confirmed", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "China"], "date", "recovered", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "China"], "date", "deaths", show = False)
plot_covid_curve(covid_19_clean[covid_19_clean.country_region == "China"], "date", "active", show = False)
plt.title("Covid-19 China", size = 14)
plt.show()
plt.figure(figsize = (12, 6))
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Philippines"], "date", "confirmed", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Philippines"], "date", "recovered", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Philippines"], "date", "deaths", show = False)
plot_covid_curve(covid_19_clean[covid_19_clean.country_region == "Philippines"], "date", "active", show = False)
plt.title("Covid-19 Philippines", size = 14)
plt.show()
plt.figure(figsize = (12, 6))
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Australia"], "date", "confirmed", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Australia"], "date", "recovered", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Australia"], "date", "deaths", show = False)
plot_covid_curve(covid_19_clean[covid_19_clean.country_region == "Australia"], "date", "active", show = False)
plt.title("Covid-19 Australia", size = 14)
plt.show()
plt.figure(figsize = (12, 6))
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Japan"], "date", "confirmed", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Japan"], "date", "recovered", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Japan"], "date", "deaths", show = False)
plot_covid_curve(covid_19_clean[covid_19_clean.country_region == "Japan"], "date", "active", show = False)
plt.title("Covid-19 Japan", size = 14)
plt.show()
plt.figure(figsize = (12, 6))
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Singapore"], "date", "confirmed", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Singapore"], "date", "recovered", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Singapore"], "date", "deaths", show = False)
plot_covid_curve(covid_19_clean[covid_19_clean.country_region == "Singapore"], "date", "active", show = False)
plt.title("Covid-19 Singapore", size = 14)
plt.show()
worldometer_data[worldometer_data.who_region == "Africa"].country_region.unique()
africa_region_data = worldometer_data[worldometer_data.who_region == "Africa"]
africa_region_data = africa_region_data.sort_values(by = ["total_cases"], ascending = False)
africa_region_data.head(5)
plt.figure(figsize = (12, 6))
plot_covid_curve(covid_all_data[covid_all_data.country_region == "South Africa"], "date", "confirmed", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "South Africa"], "date", "recovered", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "South Africa"], "date", "deaths", show = False)
plot_covid_curve(covid_19_clean[covid_19_clean.country_region == "South Africa"], "date", "active", show = False)
plt.title("Covid-19 South Africa", size = 14)
plt.show()
plt.figure(figsize = (12, 6))
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Nigeria"], "date", "confirmed", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Nigeria"], "date", "recovered", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Nigeria"], "date", "deaths", show = False)
plot_covid_curve(covid_19_clean[covid_19_clean.country_region == "Nigeria"], "date", "active", show = False)
plt.title("Covid-19 Nigeria", size = 14)
plt.show()
worldometer_data[worldometer_data.who_region == "EasternMediterranean"].country_region.unique()
eastern_medit_region_data = worldometer_data[worldometer_data.who_region == "EasternMediterranean"]
eastern_medit_region_data = eastern_medit_region_data.sort_values(by = ["total_cases"], ascending = False)
eastern_medit_region_data.head(5)
plt.figure(figsize = (12, 6))
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Iran"], "date", "confirmed", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Iran"], "date", "recovered", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Iran"], "date", "deaths", show = False)
plot_covid_curve(covid_19_clean[covid_19_clean.country_region == "Iran"], "date", "active", show = False)
plt.title("Covid-19 Iran", size = 14)
plt.show()
plt.figure(figsize = (12, 6))
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Saudi Arabia"], "date", "confirmed", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Saudi Arabia"], "date", "recovered", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Saudi Arabia"], "date", "deaths", show = False)
plot_covid_curve(covid_19_clean[covid_19_clean.country_region == "Saudi Arabia"], "date", "active", show = False)
plt.title("Covid-19 Saudi Arabia", size = 14)
plt.show()
worldometer_data[worldometer_data.who_region == "South-EastAsia"].country_region.unique()
south_eastasia_region_data = worldometer_data[worldometer_data.who_region == "South-EastAsia"]
south_eastasia_region_data = south_eastasia_region_data.sort_values(by = ["total_cases"], ascending = False)
south_eastasia_region_data.head(5)
plt.figure(figsize = (12, 6))
plot_covid_curve(covid_all_data[covid_all_data.country_region == "India"], "date", "confirmed", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "India"], "date", "recovered", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "India"], "date", "deaths", show = False)
plot_covid_curve(covid_19_clean[covid_19_clean.country_region == "India"], "date", "active", show = False)
plt.title("Covid-19 India", size = 14)
plt.show()
plt.figure(figsize = (12, 6))
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Bangladesh"], "date", "confirmed", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Bangladesh"], "date", "recovered", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Bangladesh"], "date", "deaths", show = False)
plot_covid_curve(covid_19_clean[covid_19_clean.country_region == "Bangladesh"], "date", "active", show = False)
plt.title("Covid-19 Bangladesh", size = 14)
plt.show()
plt.figure(figsize = (12, 6))
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Indonesia"], "date", "confirmed", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Indonesia"], "date", "recovered", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Indonesia"], "date", "deaths", show = False)
plot_covid_curve(covid_19_clean[covid_19_clean.country_region == "Indonesia"], "date", "active", show = False)
plt.title("Covid-19 Indonesia", size = 14)
plt.show()
worldometer_data[worldometer_data.who_region == "Europe"].country_region.unique()
europe_region_data = worldometer_data[worldometer_data.who_region == "Europe"]
europe_region_data = europe_region_data.sort_values(by = ["total_cases"], ascending = False)
europe_region_data.head()
plt.figure(figsize = (12, 6))
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Russia"], "date", "confirmed", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Russia"], "date", "recovered", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Russia"], "date", "deaths", show = False)
plot_covid_curve(covid_19_clean[covid_19_clean.country_region == "Russia"], "date", "active", show = False)
plt.title("Covid-19 Russia", size = 14)
plt.show()
plt.figure(figsize = (12, 6))
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Spain"], "date", "confirmed", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Spain"], "date", "recovered", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Spain"], "date", "deaths", show = False)
plot_covid_curve(covid_19_clean[covid_19_clean.country_region == "Spain"], "date", "active", show = False)
plt.title("Covid-19 Spain", size = 14)
plt.show()
plt.figure(figsize = (12, 6))
plot_covid_curve(covid_all_data[covid_all_data.country_region == "United Kingdom"], "date", "confirmed", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "United Kingdom"], "date", "recovered", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "United Kingdom"], "date", "deaths", show = False)
plot_covid_curve(covid_19_clean[covid_19_clean.country_region == "United Kingdom"], "date", "active", show = False)
plt.title("Covid-19 United Kingdom", size = 14)
plt.show()
plt.figure(figsize = (12, 6))
#plot_covid_curve(covid_all_data[covid_all_data.country_region == "India"], "date", "confirmed", show = False)
#plot_covid_curve(covid_all_data[covid_all_data.country_region == "US"], "date", "confirmed", show = False)
#plot_covid_curve(covid_all_data[covid_all_data.country_region == "Brazil"], "date", "confirmed", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Italy"], "date", "confirmed", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Italy"], "date", "recovered", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Italy"], "date", "deaths", show = False)
plot_covid_curve(covid_19_clean[covid_19_clean.country_region == "Italy"], "date", "active", show = False)
plt.title("Covid-19 Italy", size = 14)
plt.show()
plt.figure(figsize = (12, 6))
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Germany"], "date", "confirmed", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Germany"], "date", "recovered", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Germany"], "date", "deaths", show = False)
plot_covid_curve(covid_19_clean[covid_19_clean.country_region == "Germany"], "date", "active", show = False)
plt.title("Covid-19 Germany", size = 14)
plt.show()
plt.figure(figsize = (12, 6))
plot_covid_curve(covid_all_data[covid_all_data.country_region == "France"], "date", "confirmed", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "France"], "date", "recovered", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "France"], "date", "deaths", show = False)
plot_covid_curve(covid_19_clean[covid_19_clean.country_region == "France"], "date", "active", show = False)
plt.title("Covid-19 France", size = 14)
plt.show()
plt.figure(figsize = (12, 6))
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Bulgaria"], "date", "confirmed", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Bulgaria"], "date", "recovered", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Bulgaria"], "date", "deaths", show = False)
plot_covid_curve(covid_19_clean[covid_19_clean.country_region == "Bulgaria"], "date", "active", show = False)
plt.title("Covid-19 Bulgaria", size = 14)
plt.show()
worldometer_data[worldometer_data.who_region == "Americas"].country_region.unique()
americas_region_data = worldometer_data[worldometer_data.who_region == "Americas"]
americas_region_data = americas_region_data.sort_values(by = ["total_cases"], ascending = False)
americas_region_data.head(5)
plt.figure(figsize = (12, 6))
plot_covid_curve(covid_all_data[covid_all_data.country_region == "US"], "date", "confirmed", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "US"], "date", "recovered", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "US"], "date", "deaths", show = False)
plot_covid_curve(covid_19_clean[covid_19_clean.country_region == "US"], "date", "active", show = False)
plt.title("Covid-19 US", size = 14)
plt.show()
plt.figure(figsize = (12, 6))
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Brazil"], "date", "confirmed", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Brazil"], "date", "recovered", show = False)
plot_covid_curve(covid_all_data[covid_all_data.country_region == "Brazil"], "date", "deaths", show = False)
plot_covid_curve(covid_19_clean[covid_19_clean.country_region == "Brazil"], "date", "active", show = False)
plt.title("Covid-19 Brazil", size = 14)
plt.show()