import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

#import folium

import glob

import os
def plotting(variable, xlabel, ylabel, title, figsize):

    plt.figure(figsize = figsize)

    saarc_countries_codes = ["IND", "AFG", "PAK", "BTN", "NPL", "BGD", "MDV", "LKA"]           

    for country in saarc_countries_codes:

        mask = variable["CountryCode"].str.contains(country)

        variable_country = variable[mask]

        plt.plot(variable_country["Year"], variable_country["Value"], label = country)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.title(title)

    plt.ylim(ymin = 0)

    plt.grid(True)

    plt.legend()

    plt.show()
data = pd.read_csv("./world-development-indicators/Indicators.csv")



data.columns
mask_saarc = data["CountryCode"].isin(["IND", "AFG", "PAK", "BTN", "NPL", "BGD", "MDV", "LKA"])



data_saarc = data[mask_saarc]



data_saarc.columns
mask_for_gdp = data_saarc["IndicatorName"].str.contains("^GDP at market prices \(constant 2005 US\$\)")



gdp_data = data_saarc[mask_for_gdp]



gdp_data.head()
plotting(gdp_data, "Year", "GDP at market prices (constant 2005 US $)", "GDP among SAARC", (10, 10))
mask_gdp_per_capita = data_saarc["IndicatorName"].str.contains("GDP per capita \(constant 2005 US\$\)")



gdp_per_capita = data_saarc[mask_gdp_per_capita]



gdp_per_capita.head()
plotting(gdp_per_capita, "Year", "GDP per capita", "GDP per capita(Constant 2005 USD $)", (10, 10))
mask_literacy_rate = data_saarc["IndicatorName"].str.contains("Adult literacy rate, population 15\+ years, both sexes")



literacy_rate = data_saarc[mask_literacy_rate]



literacy_rate.head()
plotting(literacy_rate, "Year", "Adult Literacy rate", "Adult literacy rate, population 15+ years %", (10, 10))
mask_unemployment_rate = data_saarc["IndicatorName"].str.contains("Unemployment, total \(% of total labor force\)")



unemployment_rate = data_saarc[mask_unemployment_rate]



unemployment_rate.head()
plt.figure(figsize = (15, 15))



plt.title("Unemployment rate for SAARC. (Y-axis ajusted to accomodate all the values)")



i = 1



for country in saarc_countries_codes:

    plt.subplot(3, 3, i)

    i += 1

    mask = unemployment_rate["CountryCode"].str.contains(country)

    ue = unemployment_rate[mask]

    plt.plot(ue["Year"], ue["Value"])

    plt.xlabel("Year")

    plt.ylabel("Value %")

    plt.title("% Unemployment Rate in "+country+"(Y-axis adjusted)")



plt.show()
literacy_rate_a = literacy_rate[["CountryCode", "Year", "Value"]]



unemployment_rate_a = unemployment_rate[["CountryCode", "Year", "Value"]]



result = pd.merge(left = literacy_rate_a, right = unemployment_rate_a, 

                  on = ["CountryCode", "Year"], suffixes = ("_literacy", "_unemployment"))



result.head()
result[["Value_literacy", "Value_unemployment"]].corr()
for country in saarc_countries_codes:

    mask = result["CountryCode"].str.contains(country)

    result_country = result[mask]

    print("Correlation between Literacy and Unemployment for "+country)

    print("\n")

    print(result_country[["Value_literacy", "Value_unemployment"]].corr())

    print("\n")
agri_data_mask = data_saarc["IndicatorName"].str.contains("Agriculture, value added \(\% of GDP\)")



agri_data = data_saarc[agri_data_mask]



agri_data.head()
plotting(agri_data, "Year", "Value", agri_data.iloc[0]["IndicatorName"], (10, 10))
agri_emp_data_mask = data_saarc["IndicatorName"].str.contains("Employment in agriculture \(\% of total employment\)")



agri_emp_data = data_saarc[agri_emp_data_mask]



agri_emp_data.head()
agri_emp_data_a = agri_emp_data[["Year", "CountryCode", "Value"]]



result = pd.merge(left = agri_emp_data_a, right = unemployment_rate_a, 

                  on = ["CountryCode", "Year"], suffixes = ("_agri", "_unemployment"))



result.head()
result[["Value_agri", "Value_unemployment"]].corr()
poverty_mask = data_saarc["IndicatorName"].str.contains("Poverty headcount ratio at \$1.90 a day \(2011 PPP\) \(\% of population\)")



poverty = data_saarc[poverty_mask]



poverty.head()
plotting(poverty, "Year", "Poverty %", poverty.iloc[0].IndicatorName, (10, 10))
tax_mask = data_saarc["IndicatorName"].str.contains("Tax revenue \(\% of GDP\)")



tax = data_saarc[tax_mask]



tax.head()
plotting(tax, "Year", "Tax Revenue % of GDP", tax.iloc[0].IndicatorName, (10, 10))
tourism_mask = data_saarc["IndicatorName"].str.contains("International tourism, expenditures \(current US\$\)")



tourism = data_saarc[tourism_mask]



tourism.head()
plotting(tourism, "Year", "Expenditure", tourism.iloc[0].IndicatorName, (10, 10))
income_mask = data_saarc["IndicatorName"].str.contains("Survey mean consumption or income per capita\, total population \(2011 PPP \$ per day\)")



income = data_saarc[income_mask]



income.head()
plotting(income, "Year", "Income per capita", income.iloc[0].IndicatorName, (10, 10))