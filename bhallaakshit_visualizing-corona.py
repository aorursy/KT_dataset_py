import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import plotly
import plotly.express as px
path = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
data_covid_infections = pd.read_csv(path + "time_series_covid19_confirmed_global.csv")
data_covid_deaths = pd.read_csv(path + "time_series_covid19_deaths_global.csv")
data_covid_recovered = pd.read_csv(path + "time_series_covid19_recovered_global.csv")
country_coord = data_covid_infections[["Country/Region", "Lat", "Long"]]
country_coord = country_coord.groupby(country_coord["Country/Region"]).mean()
def CleanMe(df, col):
    df = df.drop(columns = ["Lat", "Long", "Province/State"])
    df = df.groupby(df["Country/Region"]).sum()
    df["Country_Region"] = df.index
    df = df.melt(id_vars = "Country_Region", var_name = "Date", value_name = col)
    df['Date'] = df['Date'].astype('datetime64[ns]').astype("str")
    return df
temp1 = CleanMe(data_covid_infections, "infections")
temp2 = CleanMe(data_covid_deaths, "deaths")
temp3 = CleanMe(data_covid_recovered, "recovered")
data_covid = pd.merge(temp1, temp2)
data_covid = pd.merge(data_covid, temp3)
data_covid
# pop = pd.read_csv("../input/2020-population/population.csv")
# pop.index = pop["Country"]
# pop = pop.drop(columns = ["Country", "Population"])

# convert = {
#     'Myanmar': 'Burma',
#     'Congo': 'Congo (Kinshasa)',
#     "Côte d'Ivoire": "Cote d'Ivoire",
#     'Czech Republic (Czechia)': 'Czechia',
#     'South Korea': 'Korea, South',
#     'Saint Kitts & Nevis': 'Saint Kitts and Nevis',
#     'St. Vincent & Grenadines': 'Saint Vincent and the Grenadines',
#     'Sao Tome & Principe': 'Sao Tome and Principe',
#     'Taiwan': 'Taiwan*',
#     'United States': 'US',
# }

# pop = pop.rename(index = convert)
# pop.index.name = "Country_Region"

# MyMap = dict(zip(pop.index, pop["pop_million"]))

# cols = ["infections", "deaths", "recovered"]
# for col in cols:
#     data_covid[col + "_million"] = data_covid["Country_Region"]
#     data_covid[col + "_million"] = data_covid[col + "_million"].map(MyMap)
#     data_covid[col + "_million"] = np.array(data_covid[col + "_million"]) * np.array(data_covid[col])

# data_covid = data_covid.drop(columns = cols)
# data_covid = data_covid.rename(columns = {
#     "infections_million": "infections",
#     "deaths_million": "deaths",
#     "recovered_million": "recovered"
# }) 
# data_covid
def PlotDailyGlobalCases(data, case_type):
    fig = px.line(
        data, 
        x = "Date", 
        y = case_type, 
        color = "Country_Region",
        title = "Global trends in " +  case_type
    )
    fig.update_traces(mode = "markers+lines")
    fig.show()
    
    plotly.offline.plot(fig, filename = "DailyGlobal_" + case_type + ".html", auto_open = False)
PlotDailyGlobalCases(data_covid, "infections")
PlotDailyGlobalCases(data_covid, "deaths")
PlotDailyGlobalCases(data_covid, "recovered")
def PlotTopGlobalCases(data, case_type):
    data = data.sort_values(by = case_type, ascending = False)
    data = data.iloc[:20, :]
    
    fig = px.bar(
        data, 
        x = data.index, 
        y = case_type, 
        title = "Highest " + case_type
    )
    fig.show()
    
    plotly.offline.plot(fig, filename = "TopGlobal_" + case_type + ".html", auto_open = False)
df = data_covid.loc[data_covid["Date"] == "2020-05-03", ["infections", "deaths", "recovered", "Country_Region"]]
df.index = df["Country_Region"]
df = df.drop(columns = ["Country_Region"])
PlotTopGlobalCases(df, "infections")
PlotTopGlobalCases(df, "deaths")
PlotTopGlobalCases(df, "recovered")
def MapGrowth(data, case_type):
    country_and_code = pd.read_csv("../input/country-codes/country_and_code.csv")
    
    df = data.sort_values(["Date", case_type])
    df.index = range(df.shape[0])

    MyMap = dict(zip(country_and_code["country"], country_and_code["code"]))
    df["code"] = df["Country_Region"].map(MyMap)

    rep = {
        "US": "USA",
        "Iran": "IRN",
        "United Kingdom": "GBR",
        "Korea, South": "PRK",
        "Russia": "RUS",
        "Diamond Princess": "DMP",    #
        "Congo (Kinshasa)": "COD",
        "Moldova": "MDA",
        "Bolivia": "BOL",
        "Taiwan*": "TWN",
        "Venezuela": "VEN",
        "Congo (Brazzaville)": "COG",
        "Kosovo": "RKS",
        "Syria": "SYR",
        "Cote d'Ivoire": "CIV",
        "MS Zaandam": "MSZ",           #
        "West Bank and Gaza": "WBG",   #
        "Burma": "MMR",
        "Brunei": "BRN",
        "Tanzania": "TZA",
        "Vietnam": "VNM",
        "Laos": "LAO"                  #
    }

    for i in range(df.shape[0]):
        if pd.isnull(df.loc[i, "code"]):
            country = df.loc[i, "Country_Region"]
            code = rep[country]
            df.loc[i, "code"] = code
        
    fig = px.scatter_geo(
        df, 
        locations = "code", 
        color = "Country_Region",
        hover_name = "Country_Region", 
        size = case_type,
        animation_frame = "Date",
        projection = "natural earth",
        title = "Growth of " + case_type 
    )
    fig.show()
    
    plotly.offline.plot(fig, filename = case_type + ".html", auto_open = False)
MapGrowth(data_covid, "infections")
MapGrowth(data_covid, "deaths")
MapGrowth(data_covid, "recovered")
