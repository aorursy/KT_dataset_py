import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import glob

import os

import datetime
# Show entire dataframes

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)
def mydateparser(dateStr):

    

    formatStrs = [

        "%m/%d/%Y %H:%M:%S",

        "%m/%d/%Y %H:%M",

        "%m/%d/%Y",

        "%m/%d/%y %H:%M:%S",

        "%m/%d/%y %H:%M",

        "%m/%d/%y",

        "%Y-%m-%dT%H:%M:%S",

        "%Y-%m-%d %H:%M:%S",

        "%Y-%m-%d %H:%M",

        "%Y-%m-%d",

    ]

    

    myDate = None

    for formatStr in formatStrs:

        try:

            myDate = pd.datetime.strptime(dateStr, formatStr)

            break

        except ValueError:

            pass

    

    return myDate
path = "/kaggle/input/jhucovid19/csse_covid_19_data/csse_covid_19_daily_reports"

all_files = glob.glob(os.path.join(path, "*.csv"))



covid_dfs = []

for filename in all_files:

    covid_df_temp = pd.read_csv(filename, index_col=None, header=0, skipinitialspace=True, parse_dates=True, date_parser=mydateparser)

    

    # Rename incoming columns to be consistent

    covid_df_temp.rename(columns={

        'Province/State': 'Province_State', 

        'Country/Region': 'Country_Region',

        'Latitude': 'Lat',

        'Longitude': 'Long',

        'Long_': 'Long',

        'Last Update': 'Last_Update'

    }, inplace=True)

    

    covid_df_temp["Date"] = pd.to_datetime(covid_df_temp["Last_Update"]).dt.date

    covid_dfs.append(covid_df_temp)



covid_df = pd.concat(covid_dfs, axis=0, ignore_index=True, sort=False)



print(covid_df.shape)
covid_df.head()
filename = "/kaggle/input/population-by-country-2020/population_by_country_2020.csv"

populations_df = pd.read_csv(filename, index_col=None, header=0)
covid_df.sort_values(by=['Last_Update'], inplace=True)
# Standardise names

country_name_mapping = {

    "Bahamas": ["Bahamas, The", "The Bahamas"],

    "China": ["Mainland China"],

    "Congo": ["Congo (Brazzaville)", "Congo (Kinshasa)", "Republic of the Congo"],

    "Ivory Coast": ["CÃ´te d'Ivoire", "Cote d'Ivoire"],

    "Curacao": ["CuraÃ§ao"],

    "Czechia": ["Czech Republic", "Czech Republic (Czechia)"],

    "Gambia": ["Gambia, The", "The Gambia"],

    "Hong Kong": ["Hong Kong SAR"],

    "Iran": ["Iran (Islamic Republic of)"],

    "Ireland": ["Republic of Ireland"],

    "Macau": ["Macao SAR", "Macao"],

    "Moldova": ["Republic of Moldova"],

    "Russia": ["Russian Federation"],

    "Saint Kitts and Nevis": ["Saint Kitts & Nevis"],

    "Sao Tome and Principe": ["Sao Tome & Principe"],

    "South Korea": ["Korea, South", "Republic of Korea"],

    "Taiwan": ["Taiwan*"],

    "United Kingdom": ["UK"],

    "United States": ["US", "USA"],

    "Vietnam": ["Viet Nam"],

    "Vatican City": ["Holy See"]

}



# Update country names for consistency

for (country_name, country_name_mappings) in country_name_mapping.items():

    for mapped_country_name in country_name_mappings:

        #print(country_name + " <- " + mapped_country_name)

        covid_df["Country_Region"].replace(mapped_country_name, country_name, inplace=True)

        populations_df["Country (or dependency)"].replace(mapped_country_name, country_name, inplace=True)
# Group by country and filename (date acquired)

covid_daily = covid_df.groupby(["Country_Region", "Date"]).agg({

    "Confirmed": ["sum"],

    "Deaths": ["sum"],

    "Recovered": ["sum"],

    "Active": ["sum"],

})



# Rework columns

#covid_daily.columns = ["Confirmed", "Deaths", "Recovered", "Active", "Last_Update"]

#covid_daily.reset_index()

covid_daily.columns = covid_daily.columns.get_level_values(0)

None
covid_daily
# TODO: Calculate the gradient for the data for each country/day

#covid_daily["Confirmed Grad"] = pd.Series(np.gradient(covid_daily["Confirmed"], covid_daily.index))



# TODO:  BUG -> There is overflow from group to group.  This operation needs to be confined to the current group.



covid_daily["Confirmed Change"] = covid_daily["Confirmed"] - covid_daily["Confirmed"].shift()

covid_daily["Deaths Change"] = covid_daily["Deaths"] - covid_daily["Deaths"].shift()

covid_daily["Recovered Change"] = covid_daily["Recovered"] - covid_daily["Recovered"].shift()

covid_daily["Active Change"] = covid_daily["Active"] - covid_daily["Active"].shift()
covid_daily.loc["Australia"]
# View dataset

#populations_df.sort_values(by=["Country (or dependency)"], inplace=True)

#populations_df
covid_latest = covid_daily.groupby(level=0).apply(max)



# View dataset

#covid_latest.sort_values(by=["Country_Region"], inplace=True)

#covid_latest
df_merged = pd.merge(covid_latest, populations_df, left_on="Country_Region", right_on="Country (or dependency)", how='inner')



df_merged.sort_values(by=['Population (2020)'], inplace=True, ascending=False)

df_merged



#df_merged.dropna(subset=['Population (2020)'])

#df_merged.sort_values(by=['Population (2020)'], inplace=True, ascending=False)
# Get all countries

covid_countries = np.sort(np.unique(covid_df["Country_Region"]))

population_countries = np.sort(np.unique(populations_df["Country (or dependency)"]))



print("COVID countries: {}".format(len(covid_countries)))

print("Countries with pop data: {}".format(len(population_countries)))



unaccounted_countries = set(covid_countries).difference(set(population_countries))

print("COVID countries/regions not accounted for: {}".format(len(unaccounted_countries)))

print(unaccounted_countries)
# Use seaborn style defaults and set the default figure size

sns.set(rc={'figure.figsize':(11, 4)})
covid_daily.loc["Australia"]["Active"].plot();
covid_df_wa = covid_df[(covid_df["Country_Region"]=="Australia") & (covid_df["Province_State"]=="Western Australia")].sort_values(by=["Last_Update"])

covid_df_aus = covid_df[(covid_df["Country_Region"]=="Australia")].sort_values(by=["Last_Update"])

covid_df_swe = covid_df[(covid_df["Country_Region"]=="Sweden")].sort_values(by=["Last_Update"])

covid_df_swe = covid_df[(covid_df["Country_Region"]=="South Africa")].sort_values(by=["Last_Update"])