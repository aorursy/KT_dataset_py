# Import dependencies

import io
import os
import numpy as np
import pandas as pd
import json
import csv
# import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 1000)
# Import COVID-19 dataset from CSSE at Johns Hopkins University Github
confirmed = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
recovered = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")
deaths = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")


# Fetch the column names of the dataset for each categories
cols_confirmed = [col for col in confirmed.columns]
cols_recovered = [col for col in recovered.columns]
cols_deaths = [col for col in deaths.columns]
# Initialize empty DataFrame so that we can assign/prepare subset columns we need
initial_columns = ['country']
country_lat_lng = pd.DataFrame(columns=initial_columns)
confirmed_temp = pd.DataFrame(columns=initial_columns)
recovered_temp = pd.DataFrame(columns=initial_columns)
deaths_temp = pd.DataFrame(columns=initial_columns)
## Country with Latitude and Longitude

country_lat_lng['country'] = confirmed[cols_confirmed[1]]
country_lat_lng['latitude'] = confirmed[cols_confirmed[2]]  
country_lat_lng['longitude'] = confirmed[cols_confirmed[3]] 

# Group by country for "Country"
# We need to do this since we are only selecting the "Country" totals

country_lat_lng.drop_duplicates(subset ="country", keep = 'last', inplace = True) 

## CONFIRMED CASES

# Fetch the necessary data column from the original "confirmed" dataset
confirmed_temp["country"] = confirmed['Country/Region']


# Confirmed data => total cases, past 24 hours, 7 days, and 30 days
confirmed_temp["total_confirmed"] = confirmed[cols_confirmed[-1]]
confirmed_temp["c_last_24hours"] = confirmed[cols_confirmed[-1]] - confirmed[cols_confirmed[-2]]
confirmed_temp["c_last_7days"] = confirmed[cols_confirmed[-1]] - confirmed[cols_confirmed[-7]]
confirmed_temp["c_last_30days"] = confirmed[cols_confirmed[-1]] - confirmed[cols_confirmed[-30]]

# Group by country for "Country"
# We need to do this since we are only selecting the "Country" totals
confirmed_unique_country = confirmed_temp.groupby('country')['total_confirmed','c_last_24hours','c_last_7days','c_last_30days'].agg('sum')
## RECOVERED CASES

# Fetch the necessary data column from the original "confirmed" dataset
recovered_temp["country"] = recovered['Country/Region']

# Confirmed data => total cases, past 24 hours, 7 days, and 30 days
recovered_temp["total_recovered"] = recovered[cols_recovered[-1]]
recovered_temp["r_last_24hours"] = recovered[cols_recovered[-1]] - recovered[cols_recovered[-2]]
recovered_temp["r_last_7days"] = recovered[cols_recovered[-1]] - recovered[cols_recovered[-7]]
recovered_temp["r_last_30days"] = recovered[cols_recovered[-1]] - recovered[cols_recovered[-30]]

# Group by country for "Country"
# We need to do this since we are only selecting the "Country" totals
recovered_unique_country = recovered_temp.groupby('country')['total_recovered','r_last_24hours','r_last_7days','r_last_30days'].agg('sum')
## DEATHS CASES

# Fetch the necessary data column from the original "confirmed" dataset
deaths_temp["country"] = deaths['Country/Region']

# Confirmed data => total cases, past 24 hours, 7 days, and 30 days
deaths_temp["total_deaths"] = deaths[cols_deaths[-1]]
deaths_temp["d_last_24hours"] = deaths[cols_deaths[-1]] - deaths[cols_deaths[-2]]
deaths_temp["d_last_7days"] = deaths[cols_deaths[-1]] - deaths[cols_deaths[-7]]
deaths_temp["d_last_30days"] = deaths[cols_deaths[-1]] - deaths[cols_deaths[-30]]

# Group by country for "Country"
# We need to do this since we are only selecting the "Country" totals
deaths_unique_country = deaths_temp.groupby('country')['total_deaths','d_last_24hours','d_last_7days','d_last_30days'].agg('sum')
## MERGE all frames by country name

# c_r for confirmed and recovered
c_r = pd.merge(confirmed_unique_country, recovered_unique_country, on='country')

# c_r_d for confirmed, recovered, and deaths
c_r_d = pd.merge(c_r, deaths_unique_country, on='country')


# all_summary for country, latitude, longitude, confirmed, recovered, and deaths data
all_summary = pd.merge(country_lat_lng, c_r_d,  on='country')

# Fetch the "all_summary" column names
[col for col in all_summary.columns]

all_summary.info()
# Save the "all_summary" data into a file named "all_summary_final.csv"

all_summary.to_csv("all_summary_final.csv", index=False)


# Read the "all_summary_final.csv" file to create a JSON object

with open('all_summary_final.csv') as csv_file:
    read_csv = csv.reader(csv_file, delimiter = ',')
    first_line = True
    all_summary_json = []
    for row in read_csv:
        if not first_line:
            all_summary_json.append({
            "country" : str(row[0]),
            "latitude" : float(row[1]),
            "longitude" : float(row[2]),
            "total_confirmed" : float(row[3]),
            "c_last_24hours" : float(row[4]),
            "c_last_7days" : float(row[5]),
            "c_last_30days" : float(row[6]),
            "total_recovered" : float(row[7]),
            "r_last_24hours" : float(row[8]),
            "r_last_7days" : float(row[9]),
            "r_last_30days" : float(row[10]),
            "total_deaths" : float(row[11]),
            "d_last_24hours" : float(row[12]),
            "d_last_7days" : float(row[13]),
            "d_last_30days" : float(row[14])
            })
        else:
            first_line = False
    

# View the JSON object

all_summary_json