# Covid deaths per million inhabitants on day 170th after 1st death by country

# Data source:
# https://github.com/owid/covid-19-data/tree/master/public/data

# Author: Matías Micheletto

import csv

def toFloat(value):
    # Helper for converting str to float and returning 0 if invalid
    try:
        result = float(value)
    except ValueError:
        result = 0
    return result

with open('../input/totalcoviddeathspermillion/total-covid-deaths-per-million.csv', newline='') as csvfile:
    # Column indexes
    column_country = 0 # Country name
    column_date = 2 # Date
    column_dpm = 3 # Deaths per million inhabitants

    myreader = csv.reader(csvfile, delimiter = ',')
    rows = list(myreader)
    
    # Data of each country
    deaths = {} # Deaths per million over time
    dates = {} # Dates of data

    for row in rows[1:]: # For every day
        m = toFloat(row[column_dpm]) # Cumulative deaths
        d = row[column_date] # Date
        if m != 0: # Add data only after 1st death
            if row[column_country] in deaths:
                deaths[row[column_country]].append(m)
                dates[row[column_country]].append(d)
            else:
                deaths[row[column_country]] = [m]
                dates[row[column_country]] = [d]
        
    # Get deaths per million on day 170th
    deaths170 = {} # Data
    dates170 = {} # Dates
    for country in deaths:
        if len(deaths[country]) >= 170:
            deaths170[country] = deaths[country][169]
            dates170[country] = dates[country][169]

    # Sort the list and print
    sorted_list = sorted(deaths170.items(), key=lambda x: x[1], reverse=True)
    print("Country".ljust(16) + "Deaths/Million".ljust(16) + "Date of day 170 after 1st death")
    print("----------------------------------------------------------------")
    for country in sorted_list:
        print(country[0].ljust(16) + str(country[1]).ljust(16) + dates170[country[0]])


