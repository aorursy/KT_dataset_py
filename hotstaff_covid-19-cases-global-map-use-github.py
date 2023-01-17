# import modules

import os

import numpy as np        # linear algebra

import pandas as pd       # data processing, CSV file I/O (e.g. pd.read_csv)

import geopandas as gpd   # geopandas

import geoplot

import matplotlib.pyplot as plt

import matplotlib.colors as colors



import glob               # Used to get a file list
# clone

!git clone --depth=1 https://github.com/CSSEGISandData/2019-nCoV.git



# daily_case_updates data 

%ls 2019-nCoV/csse_covid_19_data/csse_covid_19_daily_reports/
# read csv files

CSV_FILES = glob.glob('./2019-nCoV/csse_covid_19_data/csse_covid_19_daily_reports/*.csv')



# join

df_list = []

for file in CSV_FILES:

    read_df = pd.read_csv(file)

    # Add Date series

    read_df["Date"] = os.path.basename(file.rstrip(".csv"))

    read_df["Date"] = pd.to_datetime(read_df["Date"], format='%m-%d-%Y')

    # Adopt format from 03-23-2020

    if 'Province_State' not in read_df.columns:

        read_df = read_df.rename(columns={

            'Province/State': 'Province_State',

            'Country/Region': 'Country_Region',

            'Last Update': 'Last_Update'})

    df_list.append(read_df)



df = pd.concat(df_list, sort=False)



# data cleaning

df = df.loc[:,["Province_State", "Country_Region", "Date", "Last_Update", "Confirmed", "Deaths", "Recovered"]]



TYPOS={

    " Azerbaijan": "Azerbaijan"

}

df.replace(TYPOS, inplace=True)



# Save

df.to_csv('INPUT.csv', encoding='utf_8', index=False)

df.info()
df = pd.read_csv("./INPUT.csv")

df.info()
# fill NaN

df = df.fillna({"Province_State": 'None'})



# Change astype

df["Date"] = pd.to_datetime(df["Date"])

df["Last_Update"] = pd.to_datetime(df["Last_Update"])

df["Confirmed"] = df["Confirmed"].fillna(0).astype('int')

df["Deaths"] = df["Deaths"].fillna(0).astype('int')

df["Recovered"] =df["Recovered"].fillna(0).astype('int')

df.info()
REPLACE_LIST={

    "Mainland China": "China",

    "Hong Kong": "China",

    "Macau": "China",

    "United States": "United States of America",

    "US": "United States of America",

    "UK": "United Kingdom",

    "Singapore": "Malaysia",

    "Ivory Coast": "Côte d'Ivoire",

    "Bahrain": "Qatar",

    "North Macedonia": "Macedonia",

    "San Marino": "Italy",

    "North Ireland": "United Kingdom",

    "Monaco": "France",

    "Dominican Republic": "Dominican Rep.",

    "Czech Republic": "Czechia",

    "Faroe Islands": "Denmark",

    "Gibraltar": "United Kingdom",

    "Saint Barthelemy": "France",

    "Vatican City": "Italy",

    "Bosnia and Herzegovina":"Bosnia and Herz.",

    "Malta": "Italy",

    "Martinique":"France",

    "Republic of Ireland": "Ireland",

    "Iran (Islamic Republic of)": "Iran",

    "Republic of Korea": "South Korea",

    "Hong Kong SAR": "China",

    "Macao SAR": "China",

    "Viet Nam": "Vietnam",

    "Taipei and environs": "Taiwan",

    "occupied Palestinian territory": "Palestine",

    "Russian Federation": "Russia",

    "Holy See": "Italy",

    "Channel Islands": "United Kingdom",

    "Republic of Moldova": "Moldova",

    "Cote d'Ivoire": "Côte d'Ivoire",

    "Congo (Kinshasa)": "Dem. Rep. Congo",

    "Korea, South": "South Korea",

    "Taiwan*": "Taiwan",

    "Reunion": "France",

    "Guadeloupe": "France",

    "Cayman Islands": "United Kingdom", 

    "Aruba": "Netherlands",

    "Curacao": "Netherlands",

    "Eswatini":"eSwatini",

    "Saint Vincent": "Italy",

    "Equatorial Guinea": "Eq. Guinea",

    "Central African Republic": "Central African Rep.",

    "Congo (Brazzaville)" : "Congo",

    "Republic of the Congo": "Congo",

    "Mayotte": "France",

    "Guam": "United States of America",

    "The Bahamas": "Bahamas",

    "Others": "Diamond Princess",

    "Cruise Ship": "Diamond Princess",

    "The Gambia": "Gambia",

    "Gambia, The": "Gambia",

    "Bahamas, The": "Bahamas",

    "Cabo Verde": "Cape Verde",

    "East Timor": "Timor-Leste",

    "West Bank and Gaza": "Palestine",

    "Burma": "Myanmar",

    "South Sudan": "S. Sudan",

    "Western Sahara": "W. Sahara"

}



# As Singapore is not on the geopanda world map,

# the solution is to transfer Singapore to a nearby country.

df["Country_Region"] = df["Country_Region"].replace(REPLACE_LIST)
# Use Date as Last Update.

LAST_UPDATE = df["Date"].max()



# datetime string

LAST_UPDATE_STRING = pd.to_datetime(LAST_UPDATE).strftime('%Y-%m-%d(EST)')



print(f"LAST_UPDATE is {LAST_UPDATE_STRING}")



# Sort

df = df.sort_values(["Date"])



# Select last dataframe

last = df[df.Date == LAST_UPDATE]



last.sort_values(["Confirmed", 'Country_Region'], ascending=False)
# Geopandas world map

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Remove Antarctica

world = world[(world.pop_est>0) & (world.name!="Antarctica")]





# Split French Guiana from France.

shape = world[world['name'] == 'France']['geometry'].all()



# shape[0] is French Guiana in South America

gu_df = gpd.GeoDataFrame({"name": ["French Guiana"],

                          "pop_est":[250109],

                          "continent":["South America"],

                          "gdp_md_est":[52000.0],

                          "iso_a3": -99},

                         geometry=[shape[0]])



world = world.append(gu_df, sort=False, ignore_index=True)



# shape[1,2] is France in Europa

fr_df = gpd.GeoDataFrame(pd.Series(['France', 'France'], name='country'),

                         geometry=[shape[1], shape[2]])

fr_geometry = fr_df.dissolve(by='country')['geometry'].values



world.at[world['name'] == 'France', 'geometry'] = fr_geometry
# For simplicity

HIDDEN = [

    "Diamond Princess", "Antigua and Barbuda", "Saint Lucia", "Jersey",

    "Liechtenstein", "Saint Vincent and the Grenadines", "Guernsey",

    "Maldives", "Andorra", "Seychelles", "Saint Martin", "Barbados",

    "Mauritius", "Cape Verde", "Dominica", "Grenada", "Saint Kitts and Nevis",

    "MS Zaandam", "Sao Tome and Principe", "Comoros"]



# Country check

for c in last["Country_Region"].unique():

    if not c in world["name"].values and not c in HIDDEN:

        print(f"Error {c} is not found."

              f" Please edit REPLACE_LIST.")

print("The rest of the country names that have not yet matched:\n",

      set(world.name.unique()) - set(last["Country_Region"].unique()))



world.name.unique()
last_c = last.groupby(['Country_Region']).sum()

last_c.sort_values(["Confirmed", 'Country_Region'], ascending=False)



# Guyana in France uses the same values as in the home country,

# because the latest format has been integrated into France.

last_c = last_c.append(last_c.loc["France"].rename("French Guiana"))

last_c
world_corona = pd.merge(world, last_c, left_on='name', right_on='Country_Region', how='left')

world_corona
# Fiji islands

world_corona = world_corona.set_index("name")

world_corona.at['Fiji', "continent"] = "World"

# Russia

world_corona.at["Russia", "continent"] = "World"
def plot(hue, maxval, title, area=world_corona, cmap='OrRd'):

    if maxval == 0:

        maxval = 1

    

    geoplot.choropleth(

        area, hue=hue,

        cmap=cmap, figsize=(16, 9), legend=True,

        norm=colors.LogNorm(vmin=1, vmax=maxval)

    )



    plt.title(title)



# log scale max

maxval = world_corona['Confirmed'].max()



# world plot

plot(world_corona["Confirmed"], maxval,

     f"Novel Coronavirus (COVID-19) Cases as of {LAST_UPDATE_STRING}",

     world_corona)

plt.savefig("Map_World.png", bbox_inches='tight',

                pad_inches=0.1, transparent=False, facecolor="white")

# option area plot

for continent in world['continent'].unique():

    if continent != 'Seven seas (open ocean)':

        area = world_corona[world_corona["continent"] == continent]

        plot(area["Confirmed"], maxval,

             f"Novel Coronavirus (2019-nCoV) Cases in {continent} as of {LAST_UPDATE_STRING}",

             area)

        plt.savefig(f"Map_{continent}.png", bbox_inches='tight',

                pad_inches=0.1, transparent=False, facecolor="white")

NDAY = -7

unique_date = np.sort(df["Date"].unique())



# select one week ago

n_days_ago = df[df["Date"] == unique_date[NDAY]].groupby(["Country_Region"]).sum()



# diff

last_c_week = last_c.sub(n_days_ago, fill_value=0).sort_values(["Confirmed", 'Country_Region'], ascending=False)

last_c_week
# merge map

world_corona_week = pd.merge(world, last_c_week, left_on='name', right_on='Country_Region', how='left')



geoplot.choropleth(

    world_corona_week, hue=world_corona_week['Confirmed'],

    cmap='coolwarm', figsize=(16, 9), legend=False,

    norm=colors.LogNorm(vmin=1, vmax=world_corona_week['Confirmed'].max())

)



plt.title(f"Countries with the high number of new cases of COVID-19 in the past week {LAST_UPDATE_STRING}")



plt.savefig("Map_Past_Weeks_World.png", bbox_inches='tight',

                pad_inches=0.1, transparent=False, facecolor="white")
