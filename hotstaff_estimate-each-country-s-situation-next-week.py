# import

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import geopandas as gpd   # geopandas

import geoplot

import matplotlib.pyplot as plt

import matplotlib.colors as colors
df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")

df.info()
threshold = 2

unique_date = np.sort(df["Date"].unique())

last_confirmed = df[df["Date"] == unique_date[-1]].groupby(["Country/Region"]).sum()["ConfirmedCases"]

one_week_ago = df[df["Date"] == unique_date[-(7 + 1)]].groupby(["Country/Region"]).sum()["ConfirmedCases"]

two_weeks_ago = df[df["Date"] == unique_date[-(14 + 1)]].groupby(["Country/Region"]).sum()["ConfirmedCases"]



diff_one = last_confirmed.sub(one_week_ago, fill_value=0).astype(int)

diff_two = one_week_ago.sub(two_weeks_ago, fill_value=0).astype(int)

diff_one = diff_one[diff_one > 1]

diff_two = diff_two[diff_two > 1]

diff = pd.concat([diff_one, diff_two], axis=1, sort=False,

                 keys=("New confirmed last week (A)", "New confirmed two weeks ago (B)"))

diff["Growth rate (A/B)"] = (diff["New confirmed last week (A)"] /

                             diff["New confirmed two weeks ago (B)"]).round(1)

diff = diff[diff["Growth rate (A/B)"] > threshold].sort_values("Growth rate (A/B)", ascending=False)

print(f"The surge has been observed in {len(diff)} countries")

diff.style.background_gradient(cmap="YlGn")

diff["New confirmed case one week later (P)"] = diff["Growth rate (A/B)"] * diff["New confirmed last week (A)"]

diff["New confirmed case one week later (P)"] = diff["New confirmed case one week later (P)"].round(0).astype(int)

diff = diff.sort_values("New confirmed case one week later (P)", ascending=False)

diff[["Growth rate (A/B)", "New confirmed case one week later (P)"]].style.background_gradient(cmap="RdPu")
# Migrate geopandas world



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

    "Others": "Cruise Ship",

    "The Gambia": "Gambia",

    "Gambia, The": "Gambia",

    "Bahamas, The": "Bahamas",

    "Cabo Verde": "Cape Verde",

    "East Timor": "Timor-Leste"

}

diff_c = diff.reset_index().rename(columns={"index": "Country/Region"})

diff_c["Country/Region"] = diff_c["Country/Region"].replace(REPLACE_LIST)



diff_c = diff_c.groupby("Country/Region").sum()[["Growth rate (A/B)", "New confirmed case one week later (P)"]]

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

corona_gdf = pd.merge(world, diff_c, left_on='name', right_on='Country/Region', how='left')

corona_gdf



def plot(hue, maxval, title, area):

    if maxval == 0:

        maxval = 1

    

    geoplot.choropleth(

        area, hue=hue,

        cmap='coolwarm', figsize=(16, 9), legend=True,

        norm=colors.LogNorm(vmin=1, vmax=maxval)

    )



    plt.title(title)



# log scale max

maxval = corona_gdf["New confirmed case one week later (P)"].max()



# world plot

plot(corona_gdf["New confirmed case one week later (P)"], maxval,

     f"New patient next week, forecast as of {unique_date[-1]}",

     corona_gdf)

plt.savefig("Map_World.png", bbox_inches='tight',

                pad_inches=0.1, transparent=False, facecolor="white")