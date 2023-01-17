# Import statements



import pandas as pd

import geojson

import folium
# Load the metal data



df_metal = pd.read_csv("../input/metal-by-nation/metal_bands_2017.csv", encoding="latin-1")

df_metal.head()
# Filter metal data columns and check for duplicates



df_metal = df_metal.filter(items=["band_name", "origin"])

df_metal.duplicated().any()
# Remove duplicates from the metal data and check for missing values



df_metal.drop_duplicates(inplace=True)

df_metal.isnull().any()
# Drop missing values from the metal data and replace acronyms with full names



df_metal.dropna(axis=0, inplace=True)

df_metal["origin"].replace(to_replace="USA", value="United States of America", inplace=True)

df_metal["origin"].replace(to_replace="UAE", value="United Arab Emirates", inplace=True)

df_metal.head()
# Load the population data



df_world = pd.read_csv("../input/metal-by-nation/world_population_1960_2015.csv", encoding='latin-1')

df_world.head()
# Filter columns, replace acronyms and rename columns in population data



df_world = df_world.filter(["Country Name", "2015"])

df_world.replace(to_replace="United States", value="United States of America", inplace=True)

df_world.rename(columns = {"2015":"Population", "Country Name":"Country"}, inplace = True)

df_world.head()
# Create a list of origin countries (preserve frequencies and split joint entries)



origins = []



for i in range(0, len(df_metal["origin"])):

    if "," not in df_metal.iloc[i, 1]:

        origins.append(df_metal.iloc[i, 1])

    else:

        vals = df_metal.iloc[i, 1].rsplit(", ")

        for val in vals:

            origins.append(val)





# Create a dataframe for the set of origin countries

            

origin_set = set(origins)

country_bands = pd.DataFrame(origin_set, columns=["Country"])





# Add the count of bands from each origin country



country_bands["Bands count"] = 0



for i in range(0, len(country_bands)):

    country_bands.iloc[i, 1] = origins.count(country_bands.iloc[i, 0])



country_bands.head()
# Add population data to country_bands dataframe



country_bands = country_bands.merge(df_world, how="inner", on=["Country"])

country_bands.head()
# Check all countries have a band count and population value



country_bands.isnull().any()
# Add a "Bands per 100k" column (no. of bands from each country per 100,000 people in population)



country_bands["Bands per 100k"] = ((country_bands["Bands count"] / country_bands["Population"]) * 100000)

country_bands.head()
# Load the countries geojson



with open("../input/country-outlines/countries.geo.json") as f:

    countries = geojson.load(f)
# Generate a map to show the total number of metal bands originating in each country



metal_map1 = folium.Map(location=[51.5017963261, 0.00187999248], zoom_start=2, tiles="cartodbdark_matter")



folium.Choropleth(geo_data=countries, name="choropleth", data=country_bands, columns=["Country", "Bands count"],

                  key_on="properties.admin", fill_color="Reds", nan_fill_color="grey", fill_opacity=0.8).add_to(metal_map1)

metal_map1
# Generate a map to show the number of metal bands originating in each country per 100,000 people



metal_map2 = folium.Map(location=[51.5017963261, 0.00187999248], zoom_start=2, tiles="cartodbdark_matter")



folium.Choropleth(geo_data=countries, name="choropleth", data=country_bands, columns=["Country", "Bands per 100k"],

                  key_on="properties.admin", fill_color="Reds", nan_fill_color="grey", fill_opacity=0.8).add_to(metal_map2)

metal_map2