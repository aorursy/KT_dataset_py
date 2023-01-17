# Imports
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt # As we are on it let's import packages that will enable us to plot things.
import seaborn as sns
temp_df = \
    pd.read_csv("../input/france_temperatures_geopositions.csv", sep=",")[["Latitude", "Longitude", "Mois", "NORTAV"]] \
      .rename(columns = {"NORTAV": "Temperature", "Mois": "Month", "Latitude": "latitude", "Longitude": "longitude"})

temp_df.head(3)
# We also need to load our mistery city temperature data from the image !
mystery_city_temp_df = \
    pd.DataFrame(
        columns=["Month", "Temperature_Mystery_Location"],
        data=[
            [1, 5.6],
            [2, 5.9],
            [3, 8.1],
            [4, 10.2],
            [5, 13],
            [6, 15.7],
            [7, 17.4],
            [8, 17.4],
            [9, 15.9],
            [10, 12.5],
            [11, 8.7],
            [12, 6.5]
        ]
    )
# The mystery city temperatures info
mystery_city_temp_df
# A bit less straightforward here, to add the longitude/latitude information
# we'll join our city_population dataframe with a city_geoposition dataframe on code_insee, a unique ID per city

# Load
city_pop_df = \
    pd.read_csv("../input/france_city_populations.csv", sep=",")[["Nom de la commune", "Population totale", "Code département", "Code commune"]]\
      .rename(columns = {"Nom de la commune": "city_name", "Population totale": "population"})

city_geoposition_df = \
    pd.read_csv("../input/france_city_geopositions.csv", sep=",")[["latitude", "longitude", "code_insee"]] \
      .rename(columns = {"codes_postaux": "postal_code"})

# Create or format the joining column according to table
city_pop_df["code_insee"] = \
    city_pop_df["Code département"].apply(lambda x:x.zfill(2)) + city_pop_df["Code commune"].apply(lambda x:'%03d'%x)

city_geoposition_df["code_insee"] = city_geoposition_df["code_insee"].apply(lambda x:'%05d'%x)

# Joining on city names can be tricky because of special characters, one has to be cautious
city_pop_geoposition_df = \
    city_pop_df\
        .merge(city_geoposition_df, on="code_insee", how="inner") \
        .dropna()

# Formatting of the longitude and latitude columns and filtering out some bad values 
# (France max longitude should be around 10)
city_pop_geoposition_df["longitude"] = \
    city_pop_geoposition_df["longitude"].apply(lambda x:round(float(x), 2) if x!='-' else 0)

city_pop_geoposition_df["latitude"] = \
    city_pop_geoposition_df["latitude"].apply(lambda x:round(float(x), 2) if x!='-' else 0)

city_pop_geoposition_df = city_pop_geoposition_df[city_pop_geoposition_df["longitude"] < 10]
city_pop_geoposition_df = city_pop_geoposition_df.drop_duplicates()[["city_name", "population", "latitude", "longitude"]]

# We should normally check for duplicates and missing values.
# In practice this won't pose any problem here, but could have
# The dataframe that contains population by city along their geoposition 
city_pop_geoposition_df.head(3)
# Merge both dataframes
temp_df_merged = temp_df.merge(mystery_city_temp_df, on="Month")

# Compute metric per position per month
temp_df_merged["diff_squared_temperatures"] = \
    (temp_df_merged["Temperature"] - temp_df_merged["Temperature_Mystery_Location"])**2 /12

# Aggregate per position
temp_metric_df = temp_df_merged\
    .groupby(["latitude", "longitude"], as_index=False)\
    .diff_squared_temperatures.agg(['mean'])\
    .reset_index() \
    .rename(columns={"mean": "mean_squared_diff"})
temp_metric_df.sort_values(["mean_squared_diff"]).reset_index().head(5)
sns.set(rc={'figure.figsize':(6,5)})
plt.scatter(
    temp_metric_df.longitude,
    temp_metric_df.latitude,
    c=temp_metric_df.mean_squared_diff
)
plt.colorbar()
plt.title("Average squared difference in temperature with the mystery city")
plt.show()
sns.set(rc={'figure.figsize':(30,10)})
sns.distplot(temp_metric_df["mean_squared_diff"], bins=500)
plt.title("Distribution of mean_squared_difference of temperature across all geolocalised points", fontsize=20)
sns.distplot(temp_metric_df[temp_metric_df["mean_squared_diff"] < 2]["mean_squared_diff"], bins=500)
plt.title("Distribution of mean_squared_difference of temperature across all geolocalised points - zoomed in", fontsize=20)
sns.set(rc={'figure.figsize':(15,13)})

fig, ax = plt.subplots(2, 2)
s = ax[0, 0].scatter(temp_metric_df.longitude, temp_metric_df.latitude, c=temp_metric_df.mean_squared_diff, vmax=0.5)
fig.colorbar(s, ax=ax[0, 0])

s = ax[0, 1].scatter(temp_metric_df.longitude, temp_metric_df.latitude, c=temp_metric_df.mean_squared_diff, vmax=0.2)
fig.colorbar(s, ax=ax[0, 1])

s = ax[1, 0].scatter(temp_metric_df.longitude, temp_metric_df.latitude, c=temp_metric_df.mean_squared_diff, vmax=0.06)
fig.colorbar(s, ax=ax[1, 0])

s = ax[1, 1].scatter(temp_metric_df.longitude, temp_metric_df.latitude, c=temp_metric_df.mean_squared_diff, vmax=0.03)
fig.colorbar(s, ax=ax[1, 1])


fig.suptitle('Mean Squared difference across all locations for different colorscales', fontsize=16)

plt.show()
# Compute the percentile information
temp_metric_df['percentile_temp'] = \
    temp_metric_df["mean_squared_diff"].rank(pct=True)
city_pop_geoposition_df.shape[0]
# A look at the distribution (with removed outliers)
sns.set(rc={'figure.figsize':(30,10)})
sns.distplot(city_pop_geoposition_df[city_pop_geoposition_df["population"] < 100000]["population"], bins=500)
plt.title("Distribution of population counts across all French cities", fontsize=20)
city_pop_geoposition_df["population_diff"] = abs(city_pop_geoposition_df["population"] - 45000)

# Computes the percentile information
city_pop_geoposition_df["percentile_population"] = \
    city_pop_geoposition_df["population_diff"].rank(pct=True)

# Let's represent the 100 cities which population count are the closest to 45000
city_pop_geoposition_df_head = city_pop_geoposition_df.sort_values(["population_diff"]).reset_index().head(100)

sns.set(rc={'figure.figsize':(8,7)})
plt.scatter(city_pop_geoposition_df.longitude, city_pop_geoposition_df.latitude, s=10, c='grey')
plt.scatter(city_pop_geoposition_df_head.longitude, city_pop_geoposition_df_head.latitude, s=100, c=city_pop_geoposition_df_head.population_diff)
plt.colorbar()
plt.title("Population count difference with the mystery city for the 100 closest cities")
plt.show()
# To join the info we have to make a join with longitude and latitude information.
# This is not a straightforward thing to do due to the different precisions the datasets might have
# Hence, we'll join information by selecting the closest point neighbours for each line of dataframe A 
# to each line of dataframe B

# This is a costly operation so we'll do on the shortlist we could have created from the population analysis above,
# keeping 2000 cities is far enough
potential_cities_df = city_pop_geoposition_df.sort_values(["population_diff"]).reset_index().head(2000)

def euc_dist(lata, latb, longa, longb):
    return np.sqrt((lata - latb)**2 + (longa - longb)**2)

# prepare the cross join
potential_cities_df['key'] = 0
temp_metric_df['key'] = 0

# perform cross join and computes distance
crossed_info = potential_cities_df.merge(temp_metric_df, on="key")
crossed_info["euc_distance"] = euc_dist(crossed_info["latitude_x"], crossed_info["latitude_y"], crossed_info["longitude_x"], crossed_info["longitude_y"])

# Select for each city the closest temperature information
crossed_info = crossed_info.sort_values('euc_distance', ascending=True).drop_duplicates(['city_name'])
# That's it ! We now have the two metric for our shortlisted cities.
# (note that we associate them with their respective percentiles)
crossed_info[["city_name", "population_diff", "mean_squared_diff", "percentile_population", "percentile_temp"]].head(5) 
crossed_info_filtered = crossed_info[crossed_info["percentile_population"] < 0.05]
crossed_info_filtered = crossed_info_filtered[crossed_info_filtered["percentile_temp"] < 0.05]
crossed_info_filtered = crossed_info_filtered.reset_index()

sns.set(rc={'figure.figsize':(15,13)})
p = sns.regplot(
    data=crossed_info_filtered,
    x="percentile_population",
    y="percentile_temp",
    fit_reg=False,
    scatter_kws={'s':200},
    color="skyblue"
)


# add annotations one by one with a loop
for line in range(0, crossed_info_filtered.shape[0]):
     p.text(
         crossed_info_filtered.percentile_population[line] + 0.0005,
         crossed_info_filtered.percentile_temp[line],
         crossed_info_filtered.city_name[line],
         horizontalalignment='left',
         size='small',
         color='grey',
     )
        
plt.xlim(0, 0.05)
plt.ylim(0, 0.05)

plt.title("Top shortlisted cities, given their population and temperature features percentiles")
crossed_info_filtered[crossed_info_filtered["city_name"] == "Auray"][["city_name", "population", "population_diff", "mean_squared_diff", "percentile_population", "percentile_temp"]]
crossed_info_filtered[crossed_info_filtered["city_name"] == "Lanester"][["city_name", "population", "population_diff", "mean_squared_diff", "percentile_population", "percentile_temp"]]
crossed_info_filtered[crossed_info_filtered["city_name"] == "Saint-Herblain"][["city_name", "population", "population_diff", "mean_squared_diff", "percentile_population", "percentile_temp"]]
print("On average, Saint-Herblain's monthly average temperature is %s°C away from our hint distribution" % round(np.sqrt(0.051807),2))
