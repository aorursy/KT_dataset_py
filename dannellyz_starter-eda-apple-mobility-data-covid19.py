import pandas as pd



#Read file and get head

apple_mobility_df = pd.read_csv("../input/apple-mobility-trends-updated-daily/Apple_Mobility_2020-04-13.csv")

apple_mobility_df.drop("Unnamed: 0", axis=1, inplace=True)

apple_mobility_df.head()
#Break into countires/regions and cities

geo_mask = apple_mobility_df["geo_type"] == "country/region"

mobility_countries = apple_mobility_df[geo_mask]

mobility_cities = apple_mobility_df[~geo_mask]

print("There are a total of {} countires and {} cities with provided mobility data.".format(len(mobility_countries),

                                                                                           len(mobility_cities)))
def get_trans_count(df):

    name = df["geo_type"].iloc[0]

    return df["transportation_type"].value_counts().rename(str(name))

transport_types_count = pd.concat([get_trans_count(mobility_countries), get_trans_count(mobility_cities)], axis=1, sort=False)

transport_types_count
#Melt into timeseries

id_vars = ["geo_type", "region","transportation_type","lat", "lng", "population"]

mobility_countries_melted = mobility_countries.melt(id_vars=id_vars,var_name="Date",value_name="pct_of_baseline")

mobility_cities_melted = mobility_cities.melt(id_vars=id_vars,var_name="Date",value_name="pct_of_baseline")

mobility_cities_melted.head()
import plotly.express as px

#Make list of any cities to plot

to_show = ["Atlanta", "Athens", "London"]



#Plot

df = mobility_cities_melted[mobility_cities_melted["region"].isin(to_show)]

fig = px.line(df, x="Date", y="pct_of_baseline", color="transportation_type",

              line_group="region", hover_name="region")

fig.show()
from simple_folium import simple_folium

simple_folium(mobility_cities, "lat", "lng", ["region","transportation_type"], "Apple Mobility Data")