import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv("../input/epa_co_daily_summary.csv")
data.columns

states = data.state_code.unique()
print(sorted(states))
data.loc[:,['state_code', 'state_name']].drop_duplicates()
len(data.site_num.unique())
data.parameter_code.unique()
data.poc.unique()
data.datum.unique()
data.parameter_name.unique()
data.units_of_measure.unique()
data.event_type.unique()
filtered_data = data.loc[:,['poc', 'latitude', 'longitude', 'sample_duration', 'pollutant_standard', 'date_local', 'event_type', 'observation_count', 'observation_percent', 'arithmetic_mean', 'first_max_value', 'first_max_hour', 'aqi', 'method_code', 'data_of_last_change']]
location_lat_longs = filtered_data.loc[:,['latitude', 'longitude']].drop_duplicates()
import folium

location_lat_longs = filtered_data.loc[:,['latitude', 'longitude']].drop_duplicates()
map_of_locations = folium.Map(location=[39, -98.5], zoom_start=3)

for _, location in location_lat_longs.iterrows():
    folium.Marker(location).add_to(map_of_locations)

map_of_locations
newly_filtered_data = data.loc[:,['state_name', 'poc', 'latitude', 'longitude', 'sample_duration', 'pollutant_standard', 'date_local', 'event_type', 'observation_count', 'observation_percent', 'arithmetic_mean', 'first_max_value', 'first_max_hour', 'aqi', 'method_code', 'data_of_last_change']]
rhode_island_data = newly_filtered_data.loc[lambda df: df.state_name == "Rhode Island", :]
ri_time = rhode_island_data.sort_values("date_local", ascending=True)
ri_time
import matplotlib.pyplot as plt
plt.style.use('ggplot')

ri_8h_data = ri_time.loc[lambda df: df.sample_duration == "8-HR RUN AVG END HOUR", :]
ri_city_8h_data = ri_8h_data.loc[lambda df: df.latitude < 41.83, :]
ri_rural_8h_data = ri_8h_data.loc[lambda df: df.latitude >= 41.83, :]

ri_city_8h_data = ri_city_8h_data.sort_values("date_local", ascending=True)
ri_rural_8h_data = ri_rural_8h_data.sort_values("date_local", ascending=True)

plt.plot(ri_city_8h_data.date_local, ri_city_8h_data.arithmetic_mean, color='red')
plt.plot(ri_rural_8h_data.date_local, ri_rural_8h_data.arithmetic_mean, color='blue')
# ri_rural_8h_data.arithmetic_mean.plot()
ri_city_1990 = ri_city_8h_data.loc[lambda df: df.date_local < "1991", :]
ri_rural_1990 = ri_rural_8h_data.loc[lambda df: df.date_local < "1991", :]
plt.plot(ri_city_1990.date_local, ri_city_1990.arithmetic_mean)
plt.plot(ri_rural_1990.date_local, ri_rural_1990.arithmetic_mean, color='blue')
ri_city_1995 = ri_city_8h_data.loc[lambda df: df.date_local > "1994", :]
ri_city_1995 = ri_city_1995.loc[lambda df: df.date_local < "1996", :]
ri_rural_1995 = ri_rural_8h_data.loc[lambda df: "1994" < df.date_local, :]
ri_rural_1995 = ri_rural_1995.loc[lambda df: "1996" > df.date_local, :]
plt.plot(ri_city_1995.date_local, ri_city_1995.arithmetic_mean)
plt.plot(ri_rural_1995.date_local, ri_rural_1995.arithmetic_mean, color='blue')
ri_rural_1995.arithmetic_mean.isnull().sum()
len(ri_rural_1995.arithmetic_mean)
len(ri_city_1995.arithmetic_mean)