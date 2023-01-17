import pandas as pd

import geopandas as gpd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
temp_data=pd.read_csv("../input/daily-temperature-of-major-cities/city_temperature.csv", index_col="Country", usecols=['Country', 'City', 'Month','Day', 'Year', 'AvgTemperature'])

temp_data.index = temp_data.index.str.replace('US','United States')
past_temp_data =temp_data.loc[temp_data.Year.isin(['1995'])]

temp_1995=past_temp_data.loc[:,['City', 'Month', 'Day', 'AvgTemperature']]

temp_1995_filtered = temp_1995[temp_1995['AvgTemperature'] != -99.0] 

temp_1995_filtered.loc[temp_1995_filtered.City == 'Algiers']

temp_1995_filtered
future_temp_data =temp_data.loc[temp_data.Year.isin(['2019'])]

temp_2019=future_temp_data.loc[:,['City', 'Month','Day', 'AvgTemperature']]

temp_2019_filtered = temp_2019[temp_2019['AvgTemperature'] != -99.0] 

temp_2019_filtered
merged_temp_df=pd.merge(temp_2019_filtered, temp_1995_filtered, on=['Country', 'City', 'Month', 'Day'], how='inner')

merged_temp_df['Temp_Difference']=merged_temp_df['AvgTemperature_x']-merged_temp_df['AvgTemperature_y']

merged_temp_df.head()
temp_difference_df=merged_temp_df.groupby(['Country', 'City']).Temp_Difference.mean()

final_temp_df=temp_difference_df.sort_values(ascending=False)

final_temp_df=final_temp_df.to_frame()

print(final_temp_df.to_string())
plt.figure(figsize=(40,10))

plt.title("Top 10 Cities With Highest Increase in Temperature 1995 to 2019", fontsize=60)

plt.tick_params(labelsize=15)

plt.xlabel("Country, City", fontsize=20)

sns.barplot(x=final_temp_df.head(10).index, y=final_temp_df.head(10)['Temp_Difference'],palette='autumn')

plt.ylabel("Increase in Temperature", fontsize=20)
opp_final_temp_df=temp_difference_df.sort_values(ascending=True)

plt.figure(figsize=(40,10))

plt.title("Top 10 Cities With Highest Decrease in Temperature from 1995 to 2019", fontsize=60)

plt.tick_params(labelsize=15)

plt.xlabel("Country, City", fontsize=20)

sns.barplot(x=opp_final_temp_df.head(10).index, y=opp_final_temp_df.head(10),palette='winter')

plt.ylabel("Decrease in Temperature", fontsize=20)
map_df = gpd.read_file("../input/worldm/World_Map.shp")

map_dff=gpd.read_file("../input/wcities/a4013257-88d6-4a68-b916-234180811b2d202034-1-1fw6kym.nqo.shp")

map_dff=map_dff.rename(columns = {'CNTRY_NAME':'Country'})

map_dff=map_dff.rename(columns = {'CITY_NAME':'City'}) 

map_dff.loc[map_dff.City =='Fairbanks']
merged = pd.merge(map_dff,final_temp_df[['Temp_Difference']],on='City')

ax=map_df.plot(figsize=(20,12), color='none', edgecolor='gainsboro', zorder=3)

vmin, vmax = -4, 6

merged.plot(column='Temp_Difference',cmap='coolwarm', ax=ax)

sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmax=vmax, vmin=vmin))

sm._A = []

cbar = plt.colorbar(sm)

ax.set_title('Average Temperature Increase from 1995 to 2020', fontdict={'fontsize': '40', 'fontweight' : '3'})