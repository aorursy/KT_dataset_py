# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import geopandas
from scipy import stats

import seaborn as sns
sns.set(style="darkgrid")

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Load the data
world_data = pd.read_csv("../input/countries of the world.csv", na_values='NaN')
# Get the information about the different attributes in the data
world_data.info()
# Get 5 random row entries
world_data.sample(5)
# Get names of regions
world_data['Region'].unique()

# I don't know what the 'NEAR EAST' region is. It is very vague and I can't pinpoint to what it represents.
# Clean Region names (remove whitespace) and set them as index
world_data['Region'] = world_data['Region'].str.strip()
world_data['Country'] = world_data['Country'].str.strip()
world_data['Region'].unique()
world_data.set_index('Region', inplace=True)
world_data.sample(5)
# Replace ',' with '.' in columns with numerical values
for column in world_data.columns:
    if (column != 'Country') and (world_data[column].dtype == 'object'):
        world_data[column] = world_data[column].str.replace(',', '.')
        world_data[column] = world_data[column].replace('NaN', np.NaN)
        world_data[column] = pd.to_numeric(world_data[column])
world_data.info()
# Get the EDA values for only complete data
complete_data = world_data.dropna()
complete_data.describe()
# Get the number of countires in different regions of the world
countries_per_region = pd.DataFrame(world_data.groupby(level=[0])['Country'].count())
countries_per_region = countries_per_region.reset_index()

# Plot Number of countries in different regions of the world
plt.figure(figsize = (12,6))
ax = sns.barplot(x='Region', y='Country', data=countries_per_region)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_ylabel('Number of Countries');      # ; to suppress the output for this line
# Get locations for different countries from geopandas library
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
world.info()
# Plot world map using geopandas
world = world.rename(columns={'name':'Country', 'geometry':'borders'}).set_geometry('borders');
ax = world.plot(figsize=(14,8));
# Merge original dataset with geopandas dataset
merged_data = world.merge(world_data, how='inner',on='Country').set_geometry('borders');
merged_data.head(5)
# Get the shape of the merged data frame
merged_data.shape
num_total_countries = 227
num_geopandas_countries = 177
num_intersection_countries = 151
# Calculate percentage for the available data
perc_data_geopandas = (num_geopandas_countries / num_total_countries) * 100
perc_data_avail = (num_intersection_countries / num_total_countries) * 100

print("% of data in geopandas: ", perc_data_geopandas)
print("% of data available after intersection of geopandas with given dataset: ", perc_data_avail)
# Plot world population
merged_data.plot(column='Population', cmap='OrRd', figsize=(14,8));
# Plot population for different regions
reset_world_data = world_data.reset_index()

plt.figure(figsize = (12,6))
ax = sns.barplot(x='Region', y='Population', data=reset_world_data)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90);  
# Show 10 countries with highest population
reset_world_data = reset_world_data.sort_values(by='Population', ascending=False)

plt.figure(figsize = (12,6))
ax = sns.barplot(x='Country', y='Population', data=reset_world_data[:10])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90);  
# Get a dataframe with entries that have non null birthrate values
birthrate_df = merged_data.loc[merged_data['Birthrate'].notnull(), :]
birthrate_df.sample(5)
# Plot birthrates on world map
birthrate_df.plot(column='Birthrate', cmap='OrRd', figsize=(16,16));
# Plot the spread of birthrates in different regions of the world 
reset_world_data = world_data.reset_index()

plt.figure(figsize = (12,6))
ax = sns.boxplot(x='Region', y='Birthrate', data=reset_world_data)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90);  
# Plot the highest 10 birthrates by countries
reset_world_data = reset_world_data.sort_values(by='Birthrate', ascending=False)

plt.figure(figsize = (12,6))
ax = sns.barplot(x='Country', y='Birthrate', data=reset_world_data[:10])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90); 
reset_world_data[(reset_world_data['Country'] == 'China') | (reset_world_data['Country'] == 'India')][['Country','Birthrate']]
# Calcualtion relations between different attributes
corr_matrix = world_data.corr()
corr_matrix
plt.figure(figsize = (13,13))

# Reverse the color map to see positive relations as darker colors
cmap = sns.cm.rocket_r

# Plot the correlation heatmap
sns.heatmap(data=corr_matrix, cmap=cmap, annot=True, cbar=True, square=True, fmt='.2f');
# Calculate correlation matrix for selected attributes only
columns_to_process = ['Population', 'Infant mortality (per 1000 births)','Area (sq. mi.)','GDP ($ per capita)', 
                      'Literacy (%)', 'Phones (per 1000)', 'Birthrate', 'Agriculture', 'Service']

new_corr_matrix = world_data[columns_to_process].corr()
new_corr_matrix
# Plot the heatmap for correlation matrix

plt.figure(figsize = (12,12));
# Reverse the color map to see positive relations as darker colors
cmap = sns.cm.rocket_r
# Plot the correlation heatmap
ax = sns.heatmap(data=new_corr_matrix, cmap=cmap, annot=True, cbar=True, fmt='.2f', square=True,);
# Plot population vs area
sns.relplot(x='Area (sq. mi.)', y='Population', hue='Region', height=5, aspect=2, data=reset_world_data);
# Sort the world data by area
sorted_area_world_data = reset_world_data.sort_values(by='Area (sq. mi.)', ascending=False)
sorted_area_world_data.sample(5)
# Try to look at the countries in the small patch on the bottom left in the previous graph
countries_in_small_patch = sorted_area_world_data[7:]

plt.figure(figsize = (12,6));
ax = sns.relplot(x='Area (sq. mi.)', y='Population', hue='Region', height=5, aspect=2,data=countries_in_small_patch);
# Try to fit a linear regression model

slope, intercept, r_value, p_value, std_err = stats.linregress(reset_world_data["Area (sq. mi.)"], reset_world_data["Population"])
print("Linear Regression - world data, r-2 value: ", r_value**2)

slope, intercept, r_value, p_value, std_err = stats.linregress(countries_in_small_patch["Area (sq. mi.)"], countries_in_small_patch["Population"])
print("Linear Regression - small patch, r-2 value: ", r_value**2)
# Plot population,area and a linear regression model fit for the entire world
plt.figure(figsize = (12,6))
ax = sns.regplot(x='Area (sq. mi.)', y='Population', data=reset_world_data, ci=50)
# Plot population,area and a linear regression model fit for the countries in the small patch
plt.figure(figsize = (12,6))
ax = sns.regplot(x='Area (sq. mi.)', y='Population', data=countries_in_small_patch)
# Plot GDP for countries on world map
merged_data.plot(column='GDP ($ per capita)', figsize=(16,8), legend=True);
# Plot range of GDP per region
plt.figure(figsize=(12,6))

ax = sns.boxplot(x='Region', y='GDP ($ per capita)', data=reset_world_data)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90);  
# Sort the world data according to GDP
sorted_gdp_world_data = reset_world_data.sort_values(by='GDP ($ per capita)', ascending=False)
sorted_gdp_world_data
# Plot a country's literacy rate against GDP
sns.relplot(x='GDP ($ per capita)', y='Literacy (%)', data=reset_world_data, kind='line', size=5, aspect=2, ci=0, legend=False);
# Plot a linear regression fit for a country's literacy rate against GDP
plt.figure(figsize=(12,6));
sns.regplot(x='GDP ($ per capita)', y='Literacy (%)', data=reset_world_data, order=1, robust=True, ci=None);
# Get GDP and Phone data without nan values and plot it
phone_data_without_nans = reset_world_data[['GDP ($ per capita)','Phones (per 1000)']].dropna()
sns.relplot(x='GDP ($ per capita)', y='Phones (per 1000)', data=phone_data_without_nans, kind='line', size=4, aspect=2, ci=0, legend=False);
# Plot a linear fit for GDP and Phones data
plt.figure(figsize=(12,6));
sns.regplot(x='GDP ($ per capita)', y='Phones (per 1000)', data=phone_data_without_nans, order=1, ci=None, robust=True);
slope, intercept, r_value, p_value, std_err = stats.linregress(phone_data_without_nans['GDP ($ per capita)'], phone_data_without_nans['Phones (per 1000)'])
print("Linear Regression - world data, slope: ", slope)
print("Linear Regression - world data, r-2 value: ", r_value**2)
# Get GDP, Birthrate, Infant Mortality data without nans and plot it
birthrate_data_without_nans = reset_world_data[['GDP ($ per capita)','Birthrate', 'Infant mortality (per 1000 births)']].dropna()

plt.figure(figsize=(12,6));
sns.lineplot(x='GDP ($ per capita)', y='Birthrate', data=birthrate_data_without_nans, ci=0, legend=False);
# Plot a linear regression fit for GDP and Birthrate
plt.figure(figsize=(12,6));
sns.regplot(x='GDP ($ per capita)', y='Birthrate', data=birthrate_data_without_nans, order=1, ci=None, robust=True);
# Plot Birthrate and infant Mortality together against GDP
plt.figure(figsize=(14,6))

# add birthrate on primary y-axis
ax = sns.lineplot(x='GDP ($ per capita)', y='Birthrate', data=birthrate_data_without_nans, ci=0, label='Birthrate')
ax.legend(loc='center right', bbox_to_anchor=(0.65, 0.955))

# add Infant mortality on secondary y-axis
ax2 = ax.twinx()
sns.lineplot(ax=ax2, x='GDP ($ per capita)', y='Infant mortality (per 1000 births)', data=birthrate_data_without_nans, color='green',label='Infant mortality (per 1000 births)', ci=0)
ax2.lines[0].set_linestyle("--")

# Change the plot settings for easy viewing
ax.yaxis.grid(which="major", linewidth=1)
ax2.yaxis.grid(which="major",linewidth=0.5, linestyle='--')

# To get the same scale, uncomment the below line and run again
# ax.set_yticks(np.linspace(0, ax2.get_yticks()[-1], len(ax2.get_yticks())));
# Get data for GDP, Service, Agriculture without any nan entries
eco_data_without_nans = reset_world_data[['GDP ($ per capita)','Service', 'Industry', 'Agriculture']].dropna()
eco_data_without_nans.sample(5)
plt.figure(figsize=(13,6))

# Plot the Service values
ax = sns.lineplot(x='GDP ($ per capita)', y='Service', data=eco_data_without_nans, ci=0, label="Service");
plt.figure(figsize=(13,6))

# Plot the Agriculture values
sns.lineplot(x='GDP ($ per capita)', y='Agriculture', data=eco_data_without_nans, ci=0, label='Agriculture');
plt.figure(figsize=(13,6))

# Plot the Industry values
sns.lineplot(x='GDP ($ per capita)', y='Industry', data=eco_data_without_nans, ci=0, label='Industry');
plt.figure(figsize=(13,6))

# Plot the Service values
ax = sns.lineplot(x='GDP ($ per capita)', y='Service', data=eco_data_without_nans, ci=0, label="Service")

ax2 = ax.twinx()
# plot the Industry values
sns.lineplot(ax=ax2, x='GDP ($ per capita)', y='Industry', color="r", data=eco_data_without_nans, ci=0, label="Industry")
# plot the Agriculture values
sns.lineplot(ax=ax2, x='GDP ($ per capita)', y='Agriculture', color="g", data=eco_data_without_nans, ci=0, label="Agriculture")

# Set legend
ax2.legend(loc='center right')
ax.legend(loc='center right', bbox_to_anchor=(1, 0.6))

# Set plot settings for easy viewing
ax.yaxis.grid(which="major", linewidth=1)
ax2.yaxis.grid(which="major",linewidth=0.5, linestyle='--')
ax2.set_yticks(np.linspace(0, ax.get_yticks()[-1], len(ax.get_yticks())));
# Plot linear regression fit for GDP and Service values
plt.figure(figsize=(12,6));
sns.regplot(x='GDP ($ per capita)', y='Service', data=eco_data_without_nans, order=1, ci=None, robust=True);
# Plot linear regression fit for GDP and Industry values
plt.figure(figsize=(12,6))
sns.regplot(x='GDP ($ per capita)', y='Industry', data=eco_data_without_nans, ci=None, truncate=True);
slope, intercept, r_value, p_value, std_err = stats.linregress(eco_data_without_nans['GDP ($ per capita)'], eco_data_without_nans['Industry'])
print("Linear Regression - world data, slope: ", slope)
print("Linear Regression - world data, r-2 value: ", r_value**2)
# Plot linear regression fit for GDP and Agriculture values
plt.figure(figsize=(12,6))
sns.regplot(x='GDP ($ per capita)', y='Agriculture', data=eco_data_without_nans, ci=None, truncate=True);
col_to_print = ['Region','Country', 'GDP ($ per capita)', 'Agriculture', 'Industry', 'Service']
# get the countries with highest Agriculture sector contributions to the GDP
print("Countries with highest Agriculture:")
reset_world_data.sort_values('Agriculture', ascending=False)[col_to_print].head(5)
# get the countries with highest Service sector contributions to the GDP
print("Countries with highest Service:")
reset_world_data.sort_values('Service', ascending=False)[col_to_print].head(5)
# get the countries with highest Industry sector contributions to the GDP
print("Countries with highest Industry:")
reset_world_data.sort_values('Industry', ascending=False)[col_to_print].head(5)