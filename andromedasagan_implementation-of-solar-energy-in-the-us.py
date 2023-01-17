import os              # accessing directory structure

import numpy as np     # linear algebra

import pandas as pd    

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import seaborn as sns

import scipy

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

# libraries for maps

from mpl_toolkits.basemap import Basemap as Basemap

from matplotlib.colors import rgb2hex

from matplotlib.patches import Polygon



# %matplotlib inline

# pyplt.style.use('seaborn-whitegrid')

# sns.set_context("poster")

print("Libraries imported.")
# create data frames from the original file that do not produce errors

dt1 = pd.read_csv('../input/deepsolar_tract.csv', nrows=6077)

dt2 = pd.read_csv('../input/deepsolar_tract.csv', skiprows =6119, names=["Unnamed: 0", "tile_count", "solar_system_count", "total_panel_area", "fips", "average_household_income", "county", "education_bachelor", "education_college", "education_doctoral", "education_high_school_graduate", "education_less_than_high_school", "education_master", "education_population", "education_professional_school", "employed", "gini_index", "heating_fuel_coal_coke", "heating_fuel_electricity", "heating_fuel_fuel_oil_kerosene", "heating_fuel_gas", "heating_fuel_housing_unit_count", "heating_fuel_none", "heating_fuel_other", "heating_fuel_solar", "land_area", "per_capita_income", "population", "population_density", "poverty_family_below_poverty_level", "poverty_family_count", "race_asian", "race_black_africa", "race_indian_alaska", "race_islander", "race_other", "race_two_more", "race_white", "state", "total_area", "unemployed", "water_area", "education_less_than_high_school_rate", "education_high_school_graduate_rate", "education_college_rate", "education_bachelor_rate", "education_master_rate", "education_professional_school_rate",	"education_doctoral_rate", "race_white_rate", "race_black_africa_rate", "race_indian_alaska_rate", "race_asian_rate", "race_islander_rate", "race_other_rate", "race_two_more_rate", "employ_rate", "poverty_family_below_poverty_level_rate", "heating_fuel_gas_rate", "heating_fuel_electricity_rate", "heating_fuel_fuel_oil_kerosene_rate", "heating_fuel_coal_coke_rate", "heating_fuel_solar_rate", "heating_fuel_other_rate", "heating_fuel_none_rate", "solar_panel_area_divided_by_area", "solar_panel_area_per_capita", "tile_count_residential", "tile_count_nonresidential", "solar_system_count_residential", "solar_system_count_nonresidential", "total_panel_area_residential", "total_panel_area_nonresidential", "median_household_income", "electricity_price_residential", "electricity_price_commercial", "electricity_price_industrial", "electricity_price_transportation", "electricity_price_overall", "electricity_consume_residential", "electricity_consume_commercial", "electricity_consume_industrial", "electricity_consume_total", "household_count", "average_household_size", "housing_unit_count", "housing_unit_occupied_count", "housing_unit_median_value", "housing_unit_median_gross_rent", "lat", "lon", "elevation", "heating_design_temperature", "cooling_design_temperature", "earth_temperature_amplitude", "frost_days", "air_temperature", "relative_humidity", "daily_solar_radiation", "atmospheric_pressure", "wind_speed", "earth_temperature", "heating_degree_days", "cooling_degree_days", "age_18_24_rate", "age_25_34_rate", "age_more_than_85_rate", "age_75_84_rate", "age_35_44_rate", "age_45_54_rate", "age_65_74_rate", "age_55_64_rate", "age_10_14_rate", "age_15_17_rate", "age_5_9_rate", "household_type_family_rate", "dropout_16_19_inschool_rate", "occupation_construction_rate", "occupation_public_rate", "occupation_information_rate", "occupation_finance_rate", "occupation_education_rate", "occupation_administrative_rate", "occupation_manufacturing_rate", "occupation_wholesale_rate", "occupation_retail_rate", "occupation_transportation_rate", "occupation_arts_rate", "occupation_agriculture_rate", "occupancy_vacant_rate", "occupancy_owner_rate", "mortgage_with_rate", "transportation_home_rate", "transportation_car_alone_rate", "transportation_walk_rate", "transportation_carpool_rate", "transportation_motorcycle_rate", "transportation_bicycle_rate", "transportation_public_rate", "travel_time_less_than_10_rate", "travel_time_10_19_rate", "travel_time_20_29_rate", "travel_time_30_39_rate", "travel_time_40_59_rate", "travel_time_60_89_rate", "health_insurance_public_rate", "health_insurance_none_rate", "age_median", "travel_time_average", "voting_2016_dem_percentage", "voting_2016_gop_percentage", "voting_2016_dem_win", "voting_2012_dem_percentage", "voting_2012_gop_percentage", "voting_2012_dem_win", "number_of_years_of_education", "diversity", "number_of_solar_system_per_household", "incentive_count_residential", "incentive_count_nonresidential", "incentive_residential_state_level", "incentive_nonresidential_state_level", "net_metering", "feedin_tariff", "cooperate_tax", "property_tax", "sales_tax", "rebate", "avg_electricity_retail_rate"])



# append data frames into a dataframe solardata

solardata = dt1.append(dt2)
# replace NA values with mean 

solardata.fillna(solardata.mean())
# return total number of rows and columns 

print("The dataset has " + str(solardata.shape[0]) + " rows and " + str(solardata.shape[1]) + " columns.")



#return the first 5 lines of the dataset

solardata.head()
print("Histogram of Solar Panel Counts")

fig, ax = plt.subplots(figsize=(20, 7))

tile_count = solardata['tile_count']

plt.hist(tile_count, 35, range=[0, 400], facecolor='goldenrod', align='mid')

plt.show()
print("Histogram of Solar Panel Counts - A Closer Look")

tile_count = solardata['tile_count']

fig, ax = plt.subplots(figsize=(20, 7))

plt.hist(tile_count, 30, range=[20, 200], facecolor='goldenrod', align='mid')

plt.show()
print("Histogram of Solar Panel Area per Capita")

area_capita = solardata['solar_panel_area_per_capita']

fig, ax = plt.subplots(figsize=(20, 5))

plt.hist(area_capita, 40, range=[0, 1], facecolor='rosybrown', align='mid')

plt.show()
print("Histogram of Solar Panel Area per Capita - A Closer Look")

area_capita = solardata['solar_panel_area_per_capita']

fig, ax = plt.subplots(figsize=(20, 8))

plt.hist(area_capita, 35, range=[0.1, 0.6], facecolor='rosybrown', align='mid')

plt.show()
print("Analysis of Bar Plot")

print()



# defining variables to display and for plotting

total_tile = int(solardata['tile_count'].sum())

print("Total Solar Panels:", total_tile)

total_system = int(solardata['solar_system_count'].sum())

print("Total Solar Panel Systems:", total_system)



avg_tile_count = total_tile/72495

avg_tile_per_system = total_tile/ total_system

print("Average number of solar panels:", round(avg_tile_count))

print("Average number of panels per solar panel system:", avg_tile_per_system)



print()

total_rtile = int(solardata['tile_count_residential'].sum())

print("Total Solar Panels for Residential Purposes:", total_rtile)

total_nrtile = int(solardata['tile_count_nonresidential'].sum())

print("Total Solar Panels for Non-Residential Purposes*:", total_nrtile)

print()

total_rsystem = int(solardata['solar_system_count_residential'].sum())

print("Total Solar Systems for Residential Purposes:", total_rsystem)

total_nrsystem = int(solardata['solar_system_count_nonresidential'].sum())

print("Total Solar Panels for Non-Residential Purposes*:", total_nrsystem)

print()



# begin plotting 

# set width of bar

barWidth = 0.2

 

# set height of bar based on data 

barstotal = [total_tile, total_system]

barsres = [total_rtile, total_rsystem]

barsnonres = [total_nrtile, total_nrsystem]

 

# Set position of bar on x axis

r1 = np.arange(len(barstotal))

r2 = [x + barWidth for x in r1]

r3 = [x + barWidth for x in r2]



fig, ax = plt.subplots(figsize=(15, 8))

# Make plot for each category of bars 

plt.bar(r1, barstotal, color='goldenrod', width=barWidth, edgecolor='white', label='total')

plt.bar(r2, barsres, color='cadetblue', width=barWidth, edgecolor='white', label='residential')

plt.bar(r3, barsnonres, color='darkseagreen', width=barWidth, edgecolor='white', label='nonresidential')

 

# Add xticks on the middle of the group bars

plt.xlabel('Residential and Non-Residential Use of Solar Energy', fontweight='bold')

plt.xticks([r + barWidth for r in range(len(barstotal))], ["Solar Panels", "Solar Systems"])

 

# Create legend & display plot

plt.legend()

plt.show()

#define columns to find correlations  

energyenv_quant_data = ["total_panel_area", "total_panel_area_residential", "total_panel_area_nonresidential","electricity_price_overall", "avg_electricity_retail_rate", "electricity_consume_total", "incentive_count_residential", "incentive_count_nonresidential", "heating_fuel_gas_rate", "heating_fuel_solar_rate","daily_solar_radiation",  "air_temperature",]



fig, axe = plt.subplots(figsize=(14,14))

sns.set_context("poster")

# RdYlBu_r

sns.set(font_scale=1.3)

corrmap = sns.color_palette("BrBG", 100)

sns.heatmap(solardata[energyenv_quant_data].corr(),annot=True, fmt='.2f',linewidths=1,cmap = corrmap)
fig, axe = plt.subplots(figsize=(8, 8))

sns.regplot(x=solardata["incentive_count_residential"], y=solardata["total_panel_area_residential"], fit_reg=True)



socioecon_quant_data = ["total_panel_area", "total_panel_area_residential", "median_household_income","number_of_years_of_education", "age_median", "poverty_family_count", "per_capita_income", "population_density", "employ_rate", "gini_index", "diversity", "voting_2016_dem_percentage", 'voting_2016_gop_percentage']



fig, axe = plt.subplots(figsize=(15, 15))

sns.set_context("poster")



sns.set(font_scale=1.3)

colmap2 = sns.color_palette("coolwarm", 100)

sns.heatmap(solardata[socioecon_quant_data].corr(),annot=True, fmt='.2f',linewidths=1,cmap = colmap2)
print("Linear Regression Plot of Median Household Income and Total Residential Panel Area")

fig, axe = plt.subplots(figsize=(8, 8))

sns.regplot(x=solardata["median_household_income"], y=solardata["total_panel_area_residential"], fit_reg=True)


